import os
import sys
import json
import random
from collections import defaultdict
from collections import OrderedDict, Counter

import h5py as h5py
import numpy as np
from fairseq.data.dictionary import Dictionary

from torch.utils.data._utils.collate import default_collate

from util.data import *
from util.dep_tree import get_actions_dfs_new, get_actions_bfs_new, get_actions_linear, _add_extra_symbol, \
    get_word_gen_order_dfs, _add_sentence_end_point
from util.vis_tree import *

from config import *
from config import _activitynet_resnet_bn_feat_dir


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.counter = {'hit': 0, 'miss': 0}

    def get(self, key):
        # if (self.counter['hit'] + self.counter['miss']) % 10000 == 0:
        #     print(self.counter)
        if key not in self.cache:
            self.counter['miss'] += 1
            return None
        else:
            self.counter['hit'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def convert_words_to_id(dictionary: Dictionary, words, max_length, dtype=np.int32):
    words = [dictionary.bos_index] + [dictionary.index(i) for i in words]
    fixed_length = max_length + 2
    _l = min(len(words), fixed_length - 1)
    id_fixed_len = np.array([dictionary.pad_index] * fixed_length, dtype=dtype)
    id_fixed_len[:_l] = words[:_l]  # the last id must be <end>
    id_fixed_len[_l] = dictionary.eos_index
    return id_fixed_len, _l + 1


class CaptionDataset(CustomDataset):
    def __init__(self, dataset_name, dictionary, max_sent_len, feat_type='fc', tree_gen_mode=None, **kwargs):
        super().__init__()

        self.dataset_name, self.dictionary, self.max_sent_len, self.feat_type = \
            dataset_name, dictionary, max_sent_len, feat_type

        if self.feat_type is None or len(self.feat_type) == 0:    # default feat_type
            if self.dataset_name == 'msvd':
                self.feat_type = 'irv2+i3d'     # InceptionResNetv2 + I3D(RGB)
            elif self.dataset_name == 'msrvtt':
                self.feat_type = 'irv2+i3d'
            elif self.dataset_name == 'charades':
                self.feat_type = 'resnet152'
            elif self.dataset_name == 'activitynet':
                self.feat_type = 'c3d'

        if self.dataset_name == 'coco':
            d = json.load(open(coco_ann_path, 'r'))
            self.dep_file = coco_dep_dir
        elif self.dataset_name == 'flickr30k':
            d = json.load(open(flickr30k_ann_path, 'r'))
            self.dep_file = flickr30k_dep_dir
        else:
            ann_path = os.path.join(data_path, dataset_name, 'annotation_{}.json'.format(dataset_name))
            d = json.load(open(ann_path, 'r'))
            self.dep_file = os.path.join(data_path, dataset_name, 'dep_tree_{}.json'.format(dataset_name))

        self.tree_gen_mode = tree_gen_mode
        self.linear = kwargs.get('linear', False)
        if self.tree_gen_mode is not None:
            self.d_dep_tree = json.load(open(self.dep_file, 'r'))
        self.extra_symbol = kwargs.get('extra_symbol', False)
        self.extra_dot = kwargs.get('extra_dot', False)

        self.images, self.sents = {}, {}
        self.pairs, self.split = [], []

        for item in d['images']:

            if self.dataset_name == 'activitynet' and self.feat_type == 'resnet_bn':
                root = _activitynet_resnet_bn_feat_dir
                i = item['filename']
                if (not os.path.exists(os.path.join(root, 'training', f'{i}_bn.npy'))) and (not os.path.exists(os.path.join(root, 'validation', f'{i}_bn.npy'))):
                    continue

            filename = item['filename']
            split = item['split']
            _imgid = item['cocoid'] if 'cocoid' in item else item['imgid']
            image_id = int(_imgid)
            self.images[image_id] = {'filename': filename}
            if 'duration' in item:
                self.images[image_id]['duration'] = item['duration']
            for sent in item['sentences']:
                sent_id = sent['sentid']
                self.sents[sent_id] = sent
                self.pairs.append((image_id, sent_id))
                if split == 'restval': split = 'train'
                self.split.append(split)

        self.image_id_to_sent = defaultdict(list)
        self.sent_id_to_image = {}
        for image_id, sent_id in self.pairs:
            self.image_id_to_sent[image_id].append(sent_id)
            self.sent_id_to_image[sent_id] = image_id

        self.cache = LRUCache(3000)

    def process_time_stamp(self, duration, timestamp, feature_length):
        """
        for activitynet
        :param duration: float
        :param timestamp: [float, float]
        :param feature_length: int
        :return:
        """
        # index = [int(max(min(feature_length * time / duration, feature_length - 1), 0)) for time in timestamp]
        index = [int(feature_length * time / duration) for time in timestamp]
        index[0] = max(min(index[0], feature_length - 1), 0)
        index[1] = max(min(index[1], feature_length - 1), 0)
        # if not (index[1] >= index[0]):
        #     print('duration: {}, timestamp: {}, feature_length: {}'.format(duration, timestamp, feature_length))
        if index[1] < index[0]:
            index[0], index[1] = index[1], index[0]
        assert index[1] >= index[0]
        if index[1] - index[0] < 2:
            if index[0] == 0:
                index[1] += 2
            elif index[1] == feature_length - 1:
                index[0] -= 2
            else:
                index[0] -= 1
                index[1] += 1
        return index

    def read_image_feat(self, filename, image_id, **kwargs):
        d = {}
        if self.feat_type == 'none':
            return d

        if self.dataset_name == 'coco':
            image_id = int(filename[:-4].split('_')[-1])

            # resnet feature
            key = filename
            if not hasattr(self, 'f_feat_fc'):
                self.f_feat_fc = h5py.File(coco_feat_path, 'r')
            if self.feat_type == 'att' and not hasattr(self, 'f_feat_att'):
                self.f_feat_att = h5py.File(coco_feat_att_path, 'r')

            if self.feat_type in ['fc', 'att']:
                feat_fc = np.array(self.f_feat_fc[key])
                d['feat_fc'] = torch.from_numpy(feat_fc)
            if self.feat_type == 'att':
                feat_att = self.cache.get(key)
                if feat_att is None:
                    feat_att = np.array(self.f_feat_att[key])  # (14, 14, 2048)
                    feat_att = feat_att.reshape(-1, feat_att.shape[-1])
                    feat_att = torch.from_numpy(feat_att)  # (14, 14, 2048)
                    self.cache.put(key, feat_att)
                d['feat_att'] = feat_att
                mask = np.ones(feat_att.shape[0], dtype=np.int8)
                d['att_mask'] = torch.from_numpy(mask)

            # bottom-up feature
            # key = str(image_id)
            # if self.feat_mode in ['fc', 'att']:
            #     feat_fc = np.load(os.path.join(coco_feat_dir, '{}.npy'.format(key)))
            #     d['feat_fc'] = torch.from_numpy(feat_fc)
            # if self.feat_mode == 'att':
            #     feat_att = self.cache.get(key)
            #     if feat_att is None:
            #         # feat_att = np.array(self.f_feat_att[key])               # (14, 14, 2048)
            #         feat_att = np.load(os.path.join(coco_feat_att_dir, '{}.npz'.format(key)))['feat']
            #         feat_att = feat_att.reshape(-1, feat_att.shape[-1])     # (?, 2048)
            #         feat_att = torch.from_numpy(feat_att)                   # (?, 2048)
            #         self.cache.put(key, feat_att)
            #     mask = np.ones(feat_att.shape[0], dtype=np.int8)
            #     d['feat_att'] = feat_att
            #     d['att_mask'] = torch.from_numpy(mask)
        elif self.dataset_name == 'flickr30k':
            key = image_id
            _feat = self.cache.get(key)
            if _feat is not None:
                feat_att, feat_fc = _feat
            else:
                feat = np.load(os.path.join(flickr30k_feat_dir, '{}.npz'.format(key)))['feat']  # (36, 2048)
                feat_att = feat
                feat_fc = feat_att.mean(axis=0)
                self.cache.put(key, (feat_att, feat_fc))

            mask = np.ones(feat_att.shape[0], dtype=np.int8)

            d['feat_att'] = torch.from_numpy(feat_att)
            d['att_mask'] = torch.from_numpy(mask)
            d['feat_fc'] = torch.from_numpy(feat_fc)

        elif self.dataset_name == 'msvd' or self.dataset_name == 'msrvtt':
            if self.feat_type == 'irv2+i3d':
                if self.dataset_name == 'msvd':
                    frame_feat_path = msvd_frame_feat_path
                    region_feat_path = msvd_region_feat_path
                    vid = int(filename[3:]) - 1
                elif self.dataset_name == 'msrvtt':
                    frame_feat_path = msrvtt_frame_feat_path
                    region_feat_path = msrvtt_region_feat_path
                    vid = int(filename[5:])
            elif self.feat_type == 'irv2+i3d_of':
                pass
            elif self.feat_type == 'resnet101+resnext101':
                pass
            elif self.feat_type == 'resnet101+i3d':
                pass
            key = vid

            _feat = self.cache.get(key)
            if _feat is not None:
                feat_att, feat_fc = _feat
            else:
                if self.feat_type == 'irv2+i3d':
                    frame_feat = np.load(os.path.join(frame_feat_path, '{}.npz'.format(key)))
                    # region_feat = np.load(os.path.join(region_feat_path, '{}.npz'.format(key)))
                    # sfeat, vfeat = region_feat['sfeat'], region_feat['vfeat']       # (26, 36, 5) (26, 36, 2048)
                    feat_cnn, feat_i3d = frame_feat['feat_cnn'], frame_feat['feat_i3d']
                    feat_att = np.concatenate((feat_cnn, feat_i3d), axis=1)
                    feat_fc = feat_att.mean(axis=0)
                    self.cache.put(key, (feat_att, feat_fc))
                elif self.feat_type == 'irv2+i3d_of':
                    pass
                elif self.feat_type == 'resnet101+resnext101':
                    pass
                elif self.feat_type == 'resnet101+i3d':
                    pass

            mask = np.ones(feat_att.shape[0], dtype=np.int8)

            d['feat_att'] = torch.from_numpy(feat_att)
            d['att_mask'] = torch.from_numpy(mask)
            d['feat_fc'] = torch.from_numpy(feat_fc)
        elif self.dataset_name == 'charades':
            key = filename
            _feat = self.cache.get(key)
            if _feat is not None:
                feat_att, feat_fc = _feat
            else:
                feat = np.load(os.path.join(charades_feat_dir, '{}.npy'.format(key)))
                feat_att = feat
                feat_fc = feat.mean(axis=0)
                self.cache.put(key, (feat_att, feat_fc))

            mask = np.ones(feat_att.shape[0], dtype=np.int8)

            d['feat_att'] = torch.from_numpy(feat_att)
            d['att_mask'] = torch.from_numpy(mask)
            d['feat_fc'] = torch.from_numpy(feat_fc)
        elif self.dataset_name == 'activitynet':
            key = filename
            _feat = self.cache.get(key)

            if _feat is None:
                if self.feat_type == 'c3d':     # 500
                    _feat = np.load(os.path.join(activitynet_c3d_feat_dir, 'v_{}.npy'.format(key)))  # (feat_length, 500)
                elif self.feat_type == 'resnet_bn':     # 3072
                    for feat_path in activitynet_resnet_bn_feat_dir:
                        if os.path.exists(os.path.join(feat_path, '{}_bn.npy'.format(key))):
                            feat_bn = np.load(os.path.join(feat_path, '{}_bn.npy'.format(key)))             # (?, 1024)
                            feat_resnet = np.load(os.path.join(feat_path, '{}_resnet.npy'.format(key)))     # (?, 2048)
                            _feat = np.concatenate((feat_bn, feat_resnet), axis=1)
                            break
                    assert _feat is not None, 'feat for {} not exist'.format(key)
                self.cache.put(key, _feat)

            duration, timestamp = kwargs['duration'], kwargs['timestamp']
            index = self.process_time_stamp(duration, timestamp, _feat.shape[0])
            if self.feat_type == 'c3d':
                feat = _feat[index[0] : index[1] + 1 : 4]
            elif self.feat_type == 'resnet_bn':
                feat = _feat[index[0] : index[1] + 1]

                # max_feat_len = 200
                # if len(feat) > max_feat_len:         # interval = 3 max 100: cut 3%; max 50: cut 13%
                #     feat = feat[::int(round(len(feat) / max_feat_len))][:max_feat_len]

                if len(feat) > 300:  # interval = 3 max 100: cut 3%; max 50: cut 13%
                    feat = feat[:300]

            assert len(feat) > 0, 'invalid index: {}'.format(index)
            # if len(feat) > 128:
            #     feat = feat[::int(len(feat) / 128)]
            feat_fc = feat.mean(axis=0)
            mask = np.ones(feat.shape[0], dtype=np.int8)

            d['feat_att'] = torch.from_numpy(feat)
            d['att_mask'] = torch.from_numpy(mask)
            d['feat_fc'] = torch.from_numpy(feat_fc)

        return d

    def __getitem__(self, index: int):
        d = {'index': index}
        image_id, sent_id = self.pairs[index]
        filename = self.images[image_id]['filename']
        sent = self.sents[sent_id]

        d.update({'image_id': image_id,  'image_filename': filename})

        if self.dataset_name == 'activitynet':
            kwargs = {'duration': self.images[image_id]['duration'], 'timestamp': sent['timestamp'], 'sampling_sec': sent['frame_to_second']}
        else:
            kwargs = {}

        d.update(self.read_image_feat(filename, image_id, **kwargs))

        tokens, raw = sent['tokens'], sent['raw']
        token_ids, token_length = convert_words_to_id(self.dictionary, tokens, self.max_sent_len)
        d.update({'sent_id': sent_id, 'raw': raw, 'words': tokens, 'token_id': token_ids, 'token_length': token_length})

        if self.tree_gen_mode is not None:
            tree = self.d_dep_tree[str(sent_id)]
            if isinstance(tree, list):          # multiple sentences
                multiple_sent = True
                tree = {'w': self.dictionary.bos_word, 'r': tree}
                if not self.linear:                 # generate trees
                    has_start_token, has_end_token = False, True
                else:                               # generate seq
                    has_start_token, has_end_token = True, True
            else:                               # normal
                multiple_sent = False
                has_start_token, has_end_token = True, True

            if self.extra_symbol:       # not used
                tree = _add_extra_symbol(tree)

            if self.extra_dot:
                if multiple_sent:
                    for i, t in enumerate(tree['r']):
                        _add_sentence_end_point(t)
                else:
                    _add_sentence_end_point(tree)

            if self.linear:
                _actions = get_actions_linear(words=tokens, vocab=self.dictionary, has_start_token=has_start_token, has_end_token=has_end_token, w_type='id')
            elif self.tree_gen_mode == 'dfs':
                _actions = get_actions_dfs_new(tree=tree, has_start_token=has_start_token, has_end_token=has_end_token, w_type='id', vocab=self.dictionary)
            elif self.tree_gen_mode == 'bfs':
                _actions = get_actions_bfs_new(tree=tree, has_start_token=has_start_token, has_end_token=has_end_token, w_type='id', vocab=self.dictionary)

            fixed_length = self.max_sent_len + 2
            action_count = min(self.max_sent_len + 2, len(_actions))    # FIXME: correct length?
            action_labels = np.zeros(shape=(fixed_length, 4))
            action_labels[:, 2] = self.dictionary.pad_index
            for i in range(action_count):
                action_labels[i][0] = _actions[i][0]                        # has_sibling
                action_labels[i][1] = _actions[i][1]                        # has_child
                action_labels[i][2] = _actions[i][2]                        # w
                action_labels[i][3] = {'l': 0, 'r': 1}[_actions[i][3]]      # child_type
            # action_labels[i][2] = self.dictionary.eos_index

            d.update({'actions': action_labels, 'action_count': action_count, 'dep_tree': tree})

            # added
            if self.linear:
                wo = None
            elif self.tree_gen_mode == 'dfs':
                wo = get_word_gen_order_dfs(tree, self.dictionary)
            elif self.tree_gen_mode == 'bfs':
                wo = None                # TODO: implement

            if wo is not None:
                wo = [(self.dictionary.bos_word, -1)] + wo      # add bos token
                word_and_order = np.zeros(shape=(fixed_length, 2))
                node_count = min(fixed_length, len(wo))
                for i in range(node_count):
                    word_and_order[i][0] = self.dictionary.index(wo[i][0])
                    word_and_order[i][1] = wo[i][1]
                d['word_and_order'] = word_and_order
            else:
                d['word_and_order'] = [0]

        return d

    def __len__(self) -> int:
        return len(self.pairs)

    def get_sub_collate_fn(self):
        d = {'index': get_collate_ndarray(dtype=np.int64)}
        d.update({'image_id': get_collate_ndarray(), 'image_filename': get_collate_ndarray()})

        if self.feat_type == 'fc':
            d.update({'feat_fc': collate_default})
        if self.feat_type in ['att', 'irv2+i3d', 'irv2+i3d_of', 'resnet101+resnext101', 'resnet101+i3d', 'c3d', 'resnet_bn', 'resnet152']:
            d.update({
                'feat_fc': collate_default,
                'feat_att': get_collate_att_feat(torch.float),
                'att_mask': get_collate_att_feat(torch.int8),
                # 'feat_att': collate_default,
                # 'att_mask': collate_default
            })
        d.update({'sent_id': get_collate_ndarray(),
                  'raw': get_collate_ndarray(),
                  'words': get_collate_ndarray(dtype=object),
                  'token_id': get_collate_seq_tensor(dtype=torch.int64, seq_len_field='token_length'),
                  'token_length': get_collate_tensor(dtype=torch.int64)})
        if self.tree_gen_mode is not None:
            d.update({'actions': get_collate_seq_tensor(dtype=torch.int64, seq_len_field='action_count'),
                      'action_count': get_collate_tensor(dtype=torch.int64),
                      'dep_tree': collate_list,})
            d['word_and_order'] = get_collate_tensor(dtype=torch.int64)
        return d

    def get_split_index(self, split, **kwargs):
        if split in ('validation', 'valid'):
            split = 'val'
        iter_mode = kwargs.get('iter_mode', 'sents')
        assert iter_mode in [
            'sent',                # iterate over all sentences
            'image'                # iterate over images uniquely
        ], 'iter_mode is \"{}\"'.format(iter_mode)
        if iter_mode == 'sent':
            return list(filter(lambda i: self.split[i] == split, range(len(self))))
        elif iter_mode == 'image':
            index = []
            image_set = set()
            for i, (image_id, sent_id) in enumerate(self.pairs):
                if self.split[i] == split:
                    if image_id not in image_set:
                        image_set.add(image_id)
                        index.append(i)
            return index

    def get_image_id_by_sent_id(self, sent_id):
        return self.sent_id_to_image[sent_id]

    def get_sent_id_by_image_id(self, image_id):
        return self.image_id_to_sent[image_id]

    def get_ground_truth_sents(self, image_id):
        sent_id = self.image_id_to_sent[image_id]
        return [self.sents[i] for i in sent_id]

