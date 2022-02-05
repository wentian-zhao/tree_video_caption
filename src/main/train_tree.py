import os
import sys
from collections import Counter
from functools import reduce

sys.path.append(os.getcwd())
sys.path.append('coco-caption')

from util.focal import masked_binary_focal_loss_with_logits

from contextlib import redirect_stdout, redirect_stderr

import torch
import numpy as np
from fairseq.optim.adam import FairseqAdam
# from fairseq.optim.lr_scheduler.fixed_schedule import FixedSchedule
from fairseq.sequence_generator import SequenceGenerator

from config import *
from util import *
from util.loss import masked_cross_entropy, masked_bce_with_logits, sequence_mask
from util.evaluate import save_metrics, evaluate, COCOResultGenerator
from util.vocab import load_dict
from util.reward import init_scorer, get_sent_scores
from util.dep_tree import action_to_tree_bfs, action_to_tree_dfs, tree_to_seq, tree_to_seq_ppl
from main.task import CustomTask
from main.data import *
from main.model.fairseq_tree_c import TopDownTreeModel
from main.model import SimpleSequenceGenerator1, SimpleSequenceGeneratorTree, GreedySequenceGenerator
from main.model.fairseq_tree_t_3 import TransformerTreeModel2

from torch.optim.lr_scheduler import StepLR
from torch.optim.adam import Adam
import torch.nn.functional as F
from main.model.merge_label import *
from util.vis_tree import to_pptree, print_tree

from util.densevid_eval3 import eval_score
from main.data import activitynet_references

import kenlm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# torch.autograd.set_detect_anomaly(True)
# if torch.cuda.is_available():
#     torch.cuda.synchronize()


def _to_tensor(data, dtype, device):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype, device=device)
    else:  # tensor
        return data.to(dtype).to(device)


class CaptionTask(CustomTask):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super(CaptionTask, self).add_arguments(parser)

        parser.add_argument('--feat-type', type=str, default='')

        parser.add_argument('--max-sent-len', type=int, default=20)

        parser.add_argument('--grad-clip-method', type=str, default='value', choices=['value', 'norm', 'none'])
        parser.add_argument('--grad-value', type=float, default=2.0, help='used for clip_grad_value_')
        parser.add_argument('--grad-norm', type=float, default=1.0, help='used for clip_grad_norm_')

        parser.add_argument('--beam-size', type=int, default=5)

        parser.add_argument('--pretrained-model', default='', type=str,
                            help='path to pretrained model (does not load scheduler)')

        parser.add_argument('--save-spice', default=0, type=int, help='1 = save spice score')

        parser.add_argument('--debug', type=int, default=0)

        parser.add_argument('--sc-after', type=int, default=-1)
        parser.add_argument('--iter-mode', type=str, default='sent', help='sent|image')

        parser.add_argument('--tree-gen-mode', type=str, default='dfs', help='dfs|bfs')
        parser.add_argument('--linear', type=int, default=0, help='linear == 1 -> no tree structure')

        parser.add_argument('--extra-symbol', type=int, default=0, help='add extra symbols to tree (*change max-sent-len when using this)')
        parser.add_argument('--extra-dot', type=int, default=0, help='add the . symbol at the end of the sentence')

        parser.add_argument('--lw-has-sibling', type=float, default=1.)
        parser.add_argument('--lw-has-child', type=float, default=1.)
        parser.add_argument('--lw-child-type', type=float, default=1.)
        # parser.add_argument('--reweight-has-sibling', type=int, default=0)
        parser.add_argument('--weight-has-sibling-pos', type=float, default=1.0)
        parser.add_argument('--weight-has-sibling-neg', type=float, default=1.0)

        parser.add_argument('--decode-use-lm',  type=int, default=0)

        parsed, unknown = parser.parse_known_args()
        model = parsed.model.lower()
        optimizer = parsed.optimizer.lower()
        scheduler = parsed.scheduler.lower()

        if model == 'topdowntree':
            TopDownTreeModel.add_args(parser)
            self.model_cls = TopDownTreeModel
        elif model == 'transformertree':
            TransformerTreeModel2.add_args(parser)
            self.model_cls = TransformerTreeModel2

        optimizer_add_args(optimizer, parser)

        scheduler_add_args(scheduler, parser)
        # if scheduler == 'steplr':
        #     self.scheduler_cls = StepLR
        # else:
        #     print('scheduler:', scheduler)

    def init_data(self):
        dataset_name = self.args.dataset
        self.feat_type = self.args.feat_type

        if dataset_name in ('coco', 'flickr30k'):
            self.args.feat_dim = 2048
        elif dataset_name in ('msvd', 'msrvtt'):
            self.args.feat_dim = 2560

        if self.args.debug:
            self.args.num_workers = 0
            print('debug mode, num_workers set to 0')

        dict_file = os.path.join(data_path, 'dictionary_{}.txt'.format(dataset_name))
        if self.args.extra_symbol:
            self.extra_special_symbols = ['<lc>', '<lf>']     # last_child, leaf
        else:
            self.extra_special_symbols = None
        dictionary = load_dict(dict_file, min_word_count=5, extra_special_symbols=self.extra_special_symbols)

        if self.args.extra_dot:
            dictionary.add_symbol('.')

        print('dictionary size:', len(dictionary))
        dataset = CaptionDataset(dataset_name, dictionary, max_sent_len=self.args.max_sent_len,
                                 feat_type=self.feat_type, tree_gen_mode=self.args.tree_gen_mode,
                                 extra_symbol=self.args.extra_symbol, linear=self.args.linear,
                                 extra_dot=self.args.extra_dot)
        train_dataloader = dataset.get_split_dataloader(
            split='train',
            dataloader_kwargs={
                'batch_size': self.args.batch_size, 'num_workers': self.args.num_workers, 'shuffle': True
            },
            get_split_index_kwargs={'iter_mode': self.args.iter_mode}
        )
        test_dataloader = dataset.get_split_dataloader(
            split='test',
            get_split_index_kwargs={'iter_mode': 'image'},
            dataloader_kwargs={
                'batch_size': self.args.batch_size_test, 'num_workers': self.args.num_workers
            }
        )
        self.dictionary = dictionary
        self.dataset = dataset
        self.dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    def init_model(self, state_dict=None):
        if self.args.debug:
            print('torch.autograd.set_detect_anomaly(True)')
            torch.autograd.set_detect_anomaly(True)
            if torch.cuda.is_available():
                print('torch.cuda.synchronize()')
                torch.cuda.synchronize()

        self.model = self.model_cls.build_model(args=self.args, task=None, dictionary=self.dictionary).to(device)
        self.optimizer = get_optimizer(self.args, self.model)
        self.scheduler = get_scheduler(self.args, self.optimizer)

        if state_dict is not None:
            assert len(self.args.pretrained_model) == 0
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

        if len(self.args.pretrained_model) > 0:
            state_dict = self.load_model(self.args.pretrained_model)
            self.model.load_state_dict(state_dict['model'])

    def prefetch_to_device(self, batch_data):
        batch_data['feat_fc'] = _to_tensor(batch_data['feat_fc'], torch.float, device)
        if 'feat_att' in batch_data:
            batch_data['feat_att'] = _to_tensor(batch_data['feat_att'], torch.float, device)
            batch_data['att_mask'] = _to_tensor(batch_data['att_mask'], torch.float, device)
        return batch_data

    def _get_model_input(self, batch_data):
        feat_input = {'feat_fc': _to_tensor(batch_data['feat_fc'], torch.float, device)}
        feat_input['src_tokens'] = feat_input['feat_fc']
        feat_input['src_lengths'] = None
        if 'feat_att' in batch_data:
            feat_input['feat_att'] = _to_tensor(batch_data['feat_att'], torch.float, device)
            feat_input['att_mask'] = _to_tensor(batch_data['att_mask'], torch.float, device)
        return feat_input

    def train_epoch(self):
        self.label_count = Counter()
        self.feat_count = Counter()
        super().train_epoch()
        print('labels:', json.dumps(self.label_count, indent=4))
        # print('feat count:', self.feat_count)

    def train_step(self, model, batch_data):
        timer = Timer()
        sc_flag = self.args.sc_after > 0 and self.epoch >= self.args.sc_after

        feat_input = self._get_model_input(batch_data)
        tokens, token_length = (_to_tensor(batch_data[key], torch.int64, device) for key in
                                ['token_id', 'token_length'])
        actions, action_count = (_to_tensor(batch_data[key], torch.int64, device) for key in
                                 ('actions', 'action_count'))   # actions: (batch_size, max_len, 4)
        # token_input, token_expected = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        # actions_merged, _ = merge_cartesian(tensors=(actions[:, :, i] for i in range(actions.shape[2])), ranges=model.decoder.label_ranges)

        word_and_order = _to_tensor(batch_data['word_and_order'], torch.int64, device)
        # TODO: remove
        feat_length = feat_input['att_mask'].detach().cpu().numpy().sum(axis=1).astype(np.int64).tolist()
        self.feat_count.update(feat_length)

        if not sc_flag:
            _output = model.forward(
                # src_tokens=None, src_lengths=None,
                prev_output_tokens=actions[:, :-1, 2], **feat_input,
                topo_labels=torch.cat((actions[:, :, :2], actions[:, :, 3:]), dim=-1).contiguous(),
                input_actions=actions,
                timer=timer,
                word_and_order=word_and_order
            )

            logits = _output[1]['logits']
            loss_dict = {}
            logit_index = {'has_sibling': 0, 'has_child': 1, 'child_type': 3}

            logit_w = _output[0]
            loss_w = masked_cross_entropy(logit_w, actions[:, 1:, 2], action_count - 1)
            loss_dict['loss_w'] = loss_w

            acc_dict = {}
            mask = sequence_mask(seq_length=(action_count - 1), max_length=logit_w.shape[1])

            for i, (key, index) in enumerate(logit_index.items()):
                # expected = actions[:, :-1, index]       # the output topology label does not contain position [0]
                expected = actions[:, 1:, index]
                logit = logits[key]
                # loss = masked_cross_entropy(logit, expected, action_count - 1)

                # if key == 'has_sibling':
                #     loss = masked_bce_with_logits(logit, expected, action_count - 1,
                #                                   weight_pos=self.args.weight_has_sibling_pos,
                #                                   weight_neg=self.args.weight_has_sibling_neg)
                # else:
                #     loss = masked_bce_with_logits(logit, expected, action_count - 1)
                if key == 'has_sibling':
                    loss = masked_binary_focal_loss_with_logits(logit, expected, action_count - 1,
                                                                alpha=0.6, gamma=1.0)
                else:
                    loss = masked_binary_focal_loss_with_logits(logit, expected, action_count - 1,
                                                                alpha=0.5, gamma=1.0)
                # elif key == 'has_child':
                #     loss = masked_binary_focal_loss_with_logits(logit, expected, action_count - 1,
                #                                                 alpha=0.45, gamma=1.0)
                # elif key == 'chile_type':
                #     loss = masked_binary_focal_loss_with_logits(logit, expected, action_count - 1,
                #                                                 alpha=0.54, gamma=1.0)

                expected_labels = expected.detach().cpu().numpy().reshape(-1).astype(np.int)
                predicted_labels = (torch.sigmoid(logit) >= 0.5).detach().cpu().numpy().reshape(-1).astype(np.int)
                words = np.array([self.dictionary.string([i]) for i in actions[:, 1:, 2].cpu().numpy().reshape(-1)])

                objs = ['{}-{}-\"{}\"'.format(*i) for i in zip(expected_labels, predicted_labels, words)]
                objs = np.array(objs).reshape(*expected.shape)

                loss_dict['loss_' + key] = loss

                if not self.args.linear:
                    tp = ((torch.sigmoid(logit) >= 0.5) * (expected == 1) * mask).sum()
                    tn = ((torch.sigmoid(logit) < 0.5) * (expected == 0) * mask).sum()
                    fp = ((torch.sigmoid(logit) >= 0.5) * (expected == 0) * mask).sum()
                    fn = ((torch.sigmoid(logit) < 0.5) * (expected == 1) * mask).sum()
                    correct = (((torch.sigmoid(logit) > 0.5) == expected) * mask).sum()
                    total = mask.sum()
                    accuracy = float((correct / total).detach().cpu())
                    precision = float(tp / (tp + fp + 1e-6))
                    recall = float(tp / (tp + fn + 1e-6))
                    recall_neg = float(tn / (tn + fp + 1e-6))
                    acc_dict[key + '_accuracy'] = round(accuracy, 4)
                    acc_dict[key + '_precision'] = round(precision, 4)
                    acc_dict[key + '_recall'] = round(recall, 4)
                    acc_dict[key + '_recall_neg'] = round(recall_neg, 4)
                    self.label_count[key + '_pos'] += int(tp + fn)
                    self.label_count[key + '_neg'] += int(tn + fp)

            loss_dict['loss_has_sibling'] = loss_dict['loss_has_sibling'] * self.args.lw_has_sibling
            loss_dict['loss_has_child'] = loss_dict['loss_has_child'] * self.args.lw_has_child
            loss_dict['loss_child_type'] = loss_dict['loss_child_type'] * self.args.lw_child_type

            # if self.epoch > 1:
            #     if self.epoch % 2 != 0:     # 4, 6, 8, 10 ...
            #         del loss_dict['loss_w']
            #     else:                       # 3, 5, 7, 9, ...
            #         del loss_dict['loss_has_sibling']
            #         del loss_dict['loss_has_child']3
            #         del loss_dict['loss_child_type']

            if not self.args.linear:
                self.writer.add_scalars('acc', acc_dict, global_step=self.global_step)
                if self.global_step % 150 == 0:
                    print(acc_dict)

        else:
            if not hasattr(self, 'seq_gen'):
                seq_gen = GreedySequenceGenerator(bos_index=self.dictionary.bos_index,
                                                   eos_index=self.dictionary.eos_index,
                                                   unk_index=self.dictionary.unk_index,
                                                   pad_index=self.dictionary.pad_index)
                df = os.path.join(data_path, self.args.dataset, 'df_{}_words.pkl'.format(self.args.dataset))
                init_scorer(df=df)
                self.seq_gen = seq_gen
                print('hasattr', hasattr(self, 'seq_gen'))
            else:
                seq_gen = getattr(self, 'seq_gen')

            max_length = self.args.max_sent_len

            batch_size = len(batch_data['image_id'])

            # prepare ground truth
            gts_raw = OrderedDict()
            for i in range(batch_size):
                # TODO: add eos word?
                # with eos word
                g = [' '.join([*s['tokens'], self.dictionary.eos_word]) for s in self.dataset.get_ground_truth_sents(batch_data['image_id'][i])]
                # #  no eos word
                # g = [' '.join(s['tokens']) for s in self.dataset.get_ground_truth_sents(batch_data['image_id'][i])]
                gts_raw[i] = g

            timer.tick('generate')

            sample = {'id': _to_tensor(batch_data['index'], torch.int64, device), 'net_input': feat_input}

            finalized_sample = seq_gen.generate(model, sample, max_length=max_length, sample_method='multinomial')
            model.eval()
            with torch.no_grad():
                finalized_greedy = seq_gen.generate(model, sample, max_length=max_length, sample_method='greedy')
            model.train()

            timer.tock('generate')
            timer.tick('prep output')

            symbols_to_ignore = self.dictionary.symbols[:self.dictionary.nspecial]

            # prepare output logprob
            mask = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.float, device=device)
            output_logprob = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.float, device=device)
            token_lengths = [0] * batch_size
            for i, output in finalized_sample.items():
                tokens, logprob = output[0]['tokens'], output[0]['positional_scores']       # tokens include </s>
                length = len(tokens)
                assert tokens.shape == logprob.shape
                token_lengths[i] = length
                output_logprob[i, :length] = logprob
                mask[i, :length] = 1

            # prepare sentences
            _gts_raw = OrderedDict()
            _res = OrderedDict()
            # _res_lengths = OrderedDict()
            for i, (batch_index, output) in enumerate([*finalized_sample.items(), *finalized_greedy.items()]):
                output = output[0]
                _actions = output['additional'].detach().cpu().numpy()
                actions = []
                for a in _actions:
                    actions.append((a[0], a[1], self.dictionary[a[2]], a[3]))
                if self.args.tree_gen_mode == 'dfs':
                    tree = action_to_tree_dfs(actions[1:])
                else:
                    tree = action_to_tree_bfs(actions[1:])
                _seq = tree_to_seq(tree)
                # TODO: whether to filter the </s> symbol in _seq?
                seq = list(filter(lambda x: isinstance(x, str) and (x not in symbols_to_ignore), _seq))
                # seq = list(filter(lambda x: isinstance(x, str), _seq))

                _res[i] = ' '.join(seq[:-1])
                # _res_lengths[i] = len(seq) - 1
                _gts_raw[i] = gts_raw[batch_index]
            # all_output_tokens = all_output_tokens.detach().cpu().numpy()
            # for i, tokens in enumerate(all_output_tokens):
            #     _res[i] = ' '.join([self.dictionary.symbols[w] for w in tokens[:_res_lengths[i]]])

            timer.tock('prep output')
            timer.tick('reward')

            # reward
            scores = get_sent_scores(_gts_raw, _res, {'cider': 1.0, 'bleu': 0.0}, n_threads=4)
            scores = scores.reshape(2, batch_size)
            reward = scores[0] - scores[1]  # shape: (batch_size,)
            _r = reward.mean()

            if abs(float(_r)) < 1e-5:
                import IPython; IPython.embed()

            timer.tock('reward')
            timer.tick('loss')

            self.writer.add_scalar(tag='reward', scalar_value=float(_r), global_step=self.global_step)

            # loss
            reward = torch.Tensor(reward).to(device)
            reward = reward.unsqueeze(1).expand_as(output_logprob)
            loss_sc = - output_logprob * reward * mask
            loss_sc = torch.sum(loss_sc) / torch.sum(mask)
            loss_dict = {'loss_sc': loss_sc}

            timer.tock('loss')

        # self-critical (beam search)
        # else:
        #     if not hasattr(self, 'seq_gen'):
        #         seq_gen = SimpleSequenceGeneratorTree(bos_index=self.dictionary.bos_index,
        #                                            eos_index=self.dictionary.eos_index,
        #                                            unk_index=self.dictionary.unk_index,
        #                                            pad_index=self.dictionary.pad_index)
        #         df = os.path.join(data_path, self.args.dataset, 'df_{}_words.pkl'.format(self.args.dataset))
        #         init_scorer(df=df)
        #         self.seq_gen = seq_gen
        #         print('hasattr', hasattr(self, 'seq_gen'))
        #     else:
        #         seq_gen = getattr(self, 'seq_gen')
        #
        #     beam_size = 5
        #     max_length = self.args.max_sent_len
        #
        #     batch_size = len(batch_data['image_id'])
        #
        #     # prepare ground truth
        #     gts_raw = OrderedDict()
        #     for i in range(batch_size):
        #         # with eos word
        #         # g = [' '.join([*s['tokens'], self.dictionary.eos_word]) for s in self.dataset.get_ground_truth_sents(batch_data['image_id'][i])]
        #         # no eos word
        #         g = [' '.join(s['tokens']) for s in self.dataset.get_ground_truth_sents(batch_data['image_id'][i])]
        #         gts_raw[i] = g
        #
        #     timer.tick('generate')
        #
        #     sample = {'id': _to_tensor(batch_data['index'], torch.int64, device), 'net_input': feat_input}
        #     model.eval()
        #     finalized = seq_gen.generate(model, sample, beam_size=beam_size, max_length=max_length)
        #     model.train()
        #
        #     timer.tock('generate')
        #     timer.tick('prep output')
        #
        #     symbols_to_ignore = self.dictionary.symbols[:self.dictionary.nspecial]
        #
        #     # prepare output sentences
        #     output_tokens = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.int64, device=device)
        #     output_tokens.fill_(self.dictionary.pad_index)
        #     output_logprob = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.float64, device=device)
        #     token_lengths = [0] * batch_size
        #     for i, output in finalized.items():
        #         tokens, logprob = output[0]['tokens'], output[0]['positional_scores']
        #         length = len(tokens)
        #         assert tokens.shape == logprob.shape
        #         token_lengths[i] = length
        #         output_tokens[i, :length] = tokens
        #         output_logprob[i, :length] = logprob
        #     # beam search output
        #     _gts_raw = OrderedDict()
        #     _res = OrderedDict()
        #     _res_lengths = OrderedDict()
        #     all_output_tokens = torch.zeros(size=(batch_size * beam_size, max_length + 1), dtype=torch.int64, device=device)
        #     for batch_index, output in finalized.items():
        #         for beam_index in range(beam_size):
        #             tokens = output[beam_index]['tokens']
        #
        #             _actions = output[beam_index]['additional'].detach().cpu().numpy()
        #             actions = []
        #             for a in _actions:
        #                 actions.append((a[0], a[1], self.dictionary[a[2]], a[3]))
        #             if self.args.tree_gen_mode == 'dfs':
        #                 tree = action_to_tree_dfs(actions[1:])
        #             else:
        #                 tree = action_to_tree_bfs(actions[1:])
        #             _seq = tree_to_seq(tree)
        #             seq = list(filter(lambda x: x not in symbols_to_ignore, _seq))
        #
        #             length = len(tokens)
        #             i = batch_index * beam_size + beam_index
        #             # all_output_tokens[i, :length] = seq[:length]
        #             _res[i] = ' '.join(seq[:-1])
        #             _res_lengths[i] = length
        #             _gts_raw[i] = gts_raw[batch_index]
        #     # all_output_tokens = all_output_tokens.detach().cpu().numpy()
        #     # for i, tokens in enumerate(all_output_tokens):
        #     #     _res[i] = ' '.join([self.dictionary.symbols[w] for w in tokens[:_res_lengths[i]]])
        #
        #     timer.tock('prep output')
        #     timer.tick('reward')
        #
        #     # reward
        #     scores = get_sent_scores(_gts_raw, _res, {'cider': 1.0, 'bleu': 0.0}, n_threads=4)
        #     scores = scores.reshape(batch_size, beam_size)
        #     reward = scores[:, 0] - scores.mean(axis=1)  # shape: (batch_size,)
        #     _r = reward.mean()
        #
        #     timer.tock('reward')
        #     timer.tick('loss')
        #
        #     self.writer.add_scalar(tag='reward', scalar_value=float(_r), global_step=self.global_step)
        #
        #     # loss
        #     reward = torch.Tensor(reward).to(device)
        #     reward = reward.unsqueeze(1).expand_as(output_logprob)
        #     mask = (output_tokens != self.dictionary.pad_index)
        #     loss_sc = - output_logprob * reward * mask
        #     loss_sc = torch.sum(loss_sc) / torch.sum(mask)
        #     loss_dict = {'loss_sc': loss_sc}
        #
        #     timer.tock('loss')

        self.writer.add_scalars(main_tag='step_time', tag_scalar_dict=timer.get_time(), global_step=self.global_step)

        return loss_dict

    def begin_test(self):
        if not hasattr(self, 'seq_gen_test'):
            seq_gen_test = SimpleSequenceGeneratorTree(bos_index=self.dictionary.bos_index, eos_index=self.dictionary.eos_index,
                                                       unk_index=self.dictionary.unk_index, pad_index=self.dictionary.pad_index)
            setattr(self, 'seq_gen_test', seq_gen_test)
            self.lm = kenlm.Model(os.path.join(data_path, "coco.arpa"))
        vis_dir = os.path.join(self.save_folder, 'vis_tree')
        if not os.path.exists(vis_dir): os.makedirs(vis_dir)
        self.f_vis = open(os.path.join(vis_dir, 'vis_{}.txt'.format(self.epoch)), 'w')
        self.f_action = open(os.path.join(vis_dir, 'action_{}.txt'.format(self.epoch)), 'w')

    def test_step(self, model, batch_data):
        models = [model]
        seq_gen_test = getattr(self, 'seq_gen_test')

        image_id = batch_data['image_id']
        feat_input = self._get_model_input(batch_data)
        sample = {'net_input': feat_input}

        with torch.no_grad():
            finalized = seq_gen_test.generate(model, sample, beam_size=self.args.beam_size, max_length=self.args.max_sent_len)

        symbols_to_ignore = self.dictionary.symbols[:self.dictionary.nspecial]

        batch_output, batch_label = [], []
        for batch_index, item in finalized.items():
            _image_id = image_id[batch_index]

            # output_actions_cartesian = item[0]['tokens']
            # actions = split_cartesian(output_actions_cartesian, self.model.decoder.label_ranges)
            # actions = torch.stack(actions, dim=1).detach().cpu().numpy()
            # tree = action_to_tree(actions)

            output_tokens = item[0]['tokens']
            assert (output_tokens[:-1] == item[0]['additional'][1:, 2]).all()
            _actions = item[0]['additional'].detach().cpu().numpy()

            actions = []
            for a in _actions:
                actions.append((a[0], a[1], self.dictionary[a[2]], a[3]))

            if self.args.tree_gen_mode == 'dfs':
                tree = action_to_tree_dfs(actions[1:])
            else:
                tree = action_to_tree_bfs(actions[1:])
            tree['w'] = '<s>'
            if self.args.decode_use_lm:
                seq = tree_to_seq_ppl(tree, self.lm)
            else:
                seq = tree_to_seq(tree)
            seq = list(filter(lambda x: x not in symbols_to_ignore, seq))

            # output_sent = self.dictionary.string(tree_to_seq(tree)).replace('<pad>', '').strip()
            output_sent = ' '.join(seq)
            batch_output.append({'image_id': _image_id, 'output': output_sent})

            pp_tree = to_pptree(tree)
            lines = ['image id: {}, sent: {}'.format(_image_id, output_sent)]
            print_tree(pp_tree, buf=lines, horizontal=False)
            lines.append('')
            for l in lines: self.f_vis.write(l + '\n')

            lines = ['image id: {}, sent: {}'.format(_image_id, output_sent)]
            for action in actions:
                lines.append('{}, {}, {:>10}, {}'.format(action[0], action[1], action[2], {0: 'l', 1: 'r'}[action[3]]))
            lines.append('')
            for l in lines: self.f_action.write(l + '\n')

            gt_sents = self.dataset.get_ground_truth_sents(_image_id)
            gt_sents = [' '.join(sent['tokens']) for sent in gt_sents]
            batch_label.append({'image_id': _image_id, 'gt_sents': gt_sents})

        self.f_vis.flush()
        self.f_action.flush()
        return batch_label, batch_output

    def calculate_metrics(self, label_list, output_list):
        try:
            self.f_vis.close()
            self.f_action.close()
        except:
            pass

        result_generator = COCOResultGenerator()
        for i, (label, output) in enumerate(zip(label_list, output_list)):
            assert label['image_id'] == output['image_id']
            image_id = label['image_id']
            if result_generator.has_output(image_id):
                continue
            for sent in label['gt_sents']:
                result_generator.add_annotation(image_id, sent)
            result_generator.add_output(image_id, output['output'])
        self._save_results(result_generator)

    def _save_results(self, result_generator, silent=True):
        if self.args.dataset == 'activitynet':
            dvc_obj = {'version': 'version 1.0', 'results': defaultdict(list), 'external_data': {'used': False}}
            for result in result_generator.result_obj:
                image_id = result['image_id']
                caption = result['caption']
                image_item = self.dataset.images[image_id]
                sent_id = self.dataset.get_sent_id_by_image_id(image_id)[0]
                sent_item = self.dataset.sents[sent_id]
                timestamp = sent_item['timestamp']
                filename = image_item['filename']
                if not filename.startswith('v_'): filename = 'v_' + filename
                dvc_obj['results'][filename].append({'sentence': caption, 'timestamp': timestamp, 'raw': sent_item['raw']})
                # dvc_obj['results'][filename].append({'sentence': sent_item['raw'].lower(), 'timestamp': timestamp, 'raw': sent_item['raw'].lower()})
            dvc_filename = os.path.join(self.save_folder, 'result_dvc_{}.json'.format(self.epoch))
            with open(dvc_filename, 'w') as f:
                json.dump(dvc_obj, f, indent=4)

            with open('.eval_stdout', 'w') as f1, open('.eval_stderr', 'w') as f2:
                with redirect_stdout(f1), redirect_stderr(f2):
                    dvc_score = eval_score(dvc_filename, onlyMeteor=1, onlyRecallPrec=0, topN=1000, reference=activitynet_references)
            scores = {}
            for key in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']:
                scores[key] = np.mean(dvc_score[key])
            print('dvc score:', dvc_score)
            print('score:', scores)

            with open(os.path.join(self.save_folder, 'dvc_score_{}.json'.format(self.epoch)), 'w') as f:
                json.dump(dvc_score, f)

            metric_file = os.path.join(self.save_folder, 'metrics.csv')
            self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=scores, global_step=self.global_step)
            save_metrics(metric_file, scores, epoch=self.epoch, global_step=self.global_step)
        else:
            ann_file, result_file, metric_file = \
                [os.path.join(self.save_folder, i) for i in
                 ('annotation.json', 'result_{}.json'.format(self.epoch), 'metrics.csv')]
            result_generator.dump_annotation_and_output(ann_file, result_file)

            if silent:
                f = open('.eval_stdout', 'w')
            else:
                f = sys.stdout

            # use_scorers = ['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
            use_scorers = ['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr']

            with redirect_stdout(f):
                metrics, img_scores = evaluate(ann_file, result_file, return_imgscores=True,
                                               use_scorers=use_scorers)
            if silent:
                f.close()
                print(metrics)

            self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=metrics, global_step=self.global_step)
            save_metrics(metric_file, metrics, epoch=self.epoch, global_step=self.global_step)
            result_generator.add_img_scores(img_scores, save_spice=self.args.save_spice)
            for item in result_generator.result_obj:
                item['ppl'] = round(self.lm.perplexity(item['caption']), 6)
            result_generator.dump_output(result_file)

    def load_model(self, save_path):
        # super(CaptionTask, self).load_model(save_path)
        if torch.cuda.is_available():
            state_dict = torch.load(save_path)
        else:
            state_dict = torch.load(save_path, map_location='cpu')
        print('loaded model at {}'.format(save_path))
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        return state_dict

    def get_state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}


if __name__ == '__main__':
    p = CaptionTask()
    p.run()

