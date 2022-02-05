import os
import sys
import json
import traceback

java_path = '/usr/local/lib/jdk1.8.0_241/bin/java'
os.environ['PATH'] += ':' + os.path.split(java_path)[0]
sys.path.append('coco-caption')
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    traceback.print_exc()
    print('import from coco-caption failed')

try:
    from .customjson import dump_custom
    dump_func = dump_custom
except:
    dump_func = json.dump


class COCOResultGenerator:
    def __init__(self):
        self.result_obj = []
        self.annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
        self.caption_id = 0
        self.annotation_image_set = set()
        self.test_image_set = set()

    def add_annotation(self, image_id, caption_raw):
        if image_id not in self.annotation_image_set:
            self.annotation_obj['images'].append({'id': image_id})
            self.annotation_image_set.add(image_id)
        self.annotation_obj['annotations'].append({'image_id': image_id, 'caption': caption_raw, 'id': self.caption_id})
        self.caption_id += 1

    def add_output(self, image_id, caption_output, image_filename=None, metadata=None):
        assert(image_id in self.annotation_image_set and image_id not in self.test_image_set)
        item = {"image_id": image_id, "caption": caption_output}
        if metadata is not None:
            item['meta'] = metadata
        if image_filename is not None:
            item["image_filename"] = image_filename
        self.result_obj.append(item)
        self.test_image_set.add(image_id)

    def has_output(self, image_id):
        return image_id in self.test_image_set

    def get_annotation_and_output(self):
        return self.annotation_obj, self.result_obj

    def dump_annotation_and_output(self, annotation_file, result_file):
        self.dump_annotation(annotation_file)
        self.dump_output(result_file)

    def dump_annotation(self, annotation_file):
        with open(annotation_file, 'w') as f:
            print('dumping {} annotations to {}'.format(len(self.annotation_obj['annotations']), annotation_file))
            dump_func(self.annotation_obj, f, indent=4)

    def dump_output(self, result_file):
        with open(result_file, 'w') as f:
            print('dumping {} results to {}'.format(len(self.result_obj), result_file))
            dump_func(self.result_obj, f, indent=4)

    def add_img_scores(self, img_scores, save_spice=False):
        """
        :param img_scores: [{'image_id': i, 'Bleu_1': 1, ...}, {'image_id': 0, 'Bleu_1': xx, }]
                returned by calling eval(ann_file, res_file, True)
        :return:
        """
        img_scores = dict([(i['image_id'], i) for i in img_scores])

        for item in self.result_obj:
            scores = img_scores[item['image_id']]
            for _key, _score in scores.items():
                if _key == 'SPICE': continue
                item[_key] = round(_score, 6)


def evaluate(ann_file, res_file, return_imgscores=False, use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr')):
    coco = COCO(ann_file)
    cocoRes = coco.loadRes(res_file)
    # create cocoEval object by taking coco and cocoRes
    # cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval = COCOEvalCap(coco, cocoRes, use_scorers=use_scorers)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    all_score = {}
    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        # print('%s: %.4f' % (metric, score))
        all_score[metric] = score

    img_scores = [cocoEval.imgToEval[key] for key in cocoEval.imgToEval.keys()]

    if return_imgscores:
        return all_score, img_scores
    else:
        return all_score


def save_metrics(metric_file, metrics, epoch=None, global_step=None):
    lines = []
    if not os.path.exists(metric_file):
        first_line = ['epoch', 'step']
        for metric in metrics:
            first_line.append(metric)
        lines.append(','.join('{:<10}'.format(i) for i in first_line))
    else:
        with open(metric_file, 'r') as f:
            first_line = [i.strip() for i in f.readline().split(',')]
        if set(first_line[2:]) != set(metrics.keys()):
            print('existing metrics:', first_line[2:])
            print('received metrics:', list(metrics.keys()))

    strs = []
    for i in first_line:
        if i == 'epoch':
            strs.append('{:<10}'.format(epoch) if epoch else ' ' * 10)
        elif i == 'step':
            strs.append('{:<10}'.format(global_step) if global_step else ' ' * 10)
        else:
            strs.append('{:<10.6f}'.format(metrics[i]))
    lines.append(','.join(strs))
    with open(metric_file, 'a') as f:
        f.writelines([i + '\n' for i in lines])


if __name__ == '__main__':
    dataset = sys.argv[1]
    annotation_file = sys.argv[2]
    result_file = sys.argv[3]

    dense = False
    if dataset == 'activitynet':
        dense = True

    # scores, _ = evaluate(annotation_file, result_file, return_imgscores=True)

    if not dense:
        d = {}
        with open(annotation_file, 'r') as f:
            d1 = json.load(f)
        with open(result_file, 'r') as f:
            d2 = json.load(f)

        image_id_map = {}
        for i in d1['annotations']:
            image_id = i['image_id']
            if image_id not in image_id_map:
                image_id_map[image_id] = str(len(image_id_map))
            # if len(image_id_map) > 50:
            #     break

        for i in d1['annotations']:
            if i['image_id'] not in image_id_map: continue
            image_id = image_id_map[i['image_id']]
            sent = i['caption']
            if image_id not in d:
                d[image_id] = {'refs': [], 'cand': []}
            d[image_id]['refs'].append(sent)
        for i in d2:
            if i['image_id'] not in image_id_map: continue
            image_id = image_id_map[i['image_id']]
            sent = i['caption']
            d[image_id]['cand'].append(sent)
    else:
        d = {}
        video_index = 0
        with open(annotation_file, 'r') as f:
            d1 = json.load(f)
        for video_name in d1['results'].keys():
            for item in d1['results'][video_name]:
                image_id = str(video_index)
                d[image_id] = {'refs': [item['sentence']], 'cand': [item['raw']]}
                video_index += 1


    eval_file = './tmp_charades.json'
    print('total:', len(d))
    with open(eval_file, 'w') as f:
        json.dump(d, f, indent=4)

    cmd = f'{sys.executable} '\
            f'/media/wentian/sdb1/work/improved-bertscore-for-image-captioning-evaluation-master/evaluate.py '\
            f" {dataset} "\
            f' \"{eval_file}\" '\
            f' -stop-word-path /media/wentian/sdb1/work/improved-bertscore-for-image-captioning-evaluation-master/stop_word_list.txt' \
            f' -invalidate-cache'
    print(cmd)
    os.system(
        cmd
    )


