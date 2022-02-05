"""
train sequence models
"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append('coco-caption')

from contextlib import redirect_stdout

import torch
import numpy as np
from fairseq.optim.adam import FairseqAdam
# from fairseq.optim.lr_scheduler.fixed_schedule import FixedSchedule
from fairseq.sequence_generator import SequenceGenerator

from config import *
from util import *
from util.loss import masked_cross_entropy
from util.evaluate import save_metrics, evaluate, COCOResultGenerator
from util.vocab import load_dict
from util.reward import init_scorer, get_sent_scores
from main.task import CustomTask
from main.data import *
from main.model import *

from torch.optim.lr_scheduler import StepLR
from torch.optim.adam import Adam

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

        parser.add_argument('--max-sent-len', type=int, default=20)
        parser.add_argument('--grad-clip', type=float, default=2.0)

        parser.add_argument('--beam-size', type=int, default=5)

        parser.add_argument('--pretrained-model', default='', type=str,
                            help='path to pretrained model (does not load scheduler)')

        parser.add_argument('--save_spice', default=0, type=int, help='1 = save spice score')

        parser.add_argument('--debug', type=int, default=0)

        parser.add_argument('--sc_after', type=int, default=-1)
        parser.add_argument('--iter-mode', type=str, default='sent', help='sent|image')

        parsed, unknown = parser.parse_known_args()
        model = parsed.model.lower()
        optimizer = parsed.optimizer.lower()
        scheduler = parsed.scheduler.lower()

        if model == 'nic':
            NICModel.add_args(parser)
            self.model_cls = NICModel  # FairseqEncoderDecoder
            self.feat_mode = 'fc'
        elif model == 'topdown':
            TopDownModel.add_args(parser, )
            self.model_cls = TopDownModel
            self.feat_mode = 'att'
        elif model == 'transformer':
            TransformerModel.add_args(parser)
            self.model_cls = TransformerModel
            self.feat_mode = 'att'

        optimizer_add_args(optimizer, parser)

        if scheduler == 'steplr':
            self.scheduler_cls = StepLR
            parser.add_argument('--learning_rate_decay_start', default=-1, type=int)
            parser.add_argument('--learning_rate_decay_every', default=3, type=int)
            parser.add_argument('--learning_rate_decay_rate', default=0.8, type=float)
        else:
            print('scheduler:', scheduler)

    def init_data(self):
        dataset_name = self.args.dataset

        if self.args.debug:
            self.args.num_workers = 0
            print('debug mode, num_workers set to 0')

        dict_file = os.path.join(data_path, 'dictionary_{}.txt'.format(dataset_name))
        dictionary = load_dict(dict_file, min_word_count=5)
        dataset = CaptionDataset(dataset_name, dictionary, max_sent_len=self.args.max_sent_len,
                                 feat_type=self.feat_mode)
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
        self.scheduler = self.scheduler_cls(step_size=self.args.learning_rate_decay_every,
                                            gamma=self.args.learning_rate_decay_rate,
                                            optimizer=self.optimizer)

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
        if self.feat_mode == 'att':
            batch_data['feat_att'] = _to_tensor(batch_data['feat_att'], torch.float, device)
            batch_data['att_mask'] = _to_tensor(batch_data['att_mask'], torch.float, device)
        return batch_data

    def _get_model_input(self, batch_data):
        feat_input = {'feat_fc': _to_tensor(batch_data['feat_fc'], torch.float, device)}
        feat_input['src_tokens'] = feat_input['feat_fc']
        feat_input['src_lengths'] = None
        if self.feat_mode == 'att':
            feat_input['feat_att'] = _to_tensor(batch_data['feat_att'], torch.float, device)
            feat_input['att_mask'] = _to_tensor(batch_data['att_mask'], torch.float, device)
        return feat_input

    def train_step(self, model, batch_data):
        timer = Timer()
        sc_flag = self.args.sc_after > 0 and self.epoch >= self.args.sc_after

        feat_input = self._get_model_input(batch_data)
        tokens, token_length = (_to_tensor(batch_data[key], torch.int64, device) for key in
                                ['token_id', 'token_length'])
        token_input, token_expected = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        if not sc_flag:
            timer.tick('forward')
            _output = model.forward(
                # src_tokens=None, src_lengths=None,
                prev_output_tokens=token_input, **feat_input
            )
            timer.tock('forward')
            logits = _output[0]
            loss_ce = masked_cross_entropy(logits, token_expected, token_length - 1)
            loss_dict = {'loss_ce': loss_ce}
        else:
            if not hasattr(self, 'seq_gen'):
                seq_gen = SimpleSequenceGenerator1(bos_index=self.dictionary.bos_index,
                                                   eos_index=self.dictionary.eos_index,
                                                   unk_index=self.dictionary.unk_index,
                                                   pad_index=self.dictionary.pad_index)
                df = os.path.join(data_path, self.args.dataset, 'df_{}_words.pkl'.format(self.args.dataset))
                init_scorer(df=df)
                self.seq_gen = seq_gen
                print('hasattr', hasattr(self, 'seq_gen'))
            else:
                seq_gen = getattr(self, 'seq_gen')

            beam_size = 5
            max_length = self.args.max_sent_len

            batch_size = len(batch_data['image_id'])

            # prepare ground truth
            gts_raw = OrderedDict()
            for i in range(batch_size):
                g = [' '.join([*s['tokens'], self.dictionary.eos_word])
                     for s in self.dataset.get_ground_truth_sents(batch_data['image_id'][i])]
                gts_raw[i] = g

            timer.tick('generate')

            sample = {'id': _to_tensor(batch_data['index'], torch.int64, device), 'net_input': feat_input}
            model.eval()
            finalized = seq_gen.generate(model, sample, beam_size=beam_size, max_length=max_length)
            model.train()

            timer.tock('generate')
            timer.tick('prep output')

            # prepare output sentences
            output_tokens = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.int64, device=device)
            output_tokens.fill_(self.dictionary.pad_index)
            output_logprob = torch.zeros(size=(batch_size, max_length + 1), dtype=torch.float64, device=device)
            token_lengths = [0] * batch_size
            for i, output in finalized.items():
                tokens, logprob = output[0]['tokens'], output[0]['positional_scores']
                length = len(tokens)
                assert tokens.shape == logprob.shape
                token_lengths[i] = length
                output_tokens[i, :length] = tokens
                output_logprob[i, :length] = logprob
            # beam search output
            _gts_raw = OrderedDict()
            _res = OrderedDict()
            _res_lengths = OrderedDict()
            all_output_tokens = torch.zeros(size=(batch_size * beam_size, max_length + 1), dtype=torch.int64, device=device)
            for batch_index, output in finalized.items():
                for beam_index in range(beam_size):
                    tokens = output[beam_index]['tokens']
                    length = len(tokens)
                    i = batch_index * beam_size + beam_index
                    all_output_tokens[i, :length] = tokens
                    _res_lengths[i] = length
                    _gts_raw[i] = gts_raw[batch_index]
            all_output_tokens = all_output_tokens.detach().cpu().numpy()
            for i, tokens in enumerate(all_output_tokens):
                _res[i] = ' '.join([self.dictionary.symbols[w] for w in tokens[:_res_lengths[i]]])

            timer.tock('prep output')
            timer.tick('reward')

            # reward
            scores = get_sent_scores(_gts_raw, _res, {'cider': 1.0, 'bleu': 0.0}, n_threads=4)
            scores = scores.reshape(batch_size, beam_size)
            reward = scores[:, 0] - scores.mean(axis=1)  # shape: (batch_size,)
            _r = reward.mean()

            timer.tock('reward')
            timer.tick('loss')

            # loss
            reward = torch.Tensor(reward).to(device)
            reward = reward.unsqueeze(1).expand_as(output_logprob)
            mask = (output_tokens != self.dictionary.pad_index)
            loss_sc = - output_logprob * reward * mask
            loss_sc = torch.sum(loss_sc) / torch.sum(mask)
            loss_dict = {'loss_sc': loss_sc}

            timer.tock('loss')

        self.writer.add_scalars(main_tag='step_time', tag_scalar_dict=timer.get_time(), global_step=self.global_step)

        return loss_dict

    def test_step(self, model, batch_data):
        models = [model]
        if not hasattr(self, 'seq_gen_test'):
            seq_gen_test = SequenceGenerator(models=models, tgt_dict=self.model.decoder.dictionary,
                                             beam_size=self.args.beam_size, max_len_a=0, max_len_b=self.args.max_sent_len)
        else:
            seq_gen_test = getattr(self, 'seq_gen_test')

        image_id = batch_data['image_id']
        feat_input = self._get_model_input(batch_data)
        sample = {'net_input': feat_input}

        with torch.no_grad():
            finalized = seq_gen_test.generate(models, sample, bos_token=self.dictionary.bos_index)

        batch_output, batch_label = [], []
        for batch_index, item in enumerate(finalized):
            _image_id = image_id[batch_index]

            output_tokens = item[0]['tokens'].detach().cpu().numpy()
            output_sent = self.dictionary.string(output_tokens)
            batch_output.append({'image_id': _image_id, 'output': output_sent})

            gt_sents = self.dataset.get_ground_truth_sents(_image_id)
            # gt_sents = [' '.join(sent.words) for sent in gt_sents]
            gt_sents = [' '.join(sent['tokens']) for sent in gt_sents]
            batch_label.append({'image_id': _image_id, 'gt_sents': gt_sents})
        return batch_label, batch_output

    def calculate_metrics(self, label_list, output_list):
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
        ann_file, result_file, metric_file = \
            [os.path.join(self.save_folder, i) for i in
             ('annotation.json', 'result_{}.json'.format(self.epoch), 'metrics.csv')]
        result_generator.dump_annotation_and_output(ann_file, result_file)

        if silent:
            f = open('.eval_stdout', 'w')
        else:
            f = sys.stdout
        with redirect_stdout(f):
            metrics, img_scores = evaluate(ann_file, result_file, return_imgscores=True,
                                           use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'])
        if silent:
            f.close()
            print(metrics)

        self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=metrics, global_step=self.global_step)
        save_metrics(metric_file, metrics, epoch=self.epoch, global_step=self.global_step)
        result_generator.add_img_scores(img_scores, save_spice=self.args.save_spice)
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

