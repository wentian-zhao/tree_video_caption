import traceback
from abc import abstractmethod

from tqdm import tqdm
import torch

from util.pipeline import SupervisedPipeline
from util import Timer
from util import clip_gradient_norm


class DataPrefetcher:
    def __init__(self, loader, to_device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.to_device = to_device
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self.to_device(self.next_batch)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.next_batch
        self.preload()
        return batch_data

    def next(self):
        return self.__next__()


class CustomTask(SupervisedPipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super(CustomTask, self).add_arguments(parser)
        parser.add_argument('--fp16', type=int, default=0)

        parser.add_argument('--model', type=str, default='nic')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--scheduler', type=str, default='steplr')
        parser.add_argument('--dataset', type=str, default='coco')

        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--batch-size-test', type=int, default=64)
        parser.add_argument('--num-workers', type=int, default=1)

        parser.add_argument('--update-interval', type=int, default=1, help='1 -> call optimizer.step() every batch')

        # parsed, unknown = parser.parse_known_args()
        # model = parsed.model.lower()
        # optimizer = parsed.optimizer.lower()
        # scheduler = parsed.scheduler.lower()

    def init_data(self):
        pass

    def init_model(self, state_dict=None):
        pass

    def _prefetch_to_device(self, batch_data):
        return self.prefetch_to_device(batch_data)

    def prefetch_to_device(self, batch_data):
        raise NotImplementedError

    def train_epoch(self):
        if self.args.fp16:
            if not hasattr(self, 'grad_scaler'):
                grad_scaler = torch.cuda.amp.GradScaler()
                self.grad_scaler = grad_scaler
            else:
                grad_scaler = self.grad_scaler

        model, optimizer, scheduler = self.model, self.optimizer, self.scheduler

        model.train()

        dataloader = self.dataloaders['train']

        timer = Timer(); timer.tick('step')
        batch_data = None

        for i, param_group in enumerate(optimizer.param_groups):
            print('epoch', self.epoch)
            print('learning rate:', param_group['lr'])

        def _loop_body(batch_data, last_batch=False):
            timer.tick('prep')

            # if torch.cuda.is_available(): torch.cuda.synchronize()
            model.train()
            # optimizer.zero_grad()

            timer.tock('prep'); timer.tick('forward')

            if self.args.fp16:
                with torch.cuda.amp.autocast():
                    loss_dict = self.train_step(self.model, batch_data)
            else:
                loss_dict = self.train_step(self.model, batch_data)
            if loss_dict is None: return

            timer.tock('forward'); timer.tick('backward')

            if 'loss_total' not in loss_dict:
                loss = sum(value for key, value in loss_dict.items())
                loss_dict['loss_total'] = loss
            else:
                loss = loss_dict['loss_total']

            timer.tick('loss.backward')
            if self.args.fp16:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            timer.tock('loss.backward')

            if self.args.grad_clip_method == 'value':
                # clip_gradient_norm(self.optimizer, self.args.grad_clip)
                torch.nn.utils.clip_grad_value_(model.parameters(), self.args.grad_value)
            elif self.args.grad_clip_method == 'norm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_norm)

            timer.tick('optimizer.step')
            if ((self.num_batches + 1) % self.args.update_interval) == 0 or last_batch:
                if self.args.fp16:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            # if torch.cuda.is_available(): torch.cuda.synchronize()
            timer.tock('optimizer.step')

            timer.tock('backward'); timer.tick('log')

            _loss_dict = {key: float(value.detach().cpu().numpy()) for key, value in loss_dict.items()}
            if True or self.global_step % 5 == 0:
                self.writer.add_scalars(main_tag='loss', tag_scalar_dict=_loss_dict, global_step=self.global_step)
                for i, param_group in enumerate(optimizer.param_groups):
                    self.writer.add_scalar('lr/{}'.format(i), scalar_value=param_group['lr'],
                                           global_step=self.global_step)
            if self.global_step % 500 == 0:
                print('loss: {:.4f}'.format(_loss_dict['loss_total']))

            timer.tock('log')

            self.global_step += 1

        # normal
        # for i, batch_data in tqdm(enumerate(dataloader), ncols=64, total=len(dataloader)):
        #     _loop_body()

        pbar = tqdm(ncols=64, total=len(dataloader))
        it = iter(dataloader)
        i = 1
        batch_data = next(it)
        next_batch = None
        while batch_data is not None:
            self.num_batches = 0
            # if i > 2000: break
            # if i > 5: break
            timer.tick('read_data')
            try:
                next_batch = next(it)
                last_batch = False
            except StopIteration:
                next_batch = None
                last_batch = True
            timer.tock('read_data')
            timer.tick('loop_body')

            try:
                _loop_body(batch_data, last_batch=last_batch)
            except KeyboardInterrupt:
                import IPython; IPython.embed()
            except:
                print('exception in training:')
                traceback.print_exc()
            self.num_batches += 1
            batch_data = next_batch
            timer.tock('loop_body')

            self.writer.add_scalars(main_tag='time', tag_scalar_dict=timer.get_time(), global_step=self.global_step)
            timer.clear()

            pbar.update(1); i += 1
        pbar.close()

        # prefetch
        # prefetcher = DataPrefetcher(dataloader, self._prefetch_to_device)
        # pbar = tqdm(ncols=64, total=len(dataloader))
        # batch_data = prefetcher.next()
        # i = 0
        # while batch_data is not None:
        #     _loop_body()
        #     pbar.update(1)
        #     batch_data = prefetcher.next()
        #     i += 1

        # update learning rate here
        if self.epoch >= self.args.learning_rate_decay_start and self.args.learning_rate_decay_start > 0:
            self.scheduler.step()

    def test_epoch(self):
        sc_flag = self.args.sc_after > 0 and self.epoch >= self.args.sc_after

        if self.args.action == 'train' and (not sc_flag) and self.args.dataset == 'activitynet' and self.epoch % 2 != 0:
            return

        try:
            model = self.model
            model.eval()

            dataloader = self.dataloaders['test']
            all_label = []
            all_output = []
            self.begin_test()
            for i, batch_data in tqdm(enumerate(dataloader), ncols=64, total=len(dataloader)):
                # if i > 10: break
                label, output = self.test_step(self.model, batch_data)
                all_label.extend(label)
                all_output.extend(output)
            assert len(all_label) == len(all_output)
            self.calculate_metrics(all_label, all_output)
        except KeyboardInterrupt:
            return

    @abstractmethod
    def train_step(self, model, batch_data):
        """
        :param batch_data: from dataloader
        :return: {'loss_total': ..., 'loss_1': ..., 'loss_2': ...}
        """
        pass

    @abstractmethod
    def begin_test(self):
        pass

    @abstractmethod
    def test_step(self, model, batch_data):
        """
        :param batch_data: from dataloader
        :return: ground_truth: list, output: list
        """
        pass

    @abstractmethod
    def calculate_metrics(self, label_list, output_list):
        pass