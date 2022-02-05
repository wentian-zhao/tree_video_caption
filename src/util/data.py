import os
import sys
import json
import time
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data._utils.collate import default_collate

from fairseq.data import FairseqDataset


def collate_default(field_data, **kwargs):
    return default_collate(field_data)


def collate_list(field_data, **kwargs):
    return field_data


def get_collate_ndarray(dtype=None):
    def collate_ndarray(field_data, **kwargs):
        return np.array(field_data, dtype=dtype)
    return collate_ndarray


def get_collate_tensor(dtype=None, device='cpu'):
    def collate_tensor(field_data, **kwargs):
        if isinstance(field_data[0], torch.Tensor):
            return torch.stack(field_data, dim=0)
        else:
            return torch.tensor(np.array(field_data), dtype=dtype, device=device)
    return collate_tensor


def get_collate_seq_ndarray(dtype, seq_len_field):
    def collate_seq(field_data, **kwargs):
        """

        :param field_data: [data1, data2, ..., datan]
        :param kwargs: {field: [data1, data2, ..., datan]}
        :return:
        """
        batch_data_dict = kwargs['batch_data_dict']
        lengths = batch_data_dict[seq_len_field]        # list of int
        max_seq_len = np.max(lengths)
        seq = np.array(field_data, dtype=dtype)[:, :max_seq_len]
        return seq

    return collate_seq


def get_collate_seq_tensor(dtype, seq_len_field):
    def collate_seq(field_data, **kwargs):
        """

        :param field_data: [data1, data2, ..., datan]
        :param kwargs: {field: [data1, data2, ..., datan]}
        :return:
        """
        batch_data_dict = kwargs['batch_data_dict']
        lengths = batch_data_dict[seq_len_field]        # list of int
        max_seq_len = np.max(lengths)
        seq = np.array(field_data, np.int64)[:, :max_seq_len]
        return torch.tensor(seq, dtype=dtype)

    return collate_seq


def get_collate_att_feat(dtype):
    def collate_att_feat(field_data, **kwargs):
        batch_size = len(field_data)
        lengths = [i.shape[0] for i in field_data]
        max_length = max(lengths)
        # print('att max length:', max_length)
        feat = torch.zeros(batch_size, max_length, *field_data[0].shape[1:], dtype=dtype)
        for i in range(batch_size):
            feat[i, :field_data[i].shape[0]] = field_data[i]
        return feat
    return collate_att_feat


# class CustomDataset(Dataset):
class CustomDataset(FairseqDataset):
    @abstractmethod
    def get_sub_collate_fn(self):
        pass

    def get_collate_fn(self, sort_key=None):
        """

        :param sort_key: a function
        :return:
        """
        sub_collate_fn = self.get_sub_collate_fn()      # dict
        sort_key = sort_key

        def collate_fn(data_dict_list):
            """
            :param data_dict_list: list of dicts returned by __getitem__
            :return:
            """
            specified_fields = sub_collate_fn.keys()
            actual_fields = data_dict_list[0].keys()
            assert specified_fields == actual_fields, \
                'keys specified by get_sub_collate_fn: {}, actual keys: {}'.format(specified_fields, actual_fields)

            if sort_key is not None:
                data_dict_list.sort(key=sort_key)

            batch_data_dict = {field: [d[field] for d in data_dict_list] for field in actual_fields}
            collated_batch_data_dict = {}
            for field, _data in batch_data_dict.items():
                _data = [d[field] for d in data_dict_list]
                collated_batch_data_dict[field] = sub_collate_fn[field](field_data=_data, batch_data_dict=batch_data_dict)

            return collated_batch_data_dict

        return collate_fn

    def collater(self, samples):
        """
        compatibility for FairseqDataset
        :param samples:
        :return:
        """
        if not hasattr(self, 'collate_fn'):
            self.collate_fn = self.get_collate_fn()
        return self.collate_fn(samples)

    @abstractmethod
    def get_split_index(self, split, **kwargs):
        """
        :return: {'train': train_indices, 'val': val_indices, ...}[split]
        indices must be indexable
        """
        pass

    def get_split_dataloader(self, split, dataloader_kwargs, get_split_index_kwargs=None, sort_key=None):
        """

        :param split: 'all', 'train', 'test', 'val', ...
        :param dataloader_kwargs:
        :param get_split_index_kwargs:
        :param sort_key:
        :return:
        """
        dataloader_kwargs['collate_fn'] = self.get_collate_fn(sort_key)

        if get_split_index_kwargs is None:
            get_split_index_kwargs = {}
        indices = self.get_split_index(split, **get_split_index_kwargs)
        subset = Subset(self, indices)

        print('split:', split, 'shuffle:', dataloader_kwargs.get('shuffle', False), 'length:', len(indices))

        split_dataloader = DataLoader(dataset=subset, **dataloader_kwargs)
        return split_dataloader


class CustomSubset(FairseqDataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def collater(self, samples):
        return self.dataset.collater(samples)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)


class _Dataset1(CustomDataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        return {'field1': index, 'field2': index * 5, 'field3': str(index)}

    def __len__(self):
        return self.length

    def get_sub_collate_fn(self):
        return {'field1': get_collate_tensor(torch.int64),
                'field2': get_collate_ndarray(np.float),
                'field3': collate_list}


if __name__ == '__main__':
    dataset = _Dataset1(length=1000)
    dataloader = dataset.get_dataloader(batch_size=10)

    for i, batch_data in enumerate(dataloader):
        field1 = batch_data['field1']
        print(field1.dtype, isinstance(field1, torch.LongTensor))
        print(batch_data)