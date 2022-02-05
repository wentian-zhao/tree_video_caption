import time
from collections import OrderedDict

import numpy as np
import torch

import multiprocessing
from multiprocessing import Pool, Process

from .bleu.bleu import *
from .ciderD.ciderD import *

ciderD_scorer = None
bleu_scorer = None

_df = None
_bleu_n = 4

pool = None


def init_scorer(df, bleu_n=4):
    global ciderD_scorer
    global bleu_scorer
    global _df
    global _gts
    _df = df

    if ciderD_scorer is None:
        print('init CIDEr-D scorer using df {}'.format(df))
        ciderD_scorer = CiderD(df=df)

    if bleu_scorer is None:
        bleu_scorer = Bleu(bleu_n)


def _compute_cider(args):
    init_scorer(args['df'], args['bleu_n'])
    gts, res, weight = (args[key] for key in ['gts', 'res', 'weight'])
    res_ = [{'image_id': i, 'caption': [res[i]]} for i in res.keys()]
    _, score_cider = ciderD_scorer.compute_score(gts, res_)
    return score_cider * weight['cider']


def _compute_bleu(args):
    gts, res, weight = (args[key] for key in ['gts', 'res', 'weight'])
    res__ = {i: [res[i]] for i in res.keys()}
    _, score_bleu = bleu_scorer.compute_score(gts, res__)
    score_bleu4 = np.array(score_bleu[3])
    return score_bleu4 * weight['bleu']


def get_scores(res, gts, weights, n_threads=4):
    """
        :param res: ['candidate1', 'candidate2']
        :param gts: {0: ['sent1', 'sent2'], 1: ['sent3', 'sent4']}
        :param weights: {'cider': 0.5, 'bleu': 0.5}
        :return:
        """
    global pool
    if pool is None:
        pool = Pool(n_threads)

    def _get_chunk_index(n_samples, n_chunks):
        chunks = []
        chunk_size = max(1, n_samples // n_chunks)

        for i in range(n_chunks):
            l, r  = i * chunk_size, (i + 1) * chunk_size
            if i == (n_chunks - 1) and (n_samples - r + 1) < (2 * chunk_size):
                r = n_samples
            chunks.append(range(l, r))
            if r >= n_samples:
                break
        return chunks

    n_samples = len(res)
    chunk_index = _get_chunk_index(n_samples, n_threads)

    chunked_args = []

    for i in range(n_threads):
        _gts = {}
        _res = OrderedDict()
        for _i in chunk_index[i]:
            _gts[_i] = gts[_i]
            _res[_i] = res[_i]

        chunked_args.append({'df': _df, 'bleu_n': _bleu_n, 'gts': _gts, 'res': _res, 'weight': weights})

    score = 0.

    # single thread
    if weights['cider'] > 0:
        score_cider = pool.map(_compute_cider, chunked_args)
        score_cider = np.concatenate(score_cider)
        score = score_cider + score
    if weights['bleu'] > 0:
        score_bleu4 = pool.map(_compute_bleu, chunked_args)
        score_bleu4 = np.concatenate(score_bleu4)
        score = score_bleu4 + score

    return score


# @arg_type(log_prob=torch.Tensor, generated_seq=np.ndarray, reward=np.ndarray)
def rl_criterion(log_prob, generated_seq, reward):
    """
    :param log_prob: (batch_size, max_len), log probability
    :param generated_seq: (batch_size, max_len)
    :param reward: (batch_size, max_len)
    :return:
    """
    device = log_prob.device

    batch_size, max_len = log_prob.shape[:2]

    assert (batch_size, max_len) == (reward.shape[0], reward.shape[1])
    assert (batch_size, max_len) == (generated_seq.shape[0], generated_seq.shape[1])

    mask = np.zeros(shape=(batch_size, max_len), dtype=np.float32)
    for i in range(batch_size):
        for j in range(max_len):
            mask[i][j] = 1
            if generated_seq[i][j] == Vocabulary.end_token_id:
                break
    mask = torch.Tensor(mask).to(device)
    reward = torch.Tensor(reward).to(device)    # (batch_size, max_len)

    log_prob_flat = log_prob.view(-1, 1)
    log_probs_seq = log_prob_flat
    reward_flat = reward.reshape(shape=(-1, 1))    # (batch_size * max_len, 1)
    mask_flat = mask.reshape(shape=(-1, 1))
    loss = - log_probs_seq * reward_flat * mask_flat
    loss = torch.sum(loss) / torch.sum(mask)
    return loss


def get_sent_scores(gts, res, weights, n_threads=4):
    """
    :param gts: OrderedDict, {image_id_1: ['gt 1', 'gt 2'], image_id_2: ['gt 3', 'gt 4']}
    :param res: OrderedDict, {image_id_1: 'sentence 1', image_id_2: 'sentence 2'}
    :param weight: {'cider': ?, 'bleu': ?}
    :return:
    """
    return get_scores(res=res, gts=gts, weights=weights, n_threads=n_threads)

    # def _compute_cider(gts, res):
    #     res_ = [{'image_id': i, 'caption': [res[i]]} for i in res.keys()]
    #     _, score_cider = ciderD_scorer.compute_score(gts, res_)
    #     return score_cider * weight['cider']
    #
    # def _compute_bleu(gts, res):
    #     res__ = {i: [res[i]] for i in res.keys()}
    #     _, score_bleu = bleu_scorer.compute_score(gts, res__)
    #     score_bleu4 = np.array(score_bleu[3])
    #     return score_bleu4 * weight['bleu']
    #
    # score = 0.
    # if weight['cider'] > 0:
    #     score += _compute_cider(gts, res)
    # if weight['bleu'] > 0:
    #     score += _compute_bleu(gts, res)
    # return score
