import math

import numpy as np
import torch

from .fairseq_lstm import NICModel
from .fairseq_topdown import TopDownModel
from .fairseq_transformer import TransformerModel
from .fairseq_tree import TopDownTreeModel
from util import Timer


def log_softmax(x, dim=-1):
    return x - torch.log(torch.exp(x).sum(dim=dim)).unsqueeze(dim).expand_as(x)


# greedy decoding / sampling
class GreedySequenceGenerator:
    def __init__(self, bos_index, eos_index, unk_index, pad_index):
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.pad_index = pad_index

    def generate(self, model, sample, sample_method='greedy',
                 max_length=20, temperature=1.0, unk_penalty=0.0, dictionary=None):
        bos_index, eos_index, unk_index, pad_index = self.bos_index, self.eos_index, self.unk_index, self.pad_index

        inf = math.inf

        net_input = sample['net_input']
        batch_size, device = net_input['src_tokens'].shape[0], net_input['src_tokens'].device
        dtype = net_input['src_tokens'].dtype

        enc_out = model.encoder(**net_input)

        candidates = {i: [] for i in range(batch_size)}

        prev_output_tokens = torch.zeros((batch_size, 1), dtype=torch.int64, device=device) + self.bos_index
        all_output_tokens = torch.zeros((batch_size, 0), dtype=torch.int64, device=device)  # (batch_size, length)
        seq_lengths = torch.zeros((batch_size,), dtype=torch.int64)
        all_log_probs = torch.zeros((batch_size, 0), dtype=torch.float, device=device)  # (batch_size, length)
        additional_output_list = []

        end_flag = torch.zeros((batch_size,), dtype=torch.bool, device=device)  # False

        incremental_state = {}

        for step in range(max_length + 1):
            _dec_out = model.decoder(prev_output_tokens, encoder_out=enc_out, incremental_state=incremental_state)
            dec_out = _dec_out[0]
            additional_output = _dec_out[2]

            dec_out = dec_out[:, -1, :].div(temperature)  # (batch_size, vocab_size)
            log_probs = log_softmax(dec_out, dim=1)  # FIXME: inplace operation?

            log_probs[:, pad_index] = -inf  # pad
            log_probs[:, unk_index] = log_probs[:, unk_index] - unk_penalty  # unk penalty; avoid in-place operation
            if step == max_length:  # max length constraint
                log_probs[:, :eos_index] = -inf
                log_probs[:, (eos_index + 1):] = -inf

            if sample_method == 'greedy':
                token_log_prob, token = torch.max(log_probs, dim=-1)  # (batch_size,)
            else:
                token = torch.multinomial(torch.exp(log_probs), num_samples=1)
                token_log_prob = log_probs.gather(index=token, dim=1)
                token_log_prob, token = token_log_prob.squeeze(1), token.squeeze(1)

            eos_mask = (token == eos_index)
            seq_lengths[eos_mask & (~end_flag)] = step + 1
            end_flag |= eos_mask

            prev_output_tokens = token.unsqueeze(1)
            all_log_probs = torch.cat((all_log_probs, token_log_prob.unsqueeze(1)), dim=1)
            all_output_tokens = torch.cat((all_output_tokens, token.unsqueeze(1)), dim=1)
            additional_output_list.append(additional_output)

            if end_flag.all():
                break

        for i in range(batch_size):
            length = seq_lengths[i]
            _seq_tokens = all_output_tokens[i][:length]
            _seq_log_prob = all_log_probs[i][:length]
            _additional = additional_output_list[-1][i]

            score = _seq_log_prob.sum() / length
            candidates[i].append({
                "tokens": _seq_tokens,  # tensor
                "score": score,  # tensor
                "length": length,
                "positional_scores": _seq_log_prob,  # tensor
                "attention": None,
                "alignment": torch.empty(0),
                "additional": _additional
            })
        return candidates


# beam search
class SimpleSequenceGenerator1:
    def __init__(self, bos_index, eos_index, unk_index, pad_index):
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.pad_index = pad_index

    def generate(self, model, sample, beam_size=5, max_length=20, temperature=1.0, unk_penalty=0.0, dictionary=None):
        bos_index, eos_index, unk_index, pad_index = self.bos_index, self.eos_index, self.unk_index, self.pad_index

        inf = math.inf
        timer = Timer()

        net_input = sample['net_input']
        batch_size, device = net_input['src_tokens'].shape[0], net_input['src_tokens'].device
        dtype = net_input['src_tokens'].dtype

        timer.tick('encode')
        # encoder
        enc_out = model.encoder(**net_input)
        timer.tock('encode')

        candidates = {i: [] for i in range(batch_size)}

        prev_output_tokens = torch.zeros((batch_size * beam_size, 1), dtype=torch.int64, device=device) + self.bos_index
        all_log_probs = \
            torch.zeros((batch_size * beam_size, 0), dtype=torch.float,
                        device=device)  # (batch_size * beam_size, length)
        all_tokens = prev_output_tokens  # (batch_size * beam_size, length)

        new_order = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, beam_size).flatten()  # (batch_size * beam_size)
        enc_out = model.encoder.reorder_encoder_out(enc_out, new_order)
        incremental_state = {}

        for step in range(max_length + 1):
            # use all previous generated tokens
            _dec_out = model.decoder(prev_output_tokens=all_tokens,
                                       encoder_out=enc_out, incremental_state=incremental_state)
            dec_out = _dec_out[0]

            dec_out = dec_out[:, -1, :].div(temperature)  # dec_out: (batch_size * beam_size, vocab_size)
            # log_probs = model.get_normalized_probs((dec_out, _), log_probs=True)    # log_probs: (batch_size * beam_size, vocab_size)
            # log_probs = F.log_softmax(dec_out, dim=1)       # (batch_size * beam_size, vocab_size)
            log_probs = log_softmax(dec_out, dim=1)  # FIXME: inplace operation?
            vocab_size = log_probs.shape[-1]

            log_probs[:, pad_index] = -inf  # pad
            log_probs[:, unk_index] -= unk_penalty  # unk penalty
            if step == max_length:  # max length constraint
                log_probs[:, :eos_index] = -inf
                log_probs[:, (eos_index + 1):] = -inf

            # beam search step
            if step == 0:
                _log_probs = log_probs[::beam_size, :].contiguous()
                log_prob_sum = _log_probs  # log_prob_sum: (batch_size * 1, vocab_size)
            else:
                _log_probs = log_probs.reshape(batch_size * beam_size, vocab_size)
                # log_prob_sum: (batch_size * beam_size, vocab_size)
                log_prob_sum = _log_probs \
                               + all_log_probs[:, :step].sum(dim=-1).unsqueeze(-1).expand_as(_log_probs)

            _log_prob_sum = log_prob_sum.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            k = min(beam_size * 2, _log_prob_sum.shape[-1])
            scores, idx = torch.topk(_log_prob_sum, k=k, dim=1)  # scores: (batch_size, k)
            cand_index = idx // vocab_size  # (batch_size, k)   range: [0, beam_size]
            tokens = idx.fmod(vocab_size)  # (batch_size, k)   range: [0, vocab_size]
            token_log_prob = torch.gather(  # (batch_size, k)
                _log_probs.view(batch_size, -1), dim=1, index=idx
            )

            eos_mask = (tokens == eos_index) & (scores != -inf)  # (batch_size, k)

            batch_offset = (torch.arange(batch_size) * beam_size).unsqueeze(1).expand_as(cand_index).to(
                device)  # (batch_size, k)
            l_batch_index = batch_offset + cand_index  # index in the large batch, shape: (batch_size, k)

            # expand
            _all_log_probs = torch.cat([
                all_log_probs.index_select(index=l_batch_index.flatten(), dim=0),  # (batch_size * k, length)
                token_log_prob.flatten().unsqueeze(1)  # (batch_size * k, 1)
            ], dim=1)

            _all_tokens = torch.cat([
                all_tokens.index_select(index=l_batch_index.flatten(), dim=0),  # (batch_size * k, length)
                tokens.flatten().unsqueeze(1)
            ], dim=1)

            _log_prob_sum = _all_log_probs.sum(dim=1)  # (batch_size * k)

            # remove ended sentences
            if step > 0 and eos_mask[:, :beam_size].any():  # only consider <eos> when among top beam_size candidates
                eos_mask[:, beam_size:] = False
                idx_eos = eos_mask.nonzero(as_tuple=False)  # (?, 2)
                batch_index = idx_eos[:, 0].detach().cpu().numpy()  # range: [0, batch_size]

                idx_flatten = eos_mask.flatten().nonzero(as_tuple=False).squeeze(
                    1).detach().cpu().numpy()  # length = (?)
                for i in range(idx_flatten.shape[0]):
                    _batch_index = batch_index[i]
                    if len(candidates[_batch_index]) >= beam_size:  # TODO: use mask to represent ended sentences
                        continue

                    _index = idx_flatten[i]
                    _seq_log_prob = _all_log_probs[_index]
                    _seq_tokens = _all_tokens[_index][1:]  # skip <bos>

                    assert len(_seq_tokens) == len(_seq_log_prob)

                    candidates[_batch_index].append({
                        "tokens": _seq_tokens,  # tensor
                        "score": _seq_log_prob.sum() / len(_seq_tokens),  # tensor
                        "length": len(_seq_tokens),
                        "positional_scores": _seq_log_prob,  # tensor
                        "attention": None,
                        "alignment": torch.empty(0),
                    })
                # TODO: ?
                eos_mask = (tokens == eos_index) & (scores != -inf)
                _log_prob_sum[eos_mask.reshape(batch_size * k)] = -inf

            num_candidates = np.array([len(v) for v in candidates.values()])
            if (num_candidates >= beam_size).all():
                break

            # expand
            _log_prob_sum = _log_prob_sum.reshape(batch_size, k)
            _, new_beam_index = torch.topk(_log_prob_sum, dim=1, k=beam_size)  # new_beam_index: (batch_size, beam_size)

            batch_offset = (torch.arange(batch_size) * k).unsqueeze(1).expand_as(new_beam_index).to(device)
            # index in the large batch
            batch_index = (batch_offset + new_beam_index).flatten()  # shape: (batch_size * beam_size,)

            all_tokens = _all_tokens.index_select(index=batch_index, dim=0)
            all_log_probs = _all_log_probs.index_select(index=batch_index, dim=0)
            model.decoder.reorder_incremental_state(
                incremental_state,
                l_batch_index.flatten().index_select(index=batch_index, dim=0).flatten()
            )

        for batch_index, c in candidates.items():
            c.sort(key=lambda x: x['score'], reverse=True)
            candidates[batch_index] = c[:beam_size]

        return candidates


# beam search
class SimpleSequenceGeneratorTree:
    def __init__(self, bos_index, eos_index, unk_index, pad_index):
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.pad_index = pad_index

    def generate(self, model, sample, beam_size=5, max_length=20, temperature=1.0, unk_penalty=0.0, dictionary=None):
        bos_index, eos_index, unk_index, pad_index = self.bos_index, self.eos_index, self.unk_index, self.pad_index

        inf = math.inf
        timer = Timer()

        net_input = sample['net_input']
        batch_size, device = net_input['src_tokens'].shape[0], net_input['src_tokens'].device
        dtype = net_input['src_tokens'].dtype

        timer.tick('encode')
        # encoder
        enc_out = model.encoder(**net_input)
        timer.tock('encode')

        candidates = {i: [] for i in range(batch_size)}

        prev_output_tokens = torch.zeros((batch_size * beam_size, 1), dtype=torch.int64, device=device) + self.bos_index
        all_log_probs = \
            torch.zeros((batch_size * beam_size, 0), dtype=torch.float,
                        device=device)  # (batch_size * beam_size, length)
        all_tokens = prev_output_tokens  # (batch_size * beam_size, length)

        new_order = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size,
                                                                                beam_size).flatten()  # (batch_size * beam_size)
        enc_out = model.encoder.reorder_encoder_out(enc_out, new_order)
        incremental_state = {}

        for step in range(max_length + 1):
            # use all previous generated tokens
            _dec_out = model.decoder(prev_output_tokens=all_tokens,
                                     encoder_out=enc_out, incremental_state=incremental_state)

            dec_out = _dec_out[0]
            additional_output = _dec_out[2]

            dec_out = dec_out[:, -1, :].div(temperature)  # dec_out: (batch_size * beam_size, vocab_size)
            # log_probs = model.get_normalized_probs((dec_out, _), log_probs=True)    # log_probs: (batch_size * beam_size, vocab_size)
            # log_probs = F.log_softmax(dec_out, dim=1)       # (batch_size * beam_size, vocab_size)
            log_probs = log_softmax(dec_out, dim=1)  # log prob for words  # FIXME: inplace operation?
            vocab_size = log_probs.shape[-1]

            log_probs[:, pad_index] = -inf  # pad
            log_probs[:, unk_index] -= unk_penalty  # unk penalty
            if step == max_length:  # max length constraint
                log_probs[:, :eos_index] = -inf
                log_probs[:, (eos_index + 1):] = -inf

            # beam search step
            if step == 0:
                _log_probs = log_probs[::beam_size, :].contiguous()
                log_prob_sum = _log_probs  # log_prob_sum: (batch_size * 1, vocab_size)
            else:
                _log_probs = log_probs.reshape(batch_size * beam_size, vocab_size)
                # log_prob_sum: (batch_size * beam_size, vocab_size)
                log_prob_sum = _log_probs \
                               + all_log_probs[:, :step].sum(dim=-1).unsqueeze(-1).expand_as(_log_probs)

            _log_prob_sum = log_prob_sum.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            k = min(beam_size * 2, _log_prob_sum.shape[-1])         # k <= beam_size * 2
            scores, idx = torch.topk(_log_prob_sum, k=k, dim=1)  # scores: (batch_size, k)
            cand_index = idx // vocab_size  # (batch_size, k)   range: [0, beam_size]
            tokens = idx.fmod(vocab_size)  # (batch_size, k)   range: [0, vocab_size]
            token_log_prob = torch.gather(  # (batch_size, k)
                _log_probs.view(batch_size, -1), dim=1, index=idx
            )

            eos_mask = (tokens == eos_index) & (scores != -inf)  # (batch_size, k)

            batch_offset = (torch.arange(batch_size) * beam_size).unsqueeze(1).expand_as(cand_index).to(
                device)  # (batch_size, k)
            l_batch_index = batch_offset + cand_index  # index in the large batch, shape: (batch_size, k)

            # expand
            _all_token_log_probs = torch.cat([
                all_log_probs.index_select(index=l_batch_index.flatten(), dim=0),  # (batch_size * k, length)
                token_log_prob.flatten().unsqueeze(1)  # (batch_size * k, 1)
            ], dim=1)           # (batch_size * k, length + 1)

            _all_tokens = torch.cat([
                all_tokens.index_select(index=l_batch_index.flatten(), dim=0),  # (batch_size * k, length)
                tokens.flatten().unsqueeze(1)
            ], dim=1)       # (batch_size * k, length + 1)

            _all_additional_output = model.decoder.reorder_additional_output(
                additional_output, new_order=l_batch_index.flatten()
            )

            _log_prob_sum = _all_token_log_probs.sum(dim=1)  # (batch_size * k)

            # remove ended sentences
            if step > 0 and eos_mask[:, :beam_size].any():  # only consider <eos> when among top beam_size candidates
                eos_mask[:, beam_size:] = False
                idx_eos = eos_mask.nonzero(as_tuple=False)  # (?, 2)
                batch_index = idx_eos[:, 0].detach().cpu().numpy()  # range: [0, batch_size]

                idx_flatten = eos_mask.flatten().nonzero(as_tuple=False).squeeze(
                    1).detach().cpu().numpy()  # length = (?)
                for i in range(idx_flatten.shape[0]):
                    _batch_index = batch_index[i]
                    if len(candidates[_batch_index]) >= beam_size:  # TODO: use mask to represent ended sentences
                        continue

                    _index = idx_flatten[i]
                    _seq_log_prob = _all_token_log_probs[_index]
                    _seq_tokens = _all_tokens[_index][1:]  # skip <bos>

                    _additional_output = _all_additional_output[_index]     # TODO: change to reorder_additional_output

                    assert len(_seq_tokens) == len(_seq_log_prob)

                    candidates[_batch_index].append({
                        "tokens": _seq_tokens,  # tensor
                        "score": _seq_log_prob.sum() / len(_seq_tokens),  # tensor
                        "length": len(_seq_tokens),
                        "positional_scores": _seq_log_prob,  # tensor
                        "attention": None,
                        "alignment": torch.empty(0),
                        "additional": _additional_output
                    })
                # TODO: ?
                eos_mask = (tokens == eos_index) & (scores != -inf)
                _log_prob_sum[eos_mask.reshape(batch_size * k)] = -inf

            num_candidates = np.array([len(v) for v in candidates.values()])
            if (num_candidates >= beam_size).all():
                break

            # expand
            _log_prob_sum = _log_prob_sum.reshape(batch_size, k)
            _, new_beam_index = torch.topk(_log_prob_sum, dim=1, k=beam_size)  # new_beam_index: (batch_size, beam_size)

            batch_offset = (torch.arange(batch_size) * k).unsqueeze(1).expand_as(new_beam_index).to(device)
            # index in the large batch
            batch_index = (batch_offset + new_beam_index).flatten()  # shape: (batch_size * beam_size,)

            all_tokens = _all_tokens.index_select(index=batch_index, dim=0)
            all_log_probs = _all_token_log_probs.index_select(index=batch_index, dim=0)
            model.decoder.reorder_incremental_state(
                incremental_state,
                l_batch_index.flatten().index_select(index=batch_index, dim=0).flatten()
            )

        for batch_index, c in candidates.items():
            c.sort(key=lambda x: x['score'], reverse=True)
            candidates[batch_index] = c[:beam_size]

        return candidates

