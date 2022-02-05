import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.data import Dictionary

from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, transformer, FairseqEncoderDecoderModel
from fairseq.models.fairseq_encoder import EncoderOut


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):        # TODO: ?
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn      # MultiHeadAttention
        self.src_attn = src_attn        # MultiHeadAttention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        """
        :param x:           (batch_size, tgt_len, d_model)
        :param memory:      (batch_size, src_len, d_model)
        :param src_mask:    (batch_size, 1, src_len)
        :param tgt_mask:    (batch_size, tgt_len, tgt_len)
        :return: 
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# _max_size = 100
# _mask_cache = {}
# def subsequent_mask(size, device):
#     "Mask out subsequent positions."
#     global _max_size
#     key = device.type
#     if key not in _mask_cache or (key in _mask_cache and size > _mask_cache[key].shape[0]):
#         _max_size = max(_max_size, size)
#         _mask_cache[key] = torch.triu(torch.ones(1, _max_size, _max_size, dtype=torch.uint8)) == 0
#     return _mask_cache[key][:, :size, :size]

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
        mask: 0 -> mask out; 1 -> keep
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # # original
        # scores = scores.masked_fill(mask == 0, -1e9)
        # for fp16
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
            p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary: Dictionary, feat_dim=2048, n_layer=6, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1,
                 use_pe=True):
        super().__init__(dictionary)

        self.fc_embed = nn.Linear(in_features=feat_dim, out_features=d_model)
        self.att_embed = nn.Linear(in_features=feat_dim, out_features=d_model)
        self.dropout = dropout

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_head, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_feedforward)
        self.encoder = Encoder(
            layer=EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout),
            N=n_layer
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.use_pe = use_pe
        if self.use_pe:
            print('using positional encoding in encoder')
            self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        :param src_tokens: {'feat_fc': (batch_size, feat_dim)}
        :param src_lengths:
        :param kwargs:
        :return:
        """
        feat_fc = kwargs['feat_fc']
        att_feats = kwargs['feat_att']      # (batch_size, ?, feat_dim)
        att_masks = kwargs['att_mask']

        feat_fc = self.fc_embed(feat_fc)
        att_feats = self.att_embed(att_feats)

        if self.use_pe:
            att_feats = self.pe(att_feats)
        else:       # TODO: where to put dropout?
            att_feats = F.dropout(att_feats, p=self.dropout, training=self.training)

        att_masks = att_masks.unsqueeze(-2)
        memory = self.encoder.forward(att_feats, att_masks)

        return {'feat_fc': feat_fc, 'memory': memory, 'att_mask': att_masks}

    def reorder_encoder_out(self, encoder_out, new_order):
        feat_fc, memory, att_masks = [encoder_out[key] for key in ('feat_fc', 'memory', 'att_mask')]
        feat_fc = feat_fc.index_select(dim=0, index=new_order)
        memory = memory.index_select(dim=0, index=new_order)
        att_masks = att_masks.index_select(dim=0, index=new_order)
        return {'feat_fc': feat_fc, 'memory': memory, 'att_mask': att_masks}


class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, n_layer=6, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1):
        super().__init__(dictionary)

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_head, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_feedforward)
        position = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.tgt_embed = Embeddings(d_model, len(self.dictionary))

        self.decoder = Decoder(
            layer=DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=dropout),
            N=n_layer
        )

        self.output_proj = nn.Linear(in_features=d_model, out_features=len(dictionary))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.embed_positions = position

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """

        :param prev_output_tokens: (batch_size, seq_len)
        :param encoder_out:
        :param incremental_state:
        :param kwargs:
        :return:
        """

        def _ss_get_input_token(step, input_tokens, prev_logit, ss_prob):
            """

            :param step: int
            :param input_tokens: (batch_size,)
            :param prev_output_tokens: (batch_size,)
            :return:
            """
            sample_mask = input_tokens.new_zeros(size=input_tokens.shape, dtype=torch.float).uniform_(0, 1) < ss_prob
            if step == 0 or sample_mask.sum() == 0:
                return input_tokens
            else:
                index = sample_mask.nonzero().view(-1)
                prev_prob = torch.exp(F.log_softmax(prev_logit, dim=-1))
                input_tokens.index_copy(0, index, torch.multinomial(prev_prob, 1).view(-1).index_select(0, index))
                return input_tokens

        if self.training and kwargs.get('ss_prob', 0) > 0:      # scheduled sampling
            # FIXME: slow
            memory, att_mask = encoder_out['memory'], encoder_out['att_mask']

            ss_prob = kwargs['ss_prob']
            batch_size, max_seq_len = prev_output_tokens.shape[:2]
            prev_logit = None
            all_logprobs = []
            all_input_tokens = []
            for i in range(max_seq_len):
                ss_input_tokens = _ss_get_input_token(i, prev_output_tokens[:, i], prev_logit, ss_prob)
                all_input_tokens.append(ss_input_tokens)

                input_tokens = torch.stack(all_input_tokens, dim=1)     # (batch_size, max_len)

                tgt_mask = input_tokens != self.dictionary.pad_index
                tgt_mask = tgt_mask.unsqueeze(-2)
                _future_mask = subsequent_mask(input_tokens.shape[1]).to(tgt_mask)
                tgt_mask = tgt_mask & _future_mask

                dec_input = self.tgt_embed(input_tokens)
                dec_input = self.embed_positions(dec_input)

                dec_out = self.decoder(dec_input, memory=memory, src_mask=att_mask, tgt_mask=tgt_mask)
                dec_out = dec_out[:, -1]        # (batch_size, vocab_size)
                prev_logit = self.output_proj(dec_out)      # (batch_size, vocab_size)
                all_logprobs.append(prev_logit)

            return torch.stack(all_logprobs, dim=1), None     # (batch_size, max_len, vocab_size)
        else:
            memory, att_mask = encoder_out['memory'], encoder_out['att_mask']
            dec_input = self.tgt_embed(prev_output_tokens)
            dec_input = self.embed_positions(dec_input)

            # feed all output tokens to decoder
            tgt_mask = prev_output_tokens != self.dictionary.pad_index
            tgt_mask = tgt_mask.unsqueeze(-2)
            _future_mask = subsequent_mask(prev_output_tokens.shape[1]).to(tgt_mask)
            tgt_mask = tgt_mask & _future_mask

            dec_out = self.decoder(dec_input, memory=memory, src_mask=att_mask, tgt_mask=tgt_mask)
            if incremental_state is not None:   # evaluation
                dec_out = dec_out[:, -1:]

            # output[1] must have 'attn' key
            return self.output_proj(dec_out), {'dec_out': dec_out, 'attn': None}

    def reorder_incremental_state(self, incremental_state, new_order):
        pass


class TransformerModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser, **kwargs):
        parser.add_argument('--n-layer', type=int, default=6)
        parser.add_argument('--d-model', type=int, default=512)
        parser.add_argument('--d-feedforward', type=int, default=2048)
        parser.add_argument('--n-head', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--feat-dim', type=int, default=2048)

    @staticmethod
    def build_model(args, task, **kwargs):
        dictionary = kwargs.get('dictionary', None)
        if dictionary is None:
            dictionary = task.dictionary

        feat_dim = args.feat_dim
        n_layer, d_model, d_feedforward, n_head, dropout = \
            args.n_layer, args.d_model, args.d_feedforward, args.n_head, args.dropout
        encoder = TransformerEncoder(None, dictionary, feat_dim=feat_dim, n_layer=n_layer, d_model=d_model, d_feedforward=d_feedforward,
                                     n_head=n_head, dropout=dropout)
        decoder = TransformerDecoder(None, dictionary, n_layer=n_layer, d_model=d_model, d_feedforward=d_feedforward,
                                     n_head=n_head, dropout=dropout)

        model = TransformerModel(encoder, decoder)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)


def main():
    att_feat = torch.rand(16, 20, 2048)
    att_mask = torch.ones(16, 20)
    dictionary = Dictionary()
    for sym in ['a', 'b', 'c']: dictionary.add_symbol(sym)
    enc = TransformerEncoder(None, dictionary, n_layer=1, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1)
    dec = TransformerDecoder(None, dictionary, n_layer=1, d_model=512, d_feedforward=2048, n_head=8, dropout=0.1)

    encoder_out = enc.forward(None, None, feat_att=att_feat, att_mask=att_mask)
    prev_output_tokens = torch.randint(low=0, high=5, size=(16, 5))
    decoder_out = dec.forward(prev_output_tokens, encoder_out)

    print('encoder_out:', encoder_out[0].shape, encoder_out[1].shape)
    print('decoder_out:', decoder_out.shape)


if __name__ == '__main__':
    main()