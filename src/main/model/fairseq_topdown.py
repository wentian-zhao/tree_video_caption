import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model
from fairseq.utils import get_incremental_state, set_incremental_state

from fairseq.data.dictionary import Dictionary


def new_zeros(module: nn.Module, size, dtype=None, requires_grad=False):
    return next(module.parameters()).new_zeros(size, dtype=dtype, requires_grad=requires_grad)


class TopDownEncoder(FairseqEncoder):
    def __init__(self, args, dictionary: Dictionary, feat_dim, word_emb_dim, hidden_dim, att_hidden_dim, dropout_prob_lm):
        super().__init__(dictionary)
        image_emb_dim = hidden_dim
        self.fc_embed = nn.Sequential(
            nn.Linear(feat_dim, image_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob_lm)
        )
        self.att_embed = nn.Sequential(
            nn.Linear(feat_dim, image_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob_lm)
        )       # TODO: batch norm
        self.ctx2att = nn.Linear(image_emb_dim, att_hidden_dim)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        :param src_tokens: {'feat_fc': (batch_size, feat_dim)}
        :param src_lengths:
        :param kwargs:
        :return:
        """
        feat_fc, feat_att = self.fc_embed(kwargs['feat_fc']), self.fc_embed(kwargs['feat_att'])
        att_mask = kwargs['att_mask']
        feat_att_p = self.ctx2att(feat_att)
        return {'feat_fc': feat_fc, 'feat_att': feat_att,
                'feat_att_p': feat_att_p, 'att_mask': att_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        feat_fc, feat_att, feat_att_p = encoder_out['feat_fc'], encoder_out['feat_att'], encoder_out['feat_att_p']
        att_mask = encoder_out['att_mask']
        return {'feat_fc': torch.index_select(feat_fc, dim=0, index=new_order),
                'feat_att': torch.index_select(feat_att, dim=0, index=new_order),
                'feat_att_p': torch.index_select(feat_att_p, dim=0, index=new_order),
                'att_mask': torch.index_select(att_mask, dim=0, index=new_order)}


class Attention(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(Attention, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class TopDownDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, feat_dim, word_emb_dim, hidden_dim, att_hidden_dim, dropout_prob_lm):
        super().__init__(dictionary)
        self.hidden_dim = hidden_dim
        self.dropout_prob_lm = dropout_prob_lm

        self.word_emb = nn.Sequential(nn.Embedding(num_embeddings=len(self.dictionary), embedding_dim=word_emb_dim,
                                                   padding_idx=self.dictionary.pad_index),
                                      nn.ReLU(),
                                      nn.Dropout(dropout_prob_lm))

        self.att_lstm = nn.LSTMCell(hidden_dim + hidden_dim + word_emb_dim, hidden_dim)
        self.lang_lstm = nn.LSTMCell(hidden_dim + hidden_dim, hidden_dim)
        self.att = Attention(hidden_dim, att_hidden_dim)

        self.output_fc = nn.Linear(in_features=hidden_dim, out_features=len(self.dictionary))

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        feat_fc, feat_att, feat_att_p = encoder_out['feat_fc'], encoder_out['feat_att'], encoder_out['feat_att_p']
        att_mask = encoder_out['att_mask']
        batch_size, max_len = prev_output_tokens.shape

        step, rnn_state = (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'rnn_state'))
        if step is None:
            step = new_zeros(self, size=(batch_size,), dtype=torch.int64)
        if rnn_state is None:   # first step
            h_0 = new_zeros(self, (batch_size, self.hidden_dim))
            rnn_state = ((h_0, h_0), (h_0, h_0))

        prev_output_emb = self.word_emb(prev_output_tokens)
        all_logits = []
        for i in range(max_len):
            word_input = prev_output_emb[:, i]
            (h_attn_0, c_attn_0), (h_lang_0, c_lang_0) = rnn_state
            x_attn = torch.cat([h_lang_0, feat_fc, word_input], dim=1)
            h_attn_1, c_attn_1 = self.att_lstm(x_attn, (h_attn_0, c_attn_0))

            # TODO: att_mask
            att = self.att.forward(h_attn_1, feat_att, feat_att_p, att_mask)

            x_lang = torch.cat([att, h_attn_1], dim=1)
            h_lang_1, c_lang_1 = self.lang_lstm(x_lang, (h_lang_0, c_lang_0))

            output = F.dropout(h_lang_1, self.dropout_prob_lm, self.training)
            logits = self.output_fc(output)

            rnn_state = (h_attn_1, c_attn_1), (h_lang_1, c_lang_1)
            all_logits.append(logits)

        for _key, _value in [('step', step), ('rnn_state', rnn_state)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)

        all_logits = torch.stack(all_logits, dim=1)     # (batch_size, max_len, vocab_size)
        return all_logits, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super(TopDownDecoder, self).reorder_incremental_state(incremental_state, new_order)
        step, rnn_state = (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'rnn_state'))

        batch_size = len(new_order)
        if step is None:
            step = new_zeros(self, size=(batch_size,), dtype=torch.int64)
        if rnn_state is None:   # first step
            h_0 = new_zeros(self, (batch_size, self.hidden_dim))
            rnn_state = ((h_0, h_0), (h_0, h_0))

        step = torch.index_select(step, dim=0, index=new_order)
        rnn_state = [(torch.index_select(state[0], dim=0, index=new_order),
                      torch.index_select(state[1], dim=0, index=new_order)) for state in rnn_state]
        for _key, _value in [('step', step), ('rnn_state', rnn_state)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)


class TopDownModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser, **kwargs):
        parser.add_argument('--feat_dim', type=int, default=2048)
        parser.add_argument('--word_emb_dim', type=int, default=300)
        parser.add_argument('--hidden_dim', type=int, default=512)      # FIXME: change to 1000? https://github.com/ruotianluo/self-critical.pytorch/blob/master/configs/updown/updown.yml#L13
        parser.add_argument('--dropout_prob_lm', type=int, default=0.5)
        parser.add_argument('--att_hidden_dim', type=int, default=512)

    @staticmethod
    def build_model(args, task, **kwargs):
        dictionary = kwargs.get('dictionary', None)
        if dictionary is None:
            dictionary = task.dictionary

        feat_dim, word_emb_dim, hidden_dim = args.feat_dim, args.word_emb_dim, args.hidden_dim
        dropout_prob_lm, att_hidden_dim = args.dropout_prob_lm, args.att_hidden_dim
        encoder = TopDownEncoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, feat_dim=feat_dim,
                                 hidden_dim=hidden_dim,
                                 att_hidden_dim=att_hidden_dim,
                                 dropout_prob_lm=dropout_prob_lm)
        decoder = TopDownDecoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, feat_dim=feat_dim,
                                 hidden_dim=hidden_dim,
                                 att_hidden_dim=att_hidden_dim,
                                 dropout_prob_lm=dropout_prob_lm)
        model = TopDownModel(encoder, decoder)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

