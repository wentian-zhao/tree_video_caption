# old model
# predict word -> topology labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model
from fairseq.utils import get_incremental_state, set_incremental_state

from fairseq.data.dictionary import Dictionary

from main.model.merge_label import merge_cartesian, split_cartesian, batch_cartesian_prod
from main.model.tree import TreeDFS
from util import Timer


timer = Timer()
import main.model.tree as ttt
ttt.timer = timer


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
        feat_fc = self.fc_embed(kwargs['feat_fc'])
        # feat_att = self.fc_embed(kwargs['feat_att'])
        # feat_att_p = self.ctx2att(feat_att)
        # att_mask = kwargs['att_mask']

        return {'feat_fc': feat_fc,
                # 'feat_att': feat_att, 'feat_att_p': feat_att_p, 'att_mask': att_mask
                }

    def reorder_encoder_out(self, encoder_out, new_order):
        feat_fc = encoder_out['feat_fc']
        # feat_att, feat_att_p = encoder_out['feat_att'], encoder_out['feat_att_p']
        # att_mask = encoder_out['att_mask']
        return {'feat_fc': torch.index_select(feat_fc, dim=0, index=new_order),
                # 'feat_att': torch.index_select(feat_att, dim=0, index=new_order),
                # 'feat_att_p': torch.index_select(feat_att_p, dim=0, index=new_order),
                # 'att_mask': torch.index_select(att_mask, dim=0, index=new_order)
                }


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


class TopDownTreeDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, feat_dim, word_emb_dim, hidden_dim, att_hidden_dim, dropout_prob_lm,
                 max_node_count=20, gen_mode='dfs'):
        super().__init__(dictionary)
        self.hidden_dim = hidden_dim
        self.word_emb_dim = word_emb_dim
        self.dropout_prob_lm = dropout_prob_lm

        self.max_node_count = max_node_count
        self.gen_mode = gen_mode

        self.word_emb = nn.Sequential(nn.Embedding(num_embeddings=len(self.dictionary), embedding_dim=word_emb_dim,
                                                   padding_idx=self.dictionary.pad_index),
                                      nn.ReLU(),
                                      nn.Dropout(dropout_prob_lm))

        self.g_a = nn.GRUCell(input_size=self.word_emb_dim, hidden_size=self.hidden_dim)
        self.g_f = nn.GRUCell(input_size=self.word_emb_dim, hidden_size=self.hidden_dim)

        self.u_f = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.u_a = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.new_node_state = lambda x, y: F.tanh(self.u_f(x) + self.u_a(y))

        self.rnn = nn.LSTMCell(input_size=self.word_emb_dim + self.hidden_dim, hidden_size=self.hidden_dim)

        self.output_has_sibling = nn.Linear(in_features=hidden_dim, out_features=2)
        self.output_has_child = nn.Linear(in_features=hidden_dim, out_features=2)
        self.output_child_type = nn.Linear(in_features=hidden_dim, out_features=2)

        self.dropout = nn.Dropout(p=dropout_prob_lm)

        self.output_fc = nn.Linear(in_features=hidden_dim, out_features=len(self.dictionary))

        self.label_ranges = (2, 2, len(self.dictionary), 2)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        timer.clear()
        timer.tick('forward')
        timer.tick('prep')

        label_ranges = self.label_ranges

        # TODO: cartesian product
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            mode = 'inference'
        else:
            mode = 'train'

        prev_output_actions = split_cartesian(prev_output_tokens, label_ranges)
        prev_output_actions = torch.stack(prev_output_actions, dim=2)

        feat_fc = encoder_out['feat_fc']
        # feat_att, feat_att_p, att_mask = encoder_out['feat_att'], encoder_out['feat_att_p'], encoder_out['att_mask']
        batch_size, max_len = prev_output_actions.shape[:2]

        step, tree, new_node_emb = \
            (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'tree', 'new_node_emb'))
        rnn_state = get_incremental_state(self, incremental_state, 'rnn_state')
        if step is None:
            step = new_zeros(self, size=(batch_size,), dtype=torch.int64)
        if tree is None:
            tree = TreeDFS(batch_size, max_node_count=self.max_node_count + 1,
                           node_emb_dim=self.hidden_dim * 2,
                           device=feat_fc.device, mode=self.gen_mode)
        if new_node_emb is None:
            new_node_emb = torch.cat((feat_fc, feat_fc), dim=1)
        if rnn_state is None:
            x = torch.cat([torch.zeros(size=(batch_size, self.word_emb_dim), device=feat_fc.device),
                           feat_fc], dim=1)
            rnn_state = self.rnn(x, None)

        timer.tock('prep')

        all_logits = {'has_sibling': [], 'has_child': [], 'w': [], 'child_type': []}
        all_logits_cartesian = []
        for i in range(max_len):
            prev_actions = prev_output_actions[:, i]     # (batch_size, 4)

            # for _i, action in enumerate(actions):
            #     _action = action.detach().cpu().numpy()
            #     print(int(step[_i]), _action[0], _action[1], self.dictionary[_action[2]], _action[3])

            timer.tick('update')
            tree.update(prev_actions, new_node_emb)
            timer.tock('update')

            timer.tick('get_node')
            # timer.tick('get_id')
            parent_id, sibling_id = tree.get_parent_id(), tree.get_left_sibling_id()
            # timer.tock('get_id')
            # timer.tick('word_emb')
            parent_word_emb, sibling_word_emb = (self.word_emb(tree.get_node_label_by_id(_i)) for _i in
                                                 (parent_id, sibling_id))
            # timer.tock('word_emb')
            # timer.tick('node_emb')
            parent_node_emb, sibling_node_emb = (tree.get_node_emb_by_id(_i) for _i in (parent_id, sibling_id))
            # timer.tock('node_emb')
            timer.tock('get_node')

            parent_a = parent_node_emb[:, :self.hidden_dim]
            sibling_f = sibling_node_emb[:, self.hidden_dim:]

            timer.tick('inference')

            h_a = self.g_a(parent_word_emb, parent_a)
            h_f = self.g_f(sibling_word_emb, sibling_f)

            new_node_emb = torch.cat([h_a, h_f], dim=1)

            h_pred = self.new_node_state(h_a, h_f)
            h_pred = self.dropout(h_pred)  # (128, 512)

            prev_word_emb = self.word_emb(prev_actions[:, 2])
            x = torch.cat([prev_word_emb, h_pred], dim=1)
            rnn_state = self.rnn(x, rnn_state)

            h_pred = rnn_state[0]

            # TODO: conditional probability
            # condition topology labels on word
            # logit_has_sibling: (batch_size, 2, vocab_size)  w: (batch_size, vocab_size)
            logit_w = self.output_fc(h_pred)

            logit_has_sibling, logit_has_child = self.output_has_sibling(h_pred), self.output_has_child(h_pred)
            logit_child_type = self.output_child_type(h_pred)

            for key, logit in {'has_sibling': logit_has_sibling, 'has_child': logit_has_child, 'w': logit_w,
                               'child_type': logit_child_type}.items():
                all_logits[key].append(logit)

            if mode == 'inference':
                logit_cartesian = batch_cartesian_prod(logit_has_sibling, logit_has_child, logit_w, logit_child_type)
                logit_cartesian = torch.sum(logit_cartesian, dim=-1)     # (batch_size, ?)
                all_logits_cartesian.append(logit_cartesian)
            else:
                all_logits_cartesian.append(logit_w)
            step += 1

            timer.tock('inference')

        for _key, _value in [('step', step), ('tree', tree), ('new_node_emb', new_node_emb)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)
        set_incremental_state(self, incremental_state, 'rnn_state', rnn_state)     

        for key, tensor in all_logits.items():
            all_logits[key] = torch.stack(tensor, dim=1)        # (batch_size, max_len, ?)

        # all_logits_cartesian = batch_cartesian_prod(*(all_logits[key] for key in ('has_sibling', 'has_child', 'w', 'child_type')))
        # all_logits_cartesian = torch.sum(all_logits_cartesian, dim=-1)
        all_logits_cartesian = torch.stack(all_logits_cartesian, dim=1)

        timer.tock('forward')
        total = timer.times['forward']
        # print(*('{}: {:.4f} ({:.2f}%)'.format(key, time, time * 100 / total) for key, time in timer.get_time().items()))

        return all_logits_cartesian, {'attn': None, 'logits': all_logits}

    def reorder_incremental_state(self, incremental_state, new_order):
        super(TopDownTreeDecoder, self).reorder_incremental_state(incremental_state, new_order)
        step, tree, new_node_emb = \
            (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'tree', 'new_node_emb'))
        rnn_state = get_incremental_state(self, incremental_state, 'rnn_state')
        
        step = torch.index_select(step, dim=0, index=new_order)
        tree.reorder(new_order)
        new_node_emb = torch.index_select(new_node_emb, dim=0, index=new_order)

        rnn_state = [i.index_select(dim=0, index=new_order) for i in rnn_state]
        for _key, _value in [('step', step), ('tree', tree), ('new_node_emb', new_node_emb)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)    
        set_incremental_state(self, incremental_state, 'rnn_state', rnn_state)


class TopDownTreeModel(FairseqEncoderDecoderModel):
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

        # defined in pipeline
        gen_mode = args.tree_gen_mode

        encoder = TopDownEncoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, feat_dim=feat_dim,
                                 hidden_dim=hidden_dim,
                                 att_hidden_dim=att_hidden_dim,
                                 dropout_prob_lm=dropout_prob_lm)
        decoder = TopDownTreeDecoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, feat_dim=feat_dim,
                                     hidden_dim=hidden_dim,
                                     att_hidden_dim=att_hidden_dim,
                                     dropout_prob_lm=dropout_prob_lm,
                                     gen_mode=gen_mode)
        model = TopDownTreeModel(encoder, decoder)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

