import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model
from fairseq.utils import get_incremental_state, set_incremental_state

from fairseq.data.dictionary import Dictionary
from main.model.fairseq_transformer import TransformerEncoder, MultiHeadedAttention, \
    PositionwiseFeedForward, Embeddings, Decoder, DecoderLayer, subsequent_mask

from main.model.merge_label import merge_cartesian, split_cartesian, batch_cartesian_prod
from main.model.tree import TreeDFS
from util import Timer

_timer = Timer()
import main.model.tree as ttt

ttt.timer = _timer


def new_zeros(module: nn.Module, size, dtype=None, requires_grad=False):
    return next(module.parameters()).new_zeros(size, dtype=dtype, requires_grad=requires_grad)


# sinusoidal positional encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, *args):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# learned positional encoding
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        self.register_buffer('pe', pe)
        self.reset_parameters()

    def reset_parameters(self):
        # from https://github.com/CyberZHG/torch-position-embedding/blob/master/torch_position_embedding/position_embedding.py
        torch.nn.init.xavier_normal_(self.pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TreePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super(TreePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        self.register_buffer('div_term', div_term)

    def forward(self, word_emb, tree):
        batch_size, max_len, d_model = word_emb.shape
        rank = tree.node_rank[:, :max_len]  # (batch_size, max_len)

        tpe = torch.zeros_like(word_emb)  # (batch_size, max_len, d_model)

        div_term = self.div_term  # (d_model / 2, )

        # (batch_size, max_len, 256) * (1, 1, 256)
        tpe[:, :, 0::2] = torch.sin(rank.unsqueeze(2).expand(batch_size, max_len, d_model // 2) * div_term.view(1, 1,
                                                                                                                d_model // 2))  # (batch_size, max_len, d_model / 2)
        tpe[:, :, 1::2] = torch.cos(
            rank.unsqueeze(2).expand(batch_size, max_len, d_model // 2) * div_term.view(1, 1, d_model // 2))

        # tpe[:, :, 0::2] = torch.sin(rank).unsqueeze(2).expand(batch_size, max_len, d_model // 2) * div_term.unsqueeze(0).unsqueeze(0)    # (batch_size, max_len, d_model / 2)
        # tpe[:, :, 1::2] = torch.cos(rank).unsqueeze(2).expand(batch_size, max_len, d_model // 2) * div_term.unsqueeze(0).unsqueeze(0)

        x = word_emb
        x = x + tpe
        return self.dropout(x)


def order_mask(word_order):
    """

    :param word_order: (batch_size, max_len), starts from 0
    :return:           (batch_size, max_len, max_len)
    """
    batch_size, max_len = word_order.shape
    mask = torch.arange(0, max_len, device=word_order.device).unsqueeze(1).expand(max_len, max_len)
    mask = mask.unsqueeze(0).expand(batch_size, max_len, max_len)
    o = word_order.unsqueeze(1).expand(batch_size, max_len, max_len)
    m = mask > o
    return m


class LearnedTreePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=256, n_chunks=16):  # TODO: use smaller max_len (128)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # sinusoidal, seq
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)       # (1, max_len, d_model)
        # self.register_buffer('pe', pe)

        # # learned, seq
        # self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        # torch.nn.init.xavier_normal_(self.pe)

        # # learned, tree
        # self.child = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.LeakyReLU(),
        #     nn.Linear(d_model, d_model)
        # )
        # self.next_sibling = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.LeakyReLU(),
        #     nn.Linear(d_model, d_model)
        # )

        # fixed(sinusoidal / one-hot), tree
        self.chunk_size = d_model // n_chunks
        assert self.chunk_size * n_chunks == d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # sinusoidal
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # one-hot
        for i in range(min(max_len, d_model)):
            pe[i][i] += 1
        pe = pe.unsqueeze(0)
        pe = pe[:, :, :self.chunk_size]     # (1, max_len, chunk_size)
        self.register_buffer('pe', pe)

        # # learned, tree
        # self.chunk_size = d_model // n_chunks
        # assert self.chunk_size * n_chunks == d_model
        # self.pe = torch.nn.Parameter(torch.zeros(1, max_len, self.chunk_size))
        # torch.nn.init.xavier_normal_(self.pe)

    def get_new_pos_emb(self, tree, step):
        """
        return the positional embeddings to be added
        :param parent_id:       (batch_size,)
        :param sibling_id:      (batch_size,)
        :param parent_pos_emb:  (batch_size, d_model)
        :param sibling_pos_emb: (batch_size, d_model)
        :param step:            int, current decoding step (start from 0)
        :return:                (batch_size, d_model)
        """
        parent_id = tree.get_parent_id()
        sibling_id = tree.get_left_sibling_id()
        parent_pos_emb = tree._get_node_pos_emb_by_id_1(parent_id)  # (batch_size, d_model)
        sibling_pos_emb = tree._get_node_pos_emb_by_id_1(sibling_id)
        current_depth = tree.get_depth()
        current_sib_index = tree.get_sibling_index()  # (batch_size,)

        batch_size = parent_id.shape[0]

        # sinusoidal positional embedding
        # return self.pe[:, step, :].expand(batch_size, self.d_model)

        # learned, tree
        # return self.child(parent_pos_emb) + self.next_sibling(sibling_pos_emb)

        # sinusoidal / learned, tree
        _index = current_sib_index.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.chunk_size)
        _pe = self.pe.expand(batch_size, self.pe.shape[1], self.chunk_size)
        _p = torch.gather(_pe, dim=1, index=_index)  # (batch_size, 1, chunk_size)
        _p = _p.squeeze(1)  # (batch_size, chunk_size)
        return torch.cat([_p, parent_pos_emb[:, :-self.chunk_size]], dim=1)

    def forward(self, x, tree):
        """
        :param x: (batch_size, max_len, d_model)
        :param tree:
        :return:
        """
        max_len = x.shape[1]
        x = x + tree.node_pos_emb[:, :max_len, :]
        return self.dropout(x)


class TransformerTreeDecoder2(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, n_layer=6, d_model=512, d_feedforward=2048,
                 n_head=8, dropout=0.1, max_node_count=20, gen_mode='dfs', pe_mode='sin',
                 dec_input_order='sent'):
        super().__init__(dictionary)
        self.max_node_count = max_node_count
        self.gen_mode = gen_mode

        """
        'tree': following the order in the tree
        'sent': following normal sentence order
        """
        self.dec_input_order = dec_input_order

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_head, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_feedforward)
        self.pe_mode = pe_mode
        if pe_mode == 'sin':
            position = PositionalEncoding(d_model=d_model, dropout=dropout)
        elif pe_mode == 'tree':
            position = TreePositionalEncoding(d_model=d_model, dropout=dropout)
        elif pe_mode == 'learn':
            position = LearnedPositionalEncoding(d_model=d_model, dropout=dropout)
        elif pe_mode == 'treelearn':
            assert dec_input_order == 'tree'
            position = LearnedTreePositionalEncoding(d_model=d_model, dropout=dropout)

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

        hidden_dim = d_model
        word_emb_dim = d_model
        self.hidden_dim = d_model

        _topology_input_size = hidden_dim * 4
        if self.pe_mode == 'treelearn':
            _topology_input_size += hidden_dim
        # _sizes = (_topology_input_size, int(math.sqrt(_topology_input_size)), 2)
        #
        # topo_output = nn.Sequential(
        #     nn.Linear(in_features=_sizes[0], out_features=_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(in_features=_sizes[1], out_features=_sizes[2])
        # )
        #
        # self.output_has_sibling = copy.deepcopy(topo_output)
        # self.output_has_child = copy.deepcopy(topo_output)
        # self.output_child_type = copy.deepcopy(topo_output)

        self.topo_attn = c(attn)
        self.topo_fc_key = nn.Linear(in_features=_topology_input_size, out_features=d_model)
        topo_output = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=int(math.sqrt(d_model))),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=int(math.sqrt(d_model)), out_features=1),
        )

        self.output_has_sibling = copy.deepcopy(topo_output)
        self.output_has_child = copy.deepcopy(topo_output)
        self.output_child_type = copy.deepcopy(topo_output)

        self.label_ranges = (2, 2, len(self.dictionary), 2)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if incremental_state is not None:
            mode = 'inference'
        else:
            mode = 'train'

        """
        inference:
            for each step:
                top_label = predict_top_label(prev_dec_out)
                dec_out = decoder(prev_output_words, memory, ...)

                w = output_fc(dec_out)
                dec_out -> incremental state

        train:
            (same as inference)
        """

        topo_labels = kwargs.get('topo_labels', None)

        batch_size = prev_output_tokens.shape[0]
        device = prev_output_tokens.device

        step = get_incremental_state(self, incremental_state, 'step')
        tree = get_incremental_state(self, incremental_state, 'tree')
        new_node_emb = get_incremental_state(self, incremental_state, 'new_node_emb')
        all_actions = get_incremental_state(self, incremental_state, 'all_actions')

        memory, att_mask = encoder_out['memory'], encoder_out['att_mask']
        feat_fc = encoder_out['feat_fc']

        if step is None:
            step = 0
        if tree is None:
            tree = TreeDFS(batch_size, max_node_count=self.max_node_count + 2,
                           node_emb_dim=self.hidden_dim,
                           device=device, mode=self.gen_mode,
                           default_label=self.dictionary.pad_index,
                           use_depth=True)
        if new_node_emb is None:
            new_node_emb = feat_fc  # TODO: from encoder
            # new_node_emb = torch.zeros((batch_size, self.hidden_dim), device=device)
        if all_actions is None:
            all_actions = torch.zeros((batch_size, 0, 4), dtype=torch.int64, device=device)

        all_logits = {'has_sibling': [], 'has_child': [], 'w': [], 'child_type': []}

        def get_node_word_emb(tree, node_id):
            mask = node_id >= 0
            node_label = tree._get_node_label_by_id_1(node_id)
            word_emb = self.tgt_embed(node_label)
            return word_emb * mask.unsqueeze(-1).expand_as(word_emb)

        if mode == 'train':
            _tree = TreeDFS(batch_size, max_node_count=self.max_node_count + 2,
                            node_emb_dim=self.hidden_dim, device=prev_output_tokens.device, mode=self.gen_mode,
                            default_label=self.dictionary.pad_index, use_depth=True)

            input_actions = kwargs.get('input_actions', None)   # all the actions (batch_size, max_len, 4)

            _all_input_parent_id, _all_input_sibling_id = [], []
            _all_input_pos_emb = []
            for i in range(input_actions.shape[1]):
                if self.pe_mode == 'treelearn':
                    new_pos_emb = self.embed_positions.get_new_pos_emb(_tree, i)
                else:
                    new_pos_emb = None
                if i > 0:  # TODO: ?
                    parent_id, sibling_id = _tree.get_parent_id(), _tree.get_left_sibling_id()
                    _all_input_parent_id.append(parent_id)
                    _all_input_sibling_id.append(sibling_id)
                    _all_input_pos_emb.append(new_pos_emb)
                _tree.update(input_actions[:, i], new_node_state=None, new_pos_emb=new_pos_emb)
            _all_input_parent_id = torch.stack(_all_input_parent_id, dim=1)  # (batch_size, max_len)
            _all_input_sibling_id = torch.stack(_all_input_sibling_id, dim=1)  # (batch_size, max_len)
            if self.pe_mode == 'treelearn':
                _all_input_pos_emb = torch.stack(_all_input_pos_emb, dim=1)  # (batch_size, max_len, d_model)

            _word_emb = self.tgt_embed(input_actions[:, :, 2])  # all the words (including <s> and </s>)
            expected_output_emb = _word_emb[:, 1:]  #

            if self.dec_input_order == 'sent':
                word_and_order = kwargs.get('word_and_order', None)
                input_words = word_and_order[:, :-1, 0]  # (batch_size, max_len)
                word_order = word_and_order[:, :-1, 1]  # word_and_order[:, :, 1] starts from 0, (batch_size, max_len)
            elif self.dec_input_order == 'tree':
                input_words = input_actions[:, :-1, 2]

            input_word_emb = self.tgt_embed(input_words)

            # dec_input = self.tgt_embed(input_word_emb)
            dec_input = input_word_emb
            if self.pe_mode == 'tree':
                dec_input = self.embed_positions(dec_input, _tree)
            elif self.pe_mode in ('sin', 'learn'):
                dec_input = self.embed_positions(dec_input)
            elif self.pe_mode == 'treelearn':
                # node_pe = _tree.node_pos_emb[:, :-1, :]        # shape: (batch_size, max_len, emb_dim)
                # dec_input = self.pe_dropout(dec_input + node_pe)
                dec_input = self.embed_positions(dec_input, _tree)

            # feed all output tokens to decoder
            tgt_mask = input_words != self.dictionary.pad_index  # (batch_size, max_len)
            tgt_mask = tgt_mask.unsqueeze(-2)  # (batch_size, max_len, 1)
            if self.dec_input_order == 'sent':
                _order_mask = order_mask(word_order)  # (batch_size, max_len, max_len)
                tgt_mask = tgt_mask & _order_mask  # (batch_size, max_len, max_len)
            elif self.dec_input_order == 'tree':
                _future_mask = subsequent_mask(input_words.shape[1]).to(tgt_mask)  # (1, max_len, max_len)
                tgt_mask = tgt_mask & _future_mask  # (batch_size, max_len, max_len)

            dec_out = self.decoder(dec_input, memory=memory, src_mask=att_mask,
                                   tgt_mask=tgt_mask)  # (batch_size, max_len, d_model)

            # update the tree with node embeddings
            if self.pe_mode == 'treelearn':
                new_pos_emb = self.embed_positions.get_new_pos_emb(tree, 0)
            else:
                new_pos_emb = None
            tree.update(input_actions[:, 0], feat_fc, new_pos_emb)  # 1st node, <s>
            _all_parent_id, _all_sibling_id = [], []
            # _all_parent_id.append(tree.get_parent_id()); _all_sibling_id.append(tree.get_left_sibling_id())      # TODO : add this line or not?
            for i in range(1, input_actions.shape[1]):
                if self.pe_mode == 'treelearn':
                    new_pos_emb = self.embed_positions.get_new_pos_emb(tree, i)
                else:
                    new_pos_emb = None
                parent_id, sibling_id = tree.get_parent_id(), tree.get_left_sibling_id()
                _all_parent_id.append(parent_id)
                _all_sibling_id.append(sibling_id)
                tree.update(action_batch=input_actions[:, i], new_node_state=dec_out[:, i - 1], new_pos_emb=new_pos_emb)
            _all_parent_id = torch.stack(_all_parent_id, dim=1)  # (batch_size, max_len)
            _all_sibling_id = torch.stack(_all_sibling_id, dim=1)  # (batch_size, max_len)

            parent_node_emb, sibling_node_emb = (tree._get_node_emb_by_id_1(_i) for _i in
                                                 (_all_parent_id, _all_sibling_id))
            _topology_query = [parent_node_emb, sibling_node_emb, dec_out, expected_output_emb]

            # parent_word_emb, sibling_word_emb = (get_node_word_emb(_tree, _i) for _i in (_all_input_parent_id, _all_input_sibling_id))
            # _topology_query = [parent_word_emb, sibling_word_emb, dec_out, expected_output_emb]

            if self.pe_mode == 'treelearn':
                _topology_query.append(_all_input_pos_emb)
            topology_query = torch.cat(_topology_query, dim=-1)  # TODO: input to topology module?

            query = self.topo_fc_key(topology_query)
            # tree.node_emb[:, 1:] == dec_out
            # k = tree.node_emb[:, 1:]
            k = tree.node_emb[:, 1:] + tree.node_pos_emb[:, 1:]         # add or concat
            topology_input = self.topo_attn(query=query, key=k, value=k, mask=tgt_mask)   # TODO: correct mask / key&value?
            # topology_input = dec_out

            logit_w = self.output_proj(dec_out)
            # logit_has_sibling = self.output_has_sibling(topology_input)
            # logit_has_child = self.output_has_child(topology_input)
            # logit_child_type = self.output_child_type(topology_input)
            logit_has_sibling = self.output_has_sibling(topology_input)[:, :, 0]
            logit_has_child = self.output_has_child(topology_input)[:, :, 0]
            logit_child_type = self.output_child_type(topology_input)[:, :, 0]

            all_logits['has_sibling'] = logit_has_sibling
            all_logits['has_child'] = logit_has_child
            all_logits['child_type'] = logit_child_type
            all_logits['w'] = logit_w

        else:  # inference
            if self.pe_mode == 'treelearn':
                new_pos_emb = self.embed_positions.get_new_pos_emb(tree, step)
            else:
                new_pos_emb = None

            # infer the topological labels
            if step == 0:
                topo_labels = torch.zeros(size=(batch_size, 1, 3), dtype=torch.int64, device=prev_output_tokens.device)
                topo_labels[:, :, 1] += 1

                topo_actions = topo_labels[:, 0, :]
                action = torch.stack(
                    [topo_actions[:, 0], topo_actions[:, 1], prev_output_tokens[:, 0], topo_actions[:, 2]], dim=1)
            else:
                prev_word_emb = self.tgt_embed(prev_output_tokens[:, -1])

                parent_id, sibling_id = tree.get_parent_id(), tree.get_left_sibling_id()

                parent_node_emb, sibling_node_emb = (tree.get_node_emb_by_id(_i) for _i in (parent_id, sibling_id))
                _topology_query = [parent_node_emb, sibling_node_emb, new_node_emb, prev_word_emb]

                # parent_word_emb, sibling_word_emb = (get_node_word_emb(tree, _i) for _i in (parent_id, sibling_id))
                # _topology_input = [parent_word_emb, sibling_word_emb, new_node_emb, prev_word_emb]      # TODO: correct input?

                if self.pe_mode == 'treelearn':
                    _topology_query.append(new_pos_emb)
                topology_query = torch.cat(_topology_query, dim=1)  # TODO: input to topology module?

                query = self.topo_fc_key(topology_query)
                # TODO: mask
                # tree.node_emb[:, 1:] : (batch_size, step - 1, d_model)
                node_emb = torch.cat([tree.node_emb[:, 1:], new_node_emb.unsqueeze(1)], dim=1)
                pos_emb = torch.cat([tree.node_pos_emb[:, 1:], new_pos_emb.unsqueeze(1)], dim=1)         # TODO: when node_pos_emb == None?
                k = node_emb + pos_emb
                topology_input = self.topo_attn(query=query, key=k, value=k, mask=None)
                _x = topology_input.squeeze(1)
                # _x = new_node_emb

                logit_has_sibling, logit_has_child, logit_child_type = (module(_x)[:, 0] for module in (
                    self.output_has_sibling, self.output_has_child, self.output_child_type))

                action_has_sibling, action_has_child, action_child_type = \
                    ((torch.sigmoid(t) > 0.5).to(torch.int64) for t in (logit_has_sibling, logit_has_child, logit_child_type))

                action = torch.stack(
                    [action_has_sibling, action_has_child, prev_output_tokens[:, -1], action_child_type], dim=1)

                for key, logit in {'has_sibling': logit_has_sibling, 'has_child': logit_has_child,
                                   'child_type': logit_child_type}.items():
                    all_logits[key].append(logit)

            all_actions = torch.cat((all_actions, action.unsqueeze(1)), dim=1)
            # update the tree
            tree.update(action, new_node_state=new_node_emb, new_pos_emb=new_pos_emb)
            # TODO: predict the topological labals of the current step here?

            if self.dec_input_order == 'sent':
                input_words = tree.get_generated_words()
            elif self.dec_input_order == 'tree':
                input_words = tree.node_label[:, :tree.step]  # words

            dec_input = self.tgt_embed(input_words)

            if self.pe_mode == 'tree':
                dec_input = self.embed_positions(dec_input, tree)
            elif self.pe_mode in ('sin', 'learn'):
                dec_input = self.embed_positions(dec_input)
            elif self.pe_mode == 'treelearn':
                dec_input = self.embed_positions(dec_input, tree)

            # feed all output tokens to decoder
            tgt_mask = input_words != self.dictionary.pad_index
            tgt_mask = tgt_mask.unsqueeze(-2)
            _future_mask = subsequent_mask(input_words.shape[1]).to(tgt_mask)
            tgt_mask = tgt_mask & _future_mask

            dec_out = self.decoder(dec_input, memory=memory, src_mask=att_mask, tgt_mask=tgt_mask)

            dec_out = dec_out[:, -1:]

            set_incremental_state(self, incremental_state, 'all_actions', all_actions)
            set_incremental_state(self, incremental_state, 'new_node_emb', dec_out.squeeze(1))

        set_incremental_state(self, incremental_state, 'step', step + 1)
        set_incremental_state(self, incremental_state, 'tree', tree)
        # output[1] must have 'attn' key
        return self.output_proj(dec_out), {'dec_out': dec_out, 'attn': None, 'logits': all_logits}, all_actions

    def reorder_incremental_state(self, incremental_state, new_order):
        super(TransformerTreeDecoder2, self).reorder_incremental_state(incremental_state, new_order)
        step, tree, new_node_emb = \
            (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'tree', 'new_node_emb'))
        all_actions = get_incremental_state(self, incremental_state, 'all_actions')

        tree.reorder(new_order)
        new_node_emb = torch.index_select(new_node_emb, dim=0, index=new_order)
        all_actions = torch.index_select(all_actions, dim=0, index=new_order)

        for _key, _value in [('step', step), ('tree', tree), ('new_node_emb', new_node_emb)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)
        set_incremental_state(self, incremental_state, 'all_actions', all_actions)

    def reorder_additional_output(self, additional_output, new_order):
        all_actions = additional_output
        return torch.index_select(all_actions, dim=0, index=new_order)


class TransformerTreeModel2(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser, **kwargs):
        parser.add_argument('--n-layer', type=int, default=6)
        parser.add_argument('--d-model', type=int, default=512)
        parser.add_argument('--d-feedforward', type=int, default=2048)
        parser.add_argument('--n-head', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--feat-dim', type=int, default=2048)
        parser.add_argument('--pe-mode', type=str, default='sin', help='sin | tree | learn | treelearn')
        parser.add_argument('--dec-input-order', type=str, default='tree', choices=['tree', 'sent'],
                            help='tree -> generated order  sent -> order in the sentence')

    @staticmethod
    def build_model(args, task, **kwargs):
        dictionary = kwargs.get('dictionary', None)
        if dictionary is None:
            dictionary = task.dictionary

        n_layer, d_model, d_feedforward, n_head, dropout = \
            args.n_layer, args.d_model, args.d_feedforward, args.n_head, args.dropout
        max_node_count = args.max_sent_len + 2
        gen_mode = args.tree_gen_mode
        pe_mode = args.pe_mode

        feat_dim = args.feat_dim
        encoder = TransformerEncoder(None, dictionary, feat_dim=feat_dim, n_layer=n_layer, d_model=d_model,
                                     d_feedforward=d_feedforward,
                                     n_head=n_head, dropout=dropout)
        decoder = TransformerTreeDecoder2(None, dictionary, n_layer=n_layer, d_model=d_model,
                                          d_feedforward=d_feedforward,
                                          n_head=n_head, dropout=dropout, max_node_count=max_node_count,
                                          gen_mode=gen_mode,
                                          pe_mode=pe_mode, dec_input_order=args.dec_input_order)

        model = TransformerTreeModel2(encoder, decoder)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)
