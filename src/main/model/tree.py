from collections import deque

import torch
import numpy as np

from util import Timer
from util.dep_tree import get_actions_dfs_new, action_to_tree_dfs, tree_to_seq, action_to_tree_bfs
from util.vis_tree import print_tree_horizontally, print_tree, to_pptree
from util.vocab import load_dict

timer = Timer()


class BatchDeque:
    def __init__(self, batch_size, capacity, dtype, device):
        self.batch_size = batch_size
        self.a = torch.zeros((batch_size, capacity + 1), dtype=dtype, device=device)
        self.head = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        self.tail = torch.zeros((batch_size,), dtype=torch.int64, device=device)

        self._vector = torch.zeros((batch_size,), dtype=dtype, device=device)

    def is_empty(self):
        return self.head == self.tail

    def popleft(self, mask=None):
        """

        :param mask: (batch_size,)
        :return:
        """
        if mask is not None:
            self.head += mask
        else:
            self.head += 1
        assert (self.head <= self.tail).all()

    def pop(self, mask=None):
        timer.tick('_pop')
        if mask is not None:
            # self.tail -= mask.to(self.tail)
            self.tail += -1 * mask
        else:
            self.tail -= 1
        # assert (self.tail >= 0).all()
        timer.tock('_pop')

    def append(self, value, mask=None):
        """

        :param value: (batch_size,)
        :param mask: (batch_size,)
        :return:
        """
        timer.tick('_append')
        if not isinstance(value, torch.Tensor):     # value is identical for all deque
            # _vec = self.a.new_zeros(size=(self.batch_size)).fill(value)
            self._vector.fill_(value)
            value = self._vector

        src = value.unsqueeze(1)
        timer.tick('_scatter')
        self.a.scatter_(dim=1, index=self.tail.unsqueeze(1), src=src)
        timer.tock('_scatter')
        if mask is not None:
            self.tail += mask
        else:
            self.tail += 1
        # assert (self.tail < self.a.shape[1]).all()
        timer.tock('_append')

    def get_head(self, default=-1):
        value = torch.gather(self.a, dim=1, index=self.head.unsqueeze(1)).squeeze(1)
        value.masked_fill_(mask=self.is_empty(), value=default)
        return value

    def get_tail(self, default=-1):
        value = torch.gather(
            self.a, dim=1,
            index=(torch.max((self.tail - 1), torch.zeros_like(self.tail))).unsqueeze(1)
        ).squeeze(1)
        value.masked_fill_(mask=self.is_empty(), value=default)
        return value

    def reorder(self, new_order):
        self.batch_size = new_order.shape[0]
        for key in 'a', 'head', 'tail':
            tensor = getattr(self, key)
            setattr(self, key, tensor.index_select(index=new_order, dim=0))


class Tree:
    def __init__(self, batch_size, max_node_count, device):
        self.batch_size, self.max_node_count = batch_size, max_node_count
        self.device = device

        self.node_count = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        self.node_label = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)

        self.edge_index = torch.zeros((batch_size, max_node_count, 2), dtype=torch.int64, device=device)
        self.edge_count = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        self.edge_label = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)

        self.actions = torch.zeros((batch_size, max_node_count, 4), dtype=torch.int64, device=device)
        self.num_updates = 0

    # correct
    def _add_node(self, new_node_label, mask=None):
        """
        add node
        :param new_node_label: (batch_size,)
        :param mask: (batch_size,)
        :return:
        """
        v = self.node_count
        # self.node_label.scatter_(dim=1, index=v.view(self.batch_size, 1),
        #                          src=new_node_label.view(self.batch_size, 1))
        self.node_label = self.node_label.scatter(dim=1, index=v.view(self.batch_size, 1),
                                 src=new_node_label.view(self.batch_size, 1))
        if mask is not None:
            self.node_count += mask
        else:
            self.node_count += 1

    # # correct
    def _add_edge(self, u, edge_label, mask=None):
        """
        add edge from `u` to the last generated nodes
        :param u: (batch_size,)
        :param edge_label: (batch_size,)
        :param mask: (batch_size,)
        :return:
        """
        v = self.node_count - 1
        new_edge_index = torch.stack((u, v), dim=1)
        self.edge_index.scatter_(dim=1,
                                 index=self.edge_count.view(self.batch_size, 1, 1).expand(self.batch_size, 1, 2),
                                 src=new_edge_index.view(self.batch_size, 1, 2))
        self.edge_label.scatter_(dim=1,
                                 index=self.edge_count.view(self.batch_size, 1),
                                 src=edge_label.view(self.batch_size, 1))

        if mask is not None:
            self.edge_count += mask.to(self.edge_count.dtype)
        else:
            self.edge_count += 1

    def update(self, action_batch, new_node_state=None):
        index = self.num_updates
        self.actions[:, index, :] = action_batch
        self.num_updates += 1


class TreeDFS(Tree):
    def __init__(self, batch_size, max_node_count, node_emb_dim, device, mode, use_depth=False, default_label=0):
        super(TreeDFS, self).__init__(batch_size, max_node_count, device)
        self.mode = mode
        self.node_emb_dim = node_emb_dim
        assert self.mode in ('bfs', 'dfs')

        # fill node_label with default_label
        # self.node_label.fill_(default_label)
        self.node_label.fill_ = default_label

        self.parent_of = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)

        self.use_depth = use_depth
        if use_depth:
            self.node_depth = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)
            self.node_rank = torch.zeros((batch_size, max_node_count), dtype=torch.float, device=device)        # sort to get order of words
            self.child_count = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)
            self.sib_index = torch.zeros((batch_size, max_node_count), dtype=torch.int64, device=device)        # the index among the sibling nodes

        self.step = 0
        self.node_emb = torch.zeros((batch_size, 0, node_emb_dim), device=device)         # (batch_size, max_length, node_emb_dim)
        self.to_expand = BatchDeque(batch_size, max_node_count, dtype=torch.int64, device=device)
        self.prev_child_id = BatchDeque(batch_size, max_node_count, dtype=torch.int64, device=device)
        self.prev_child_id.append(-1)

        self.node_pos_emb = torch.zeros((batch_size, 0, node_emb_dim), device=device)

    def new_zeros(self, size, dtype):
        return torch.zeros(size=size, dtype=dtype, device=self.device)

    # correct
    def _add_node(self, new_node_label, mask=None):
        """
        add node
        :param new_node_label: (batch_size,)
        :param mask: (batch_size,)
        :return:
        """
        v = self.node_count
        # self.node_label.scatter_(dim=1, index=v.view(self.batch_size, 1),
        #                          src=new_node_label.view(self.batch_size, 1))
        self.node_label = self.node_label.scatter(dim=1, index=v.view(self.batch_size, 1),
                                 src=new_node_label.view(self.batch_size, 1))

        if mask is not None:
            self.node_count += mask
        else:
            self.node_count += 1

    def _add_edge(self, u, edge_label, mask=None):
        """
        add edge from `u` to the last generated nodes
        :param u: (batch_size,)
        :param edge_label: (batch_size,) \in {0, 1}   0 -> l 1 -> r
        :param mask: (batch_size,)
        :return:
        """
        v = self.node_count - 1
        new_edge_index = torch.stack((u, v), dim=1)
        self.edge_index.scatter_(dim=1,
                                 index=self.edge_count.view(self.batch_size, 1, 1).expand(self.batch_size, 1, 2),
                                 src=new_edge_index.view(self.batch_size, 1, 2))
        self.edge_label.scatter_(dim=1,
                                 index=self.edge_count.view(self.batch_size, 1),
                                 src=edge_label.view(self.batch_size, 1))

        if mask is not None:
            self.edge_count += mask.to(self.edge_count.dtype)
        else:
            self.edge_count += 1

        # parent_of[v] = u
        _v = v.view(self.batch_size, 1)
        self.parent_of.scatter_(dim=1, index=_v, src=u.view(self.batch_size, 1))

        if self.use_depth:
            _u = torch.masked_fill(u, ~mask, 0).view(self.batch_size, 1)
            u_depth = torch.gather(self.node_depth, index=_u, dim=1)
            v_depth = u_depth + 1
            self.node_depth.scatter_(dim=1, index=v.view(self.batch_size, 1), src=v_depth)

            # update sib_index
            _sib_index = torch.gather(self.child_count, index=_u, dim=1)
            self.sib_index.scatter_(dim=1, index=v.view(self.batch_size, 1), src=_sib_index)

            self.child_count.scatter_add_(dim=1, index=_u, src=torch.ones_like(v_depth))
            u_child_count = torch.gather(self.child_count, index=_u, dim=1)
            u_rank = torch.gather(self.node_rank, index=_u, dim=1)
            _a = torch.ones(u_depth.shape, device=u_depth.device) * 4      # the base (4) should be larger than maximum child num
            rank_delta = torch.pow(_a, -u_depth.to(torch.float))
            rank_direction = edge_label.masked_fill(edge_label == 0, -1)
            v_rank = u_rank + rank_delta * rank_direction.to(rank_delta) * u_child_count.to(rank_delta)
            self.node_rank.scatter_(dim=1, index=v.view(self.batch_size, 1), src=v_rank)

    def get_ordered_label(self):
        _, node_index = torch.sort(self.node_rank)
        ordered_label = torch.gather(self.node_label, dim=1, index=node_index)
        return ordered_label

    def update(self, action_batch, new_node_state=None, new_pos_emb=None):
        """

        :param action_batch: (batch_size, 4)
        :param new_node_state: (batch_size, node_state_dim)
        :return:
        """
        super().update(action_batch, new_node_state)

        has_sibling, has_child, w, child_type = (action_batch[:, i] for i in range(4))

        if self.step == 0:
            self.to_expand.append(self.new_zeros(size=(self.batch_size,), dtype=torch.int64))
            self._add_node(w)
            if new_node_state is not None:
                self.node_emb = torch.cat((self.node_emb, new_node_state.unsqueeze(1)), dim=1)
            if new_pos_emb is not None:
                self.node_pos_emb = torch.cat((self.node_pos_emb, new_pos_emb.unsqueeze(1)), dim=1)
        else:
            timer.tick('_update_1')

            has_new_node = (self.to_expand.is_empty() == 0)
            self._add_node(w, has_new_node)

            if self.mode == 'dfs':
                to_expand = self.to_expand.get_tail()           # DFS
            else:
                to_expand = self.to_expand.get_head()           # BFS
            self._add_edge(u=to_expand, mask=has_new_node, edge_label=child_type)

            if new_node_state is not None:
                self.node_emb = torch.cat((self.node_emb, new_node_state.unsqueeze(1)), dim=1)
            if new_pos_emb is not None:
                self.node_pos_emb = torch.cat((self.node_pos_emb, new_pos_emb.unsqueeze(1)), dim=1)

            timer.tock('_update_1')
            timer.tick('_update_2')

            if self.mode == 'dfs':
                self.prev_child_id.pop(mask=has_new_node)
                self.prev_child_id.append(self.step, mask=has_sibling & has_new_node)
                self.prev_child_id.append(-1, mask=has_child & has_new_node)

                self.to_expand.pop(mask=(~has_sibling) & has_new_node)
                self.to_expand.append(self.step, mask=has_child & has_new_node)
            else:
                self.to_expand.append(self.step, mask=has_child & has_new_node)
                self.to_expand.popleft(mask=(~has_sibling) & has_new_node)

                self.prev_child_id.pop(mask=has_new_node)
                self.prev_child_id.append(-1, mask=(~has_sibling) & has_new_node)
                self.prev_child_id.append(self.step, mask=has_sibling & has_new_node)
            timer.tock('_update_2')

        self.step += 1

    def get_parent_id(self):        # parent id of the current node
        if self.mode == 'dfs':
            return self.to_expand.get_tail(default=-1)
        else:
            # original
            # TODO: why default=0?
            # return self.to_expand.get_head(default=0)
            return self.to_expand.get_head(default=-1)

    def get_left_sibling_id(self):  # last sibling id of the current node
        if self.mode == 'dfs':
            return self.prev_child_id.get_tail(default=-1)
        else:
            return self.prev_child_id.get_tail(default=-1)

    def get_depth(self):        # the depth of the current node, start from 1
        parent_id = self.get_parent_id()            # (batch_size,)
        mask = (parent_id >= 0).unsqueeze(1).to(self.device)        # (batch_size, 1)
        index = torch.max(parent_id, torch.zeros_like(parent_id)).unsqueeze(1)  # (batch_size, 1)
        parent_depth = torch.gather(self.node_depth, index=index, dim=1)        # (batch_size, 1)
        current_depth = (parent_depth + 1) * mask
        return current_depth.squeeze(1)     # (batch_size,)

    def get_sibling_index(self):    # the sibling index of the current word, start from 0
        left_sibling_id = self.get_left_sibling_id()
        mask = (left_sibling_id >= 0).unsqueeze(1).to(self.device)
        index = torch.max(left_sibling_id, torch.zeros_like(left_sibling_id)).unsqueeze(1)
        left_sibling_index = torch.gather(self.sib_index, index=index, dim=1)
        sibling_index = (left_sibling_index + 1) * mask
        return sibling_index.squeeze(1)     # (batch_size,)

    def get_node_emb_by_id(self, node_id):
        """

        :param node_id: (batch_size,)
        :return:
        """
        mask = (node_id >= 0).unsqueeze(1).unsqueeze(2).to(self.device)
        index = torch.max(node_id, torch.zeros_like(node_id))               # (batch_size,)
        index = index.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, self.node_emb_dim)

        emb = torch.gather(self.node_emb, index=index, dim=1)
        emb = emb * mask.expand_as(emb)
        return emb.squeeze(1)

    def _get_node_emb_by_id_1(self, node_id):
        """
        :param node_id: (batch_size, *)
        :return:
        """
        _shape = node_id.shape[1:]
        node_id = node_id.view(self.batch_size, -1)     # (batch_size, ?)

        mask = (node_id >= 0).unsqueeze(2).to(self.device)
        index = torch.max(node_id, torch.zeros_like(node_id))  # (batch_size,)
        index = index.unsqueeze(2).expand(self.batch_size, node_id.shape[1], self.node_emb_dim)

        emb = torch.gather(self.node_emb, index=index, dim=1)
        emb = emb * mask.expand_as(emb)
        emb = emb.reshape(self.batch_size, *_shape, self.node_emb_dim)
        return emb

    def _get_node_pos_emb_by_id_1(self, node_id):
        """
        :param node_id: (batch_size, *)
        :return: (batch_size, *, )
        """
        _shape = node_id.shape[1:]
        node_id = node_id.view(self.batch_size, -1)     # (batch_size, ?)

        mask = (node_id >= 0).unsqueeze(2).to(self.device)
        if not mask.any():
            return self.new_zeros((self.batch_size, *_shape, self.node_emb_dim), dtype=self.node_pos_emb.dtype)
        index = torch.max(node_id, torch.zeros_like(node_id))  # (batch_size,)
        index = index.unsqueeze(2).expand(self.batch_size, node_id.shape[1], self.node_emb_dim)

        emb = torch.gather(self.node_pos_emb, index=index, dim=1)
        emb = emb * mask.expand_as(emb)
        emb = emb.reshape(self.batch_size, *_shape, self.node_emb_dim)
        return emb

    def get_node_label_by_id(self, node_id):
        """

        :param node_id: (batch_size,)
        :return:
        """
        mask = (node_id >= 0).unsqueeze(1).to(self.device)
        index = torch.max(node_id, torch.zeros_like(node_id))
        index = index.unsqueeze(1)

        label = torch.gather(self.node_label, index=index, dim=1)
        label = label * mask.expand_as(label)
        return label.squeeze(1)

    def _get_node_label_by_id_1(self, node_id):
        """

        :param node_id: (batch_size, *)
        :return:
        """
        _shape = node_id.shape[1:]
        node_id = node_id.view(self.batch_size, -1)  # (batch_size, ?)

        mask = (node_id >= 0).to(self.device)
        index = torch.max(node_id, torch.zeros_like(node_id))
        index = index

        label = torch.gather(self.node_label, index=index, dim=1)
        label = label * mask.expand_as(label)
        label = label.reshape(self.batch_size, *_shape)
        return label

    def reorder(self, new_order):
        for key in 'node_count', 'node_label', 'edge_index', 'edge_count', 'edge_label', 'node_emb':
            tensor = getattr(self, key)
            setattr(self, key, tensor.index_select(index=new_order, dim=0))
        self.batch_size = new_order.shape[0]
        self.to_expand.reorder(new_order)
        self.prev_child_id.reorder(new_order)

    def get_generated_words(self):
        max_node_count = self.num_updates
        all_word_ids = np.zeros((self.batch_size, max_node_count), dtype=np.int)
        for i in range(self.batch_size):
            actions = self.actions[i][:max_node_count].cpu().numpy()
            if self.mode == 'dfs':
                t = action_to_tree_dfs(actions)
            elif self.mode == 'bfs':
                t = action_to_tree_bfs(actions)
            seq = tree_to_seq(t)
            # TODO: why 0 in the first?
            word_ids = seq[:-1]        # remove eos
            l = len(word_ids)
            all_word_ids[i, :l] = word_ids
        return torch.tensor(all_word_ids, device=self.device)


if __name__ == '__main__':
    vocab = load_dict('/media/wentian/sdb1/work/graph_cap_2/data/dictionary_coco.txt')

    t = {'w': 'woman', 'd': 'ROOT', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}], 'r': [
    {'w': 'wearing', 'd': 'acl', 'p': 'VERB', 'r': [
        {'w': 'net', 'd': 'dobj', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}], 'r': [
            {'w': 'on', 'd': 'prep', 'p': 'ADP',
             'r': [{'w': 'head', 'd': 'pobj', 'p': 'NOUN', 'l': [{'w': 'her', 'd': 'poss', 'p': 'DET'}]}]}]}]},
    {'w': 'cutting', 'd': 'acl', 'p': 'VERB',
     'r': [{'w': 'cake', 'd': 'dobj', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}]}]}]}

    actions = get_actions_dfs_new(t, vocab, has_start_token=True, w_type='id')
    for a in actions:
        a[3] = {'l': 0, 'r': 1}[a[3]]
    actions = torch.LongTensor(actions)

    tree = TreeDFS(1, 15, 1, torch.device('cpu'), mode='dfs', use_depth=True)

    for i in range(len(actions)):
        word = vocab.string(actions[i][2].unsqueeze(0))
        print('current word: \"{:>10}\"'.format(word),
              'left_sibling:', tree.get_left_sibling_id(),
              'parent:', tree.get_parent_id())
        print('current depth:', tree.get_depth(), 'sibling index:', tree.get_sibling_index())
        tree.update(actions[i].unsqueeze(0), torch.zeros(1, 1))
        words = tree.get_generated_words()
        _words = [[vocab.string([i]) for i in j] for j in words]
        print('after update words:', _words)


    seq = tree.get_ordered_label()
    print(seq)

    p = to_pptree(t)
    print_tree(p, horizontal=False)

    print('nodes:\n', *('{:>8}'.format(vocab[i]) for i in tree.node_label[0][1:]))
    print('depth:\n', *('{:>8}'.format(i) for i in tree.node_depth[0][1:]))
    print('rank:\n', *('{:>8}'.format(i) for i in tree.node_rank[0][1:]))
    print('sib_index:\n', *('{:>8}'.format(i) for i in tree.sib_index[0][1:]))

    sent = vocab.string(seq[0])
    print(sent)

    print(tree)


