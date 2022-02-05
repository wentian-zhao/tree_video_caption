# BFS
import os
import sys
from collections import deque
import numpy as np
from fairseq.data import Dictionary


def tree_to_seq(tree, vocab=None, return_id=False):
    seq = []
    if 'l' in tree:
        for child in tree['l']:
            seq.extend(tree_to_seq(child, vocab, return_id))
    # w = vocab.get_word(tree['w']) if vocab is not None else tree['w']
    w = vocab[tree['w']] if vocab is not None else tree['w']
    id = tree['id'] if return_id else None
    if return_id:
        seq.append((w, id))
    else:
        seq.append(w)
    if 'r' in tree:
        for child in tree['r']:
            seq.extend(tree_to_seq(child, vocab, return_id))
    return seq


# vocab:
def get_actions_bfs_new(tree, vocab, has_start_token=False, has_end_token=False, w_type='word'):
    queue = [(tree, 0, 'r')]    # node, has_sibling, child_type
    actions = []

    if has_start_token:
        actions.append([0, 1, vocab.bos_word, 'r'])        # has_sibling, has_child, w, child_type

    i = 0
    while i < len(queue):
        node, has_sibling, child_type = queue[i]

        has_child = ('l' in node and len(node['l']) > 0) or ('r' in node and len(node['r']) > 0)
        action = [has_sibling, int(has_child), node['w'], child_type]
        actions.append(action)
        # print(i, action)

        if has_child:
            child_list = []
            if 'l' in node:
                for c in node['l']: child_list.append((c, 'l'))
            if 'r' in node:
                for c in node['r']: child_list.append((c, 'r'))
            for _i, (c, child_type) in enumerate(child_list):
                queue.append((c, int(_i < len(child_list) - 1), child_type))
        i += 1

    if has_end_token:
        actions[-1][1] = 1
        actions.append([0, 0, vocab.eos_word, 'l'])

    # actions[0] create virtual root
    for a in actions:
        w = a[2]
        if w_type == 'word':
            if isinstance(w, int):
                # w = vocab.get_word(w)
                w = vocab[w]
        elif w_type == 'id':
            if isinstance(w, str):
                # w = vocab.get_index(w)
                w = vocab.index(w)
        a[2] = w
    return actions


def get_actions_dfs_new(tree, vocab, has_start_token=False, has_end_token=False, w_type='word'):
    actions = []
    # has_sibling, has_child, word, child_type
    if has_start_token:
        actions.append([0, 1, vocab.bos_word, 'r'])

    # if has_end_token:
    #     rightmost = tree
    #     while 'r' in rightmost and len(rightmost['r']) > 0:
    #         rightmost = rightmost['r'][-1]
    #     rightmost['r'] = [{'w': vocab.eos_word}]

    def vis(root, parent_id, edge_label, has_sibling):
        has_child = ('l' in root and len(root['l']) > 0) or ('r' in root and len(root['r']) > 0)
        action = [int(has_sibling), int(has_child), root['w'], edge_label]
        actions.append(action)

        current_node_id = len(actions) - 1

        child_list, edge_type_list = [], []
        if 'l' in root:
            for child in root['l']:
                child_list.append(child)
                edge_type_list.append('l')
        if 'r' in root:
            for child in root['r']:
                child_list.append(child)
                edge_type_list.append('r')
        for i in range(len(child_list)):
            child = child_list[i]
            edge_type = edge_type_list[i]
            vis(child, current_node_id, edge_type, i < len(child_list) - 1)

    vis(tree, 0, 'l', 0)

    if has_end_token:
        actions[-1][1] = 1
        actions.append([0, 0, vocab.eos_word, 'l'])

    # actions[0] create virtual root
    for a in actions:
        w = a[2]
        if w_type == 'word':
            if isinstance(w, int):
                # w = vocab.get_word(w)
                w = vocab[w]
        elif w_type == 'id':
            if isinstance(w, str):
                # w = vocab.get_index(w)
                w = vocab.index(w)
        a[2] = w
    return actions


def get_actions_linear(words, vocab=None, has_start_token=False, has_end_token=False, w_type='word'):
    actions = []
    if has_start_token:
        actions = [[1, 0, '<s>', 'r']]
    for i, w in enumerate(words):
        actions.append([1, 0, w, 'r'])
    if has_end_token:
        actions.append([1, 0, '</s>', 'r'])

    for a in actions:
        w = a[2]
        if w_type == 'word':
            if isinstance(w, int):
                # w = vocab.get_word(w)
                w = vocab[w]
        elif w_type == 'id':
            if isinstance(w, str):
                # w = vocab.get_index(w)
                w = vocab.index(w)
        a[2] = w
    return actions


def action_to_tree_bfs(actions):
    root = {'w': 0, 'l': [], 'r': []}
    to_expand = deque()
    to_expand.append(root)
    prev_child = None
    for a in actions:
        if len(to_expand) == 0:
            break
        u = to_expand[0]
        # print(a[2], 'expand:', u['w'], 'prev_child:', prev_child['w'] if prev_child else None, 'to_expand:', [i['w'] for i in to_expand])
        has_sibling, has_child, w, child_type = a
        current_node = {'w': w, 'l': [], 'r': []}
        if has_child:
            to_expand.append(current_node)
        if child_type == 'l' or child_type == 0:
            u['l'].append(current_node)
        elif child_type == 'r' or child_type == 1:
            u['r'].append(current_node)
        if has_sibling == 0:
            to_expand.popleft()
            prev_child = None
        else:
            prev_child = current_node
    return root


def action_to_tree_dfs(actions):
    root = {'w': 0, 'l': [], 'r': []}
    to_expand = deque()     # stack
    to_expand.append(root)
    for a in actions:
        if len(to_expand) == 0:
            break
        u = to_expand[-1]

        has_sibling, has_child, w, child_type = a
        current_node = {'w': w, 'l': [], 'r': []}

        if child_type == 'l' or child_type == 0:
            u['l'].append(current_node)
        elif child_type == 'r' or child_type == 1:
            u['r'].append(current_node)

        if not has_sibling:
            to_expand.pop()

        if has_child:
            to_expand.append(current_node)
    return root


def tree_to_seq_ppl(tree, lm, vocab=None, verbose=False, return_id=False, k=2):
    def _tree_to_seq_ppl(tree, lm, vocab=None, verbose=False, return_id=False, k=2):
        child_candidates = []
        child_count = 0
        if 'l' in tree:
            child_count += len(tree['l'])
            for child in tree['l']:
                child_candidates.append(_tree_to_seq_ppl(child, lm, vocab, verbose, return_id))
        if 'r' in tree:
            child_count += len(tree['r'])
            for child in tree['r']:
                child_candidates.append(_tree_to_seq_ppl(child, lm, vocab, verbose, return_id))

        word = vocab.get_word(tree['w']) if vocab is not None else tree['w']
        id = tree['id'] if return_id else None

        if child_count == 0:
            return [[(word, id)]]

        def get_all_candidate_seq(candidates):
            seq = []
            if len(candidates) == 1:
                return candidates[0]
            for c in candidates[0]:
                if verbose: print(c)
                candidate_seq = get_all_candidate_seq(candidates[1:])
                for j in candidate_seq:
                    if verbose: print('\t', j)
                    seq.append(c + j)
            return seq

        seq = []
        if word == '<pad>':
            candidates = child_candidates + [[[(word, id)]]]
            seq.extend(get_all_candidate_seq(candidates))
        else:
            for i in range(len(child_candidates) + 1):
                candidates = child_candidates[:i] + [[[(word, id)]]] + child_candidates[i:]
                seq.extend(get_all_candidate_seq(candidates))

        ppl = np.array([lm.perplexity(' '.join([i[0] for i in s if i[0] not in ('<pad>', '<unk>', '<lc>', '<lf>')])) for s in seq])
        index = np.argsort(ppl)
        # seq = seq[index][:2]
        _seq = []
        for i in index:
            _seq.append(seq[i])
        return _seq[:k]

    seqs = _tree_to_seq_ppl(tree, lm, vocab, verbose, return_id, k)
    if return_id:
        return seqs[0]
    else:
        return [i[0] for i in seqs[0]]


def _add_extra_symbol(tree):
    _extra_sibling = {'w': '<lc>'}

    def _vis_node(tree):
        ll = len(tree.get('l', []))
        lr = len(tree.get('r', []))

        if 'l' in tree:
            for child in tree['l']: _vis_node(child)
        if 'r' in tree:
            for child in tree['r']: _vis_node(child)

        if ll + lr > 0:
            if 'r' not in tree:
                tree['r'] = []
            tree['r'].append(_extra_sibling)
    _vis_node(tree)
    return tree


def _add_sentence_end_point(tree):
    node = tree
    while ('r' in node and len(node['r']) > 0):
        node = node['r'][-1]
    if 'r' not in node:
        node['r'] = []
    node['r'].append({'w': '.'})
    return tree


def _it_tree(tree, method='dfs'):
    queue = deque([tree])
    while len(queue) > 0:
        node = queue.popleft()

        has_child = ('l' in node and len(node['l']) > 0) or ('r' in node and len(node['r']) > 0)
        if has_child:
            child_list = []
            if 'l' in node: child_list.extend(node['l'])
            if 'r' in node: child_list.extend(node['r'])

            if method == 'bfs':
                for _i, child in enumerate(child_list):
                    queue.append(child)
            elif method == 'dfs':
                for _i, child in enumerate(child_list[::-1]):   # keep the order of child
                    queue.appendleft(child)

        yield node


def get_child_rule(tree, vocab, rule_l_dict, rule_r_dict, method='dfs'):
    rule_l_label = []
    rule_r_label = []

    for node in _it_tree(tree, method):
        pos = node['p']
        left = tuple(node['p'] for node in tree['l']) if 'l' in tree else tuple()
        right = tuple(node['p'] for node in tree['r']) if 'r' in tree else tuple()
        rule_l = (pos, left)
        rule_r = (pos, right)

        _rule_l_label = rule_l_dict.get(rule_l, -1) + 1
        _rule_r_label = rule_r_dict.get(rule_r, -1) + 1
        rule_l_label.append(_rule_l_label)
        rule_r_label.append(_rule_r_label)
    return rule_l_label, rule_r_label


def get_word_gen_order_dfs(tree, vocab: Dictionary, return_word_id=False):
    _counter = {'value': 0}

    def _vis(root, counter):
        value = counter['value']
        if return_word_id:
            seq = [(vocab.index(root['w']), value)]
        else:
            seq = [(root['w'], value)]
        counter['value'] += 1

        if 'l' in root:
            _seq = []
            for c in root['l']:
                _seq.extend(_vis(c, counter))
            seq = _seq + seq
        if 'r' in root:
            _seq = []
            for c in root['r']:
                _seq.extend(_vis(c, counter))
            seq = seq + _seq
        return seq

    return _vis(tree, _counter)


if __name__ == '__main__':
    tree = {'w': 'boy', 'd': 'ROOT', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}, {'w': 'little', 'd': 'amod', 'p': 'ADJ'}], 'r': [{'w': 'playing', 'd': 'acl', 'p': 'VERB', 'r': [{'w': 'on', 'd': 'prep', 'p': 'ADP', 'r': [{'w': 'gym', 'd': 'pobj', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}, {'w': 'jungle', 'd': 'amod', 'p': 'NOUN'}]}]}]}]}
    it = _it_tree(tree, method='bfs')
    for node in it:
        print(node['w'])

    print('----')

    it = _it_tree(tree, method='dfs')
    for node in it:
        print(node['w'])

    print(get_word_gen_order_dfs(tree, None))
    print(tree_to_seq(tree))
