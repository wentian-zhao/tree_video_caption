import pptree
from pptree.utils import *


def get_print_func(buf=None):
    if buf is None:
        return print
    else:
        def print_to_buf(line):
            buf.append(line)
        return print_to_buf


def print_tree(current_node, childattr='children', nameattr='name', horizontal=True, buf=None):
    if hasattr(current_node, nameattr):
        name = lambda node: getattr(node, nameattr)
    else:
        name = lambda node: str(node)

    children = lambda node: getattr(node, childattr)
    nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1

    def balanced_branches(current_node):
        size_branch = {child: nb_children(child) for child in children(current_node)}

        """ Creation of balanced lists for "a" branch and "b" branch. """
        # a = sorted(children(current_node), key=lambda node: nb_children(node))
        child = children(current_node)
        b = []

        has_child_type = True
        for c in child:
            if not hasattr(c, 'child_type'): has_child_type = False

        if not has_child_type:
            a = child
            while a and sum(size_branch[node] for node in b) < sum(size_branch[node] for node in a):
                b.append(a.pop())
        else:
            a, b = [], []
            for c in child:
                if c.child_type == 'l': a.append(c)
                elif c.child_type == 'r': b.append(c)

        return a, b

    if horizontal:
        print_tree_horizontally(current_node, balanced_branches, name, buf=buf)
    else:
        print_tree_vertically(current_node, balanced_branches, name, children, buf=buf)


def print_tree_horizontally(current_node, balanced_branches, name_getter, indent='', last='updown', buf=None):
    print = get_print_func(buf)
    up, down = balanced_branches(current_node)

    """ Printing of "up" branch. """
    for child in up:
        next_last = 'up' if up.index(child) == 0 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', ' ' * len(name_getter(current_node)))
        print_tree_horizontally(child, balanced_branches, name_getter, next_indent, next_last, buf=buf)

    """ Printing of current node. """
    if last == 'up':
        start_shape = '┌'
    elif last == 'down':
        start_shape = '└'
    elif last == 'updown':
        start_shape = ' '
    else:
        start_shape = '├'

    if up:
        end_shape = '┤'
    elif down:
        end_shape = '┐'
    else:
        end_shape = ''

    print('{0}{1}{2}{3}'.format(indent, start_shape, name_getter(current_node), end_shape))

    """ Printing of "down" branch. """
    for child in down:
        next_last = 'down' if down.index(child) is len(down) - 1 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', ' ' * len(name_getter(current_node)))
        print_tree_horizontally(child, balanced_branches, name_getter, next_indent, next_last, buf=buf)


def tree_repr(current_node, balanced_branches, name, children):
    sx, dx = balanced_branches(current_node)

    """ Creation of children representation """

    tr_rpr = lambda node: tree_repr(node, balanced_branches, name, children)

    left = branch_left(map(tr_rpr, sx)) if sx else ()
    right = branch_right(map(tr_rpr, dx)) if dx else ()

    children_repr = tuple(
        connect_branches(
            left,
            right
        ) if sx or dx else ()
    )

    current_name = name(current_node)

    name_len = len(current_name)
    name_l, name_r = name_len // 2, name_len // 2

    left_len, right_len = blocklen(left), blocklen(right)

    current_name = f"{' ' * (left_len - name_l)}{current_name}{' ' * (right_len - name_r)}"

    return multijoin([[current_name, *children_repr]]), (max(left_len, name_l), max(right_len, name_r))


def print_tree_vertically(*args, buf=None):
    print = get_print_func(buf)
    print('\n'.join(tree_repr(*args)[0]))


def to_pptree(root, child_type=None, parent_pp_node=None, vocab=None):
    w = root['w']
    if not isinstance(w, str) and vocab is not None:
        w = vocab.get_word(w)
    if child_type is not None:
        w = w + '(' + child_type + ')'

    u = pptree.Node(w, parent_pp_node)
    u.child_type = child_type

    if 'l' in root:
        for i in root['l']: to_pptree(i, 'l', u, vocab)
    if 'r' in root:
        for i in root['r']: to_pptree(i, 'r', u, vocab)
    return u

_tree = {'w': 'woman', 'd': 'ROOT', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}], 'r': [
    {'w': 'wearing', 'd': 'acl', 'p': 'VERB', 'r': [
        {'w': 'net', 'd': 'dobj', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}], 'r': [
            {'w': 'on', 'd': 'prep', 'p': 'ADP',
             'r': [{'w': 'head', 'd': 'pobj', 'p': 'NOUN', 'l': [{'w': 'her', 'd': 'poss', 'p': 'DET'}]}]}]}]},
    {'w': 'cutting', 'd': 'acl', 'p': 'VERB',
     'r': [{'w': 'cake', 'd': 'dobj', 'p': 'NOUN', 'l': [{'w': 'a', 'd': 'det', 'p': 'DET'}]}]}]}


def to_pptree_1(root, child_type=None, parent_pp_node=None, vocab=None):
    w = root['w']
    if not isinstance(w, str) and vocab is not None:
        w = vocab.get_word(w)
    if child_type is not None:
        w = w + '(' + child_type + ')' + root['p']

    u = pptree.Node(w, parent_pp_node)
    u.child_type = child_type

    if 'l' in root:
        for i in root['l']: to_pptree_1(i, 'l', u, vocab)
    if 'r' in root:
        for i in root['r']: to_pptree_1(i, 'r', u, vocab)
    return u


if __name__ == '__main__':
    pp = to_pptree_1(_tree)
    buf = []
    print_tree(pp, buf=buf, horizontal=False)
    for line in buf:
        print(line)