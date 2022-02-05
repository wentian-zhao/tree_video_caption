import os
import json

from fairseq.data.dictionary import Dictionary


def load_dict(dict_file, min_word_count=5, extra_special_symbols=None):
    dict = Dictionary(extra_special_symbols=extra_special_symbols)
    with open(dict_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        strs = line.split()
        assert len(strs) == 2
        word, count = strs[0], int(strs[1])
        if count < min_word_count:
            continue
        dict.add_symbol(word, count)
    print('loaded vocab from {}, min_word_count={}, total {} words'.format(dict_file, min_word_count, len(dict)))
    return dict
