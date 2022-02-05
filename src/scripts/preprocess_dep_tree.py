import math
import multiprocessing
import os
import re
import sys
import json
from collections import defaultdict

import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import sys
sys.path.append('.')

from util.dep_tree import tree_to_seq
from util.customjson import *
from config import *


def get_subtree(token):
    subtree = {'w': token.text, 'd': token.dep_, 'p': token.pos_}
    lefts, rights = list(token.lefts), list(token.rights)
    if len(lefts) > 0:
        subtree['l'] = []
        for c in lefts:
            subtree['l'].append(get_subtree(c))
    if len(rights) > 0:
        subtree['r'] = []
        for c in rights:
            subtree['r'].append(get_subtree(c))
    return subtree


def get_custom_tokenizer(nlp):
    def custom_tokenizer(sent):
        tokens = sent.split()
        return Doc(nlp.vocab, tokens)
    return custom_tokenizer


def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


def process_sents(arg):
    sent_ids, sents = arg
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = get_custom_tokenizer(nlp)
    nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sdb', before='parser')
    docs = nlp.pipe(sents)
    data = {}
    for i, doc in tqdm(enumerate(docs), total=len(sents)):
        tokens = list(doc)
        for token in tokens:
            if token.dep_ == 'ROOT':
                root = token
                break
        tree = get_subtree(root)
        data[sent_ids[i]] = tree
    return data


def preprocess_dep_tree(dataset_name):
    with open(os.path.join(data_path, dataset_name, 'annotation_{}.json'.format(dataset_name)), 'r') as f:
        annotation = load_custom(f)

    sent_ids = []
    sents = []
    sents_dict = {}
    for img in annotation['images']:
        for sent in img['sentences']:
            sent_id = sent['sentid']
            sents_dict[sent_id] = sent
            sent = ' '.join(sent['tokens'])
            sent_ids.append(sent_id)
            sents.append(sent)

    n_threads = 4
    chunk_size = math.ceil(len(sents) / n_threads)
    print('chunk size:', chunk_size)

    args = []
    for i in range(n_threads):
        _sent_ids = sent_ids[i * chunk_size : min((i + 1) * chunk_size, len(sent_ids))]
        _sents = sents[i * chunk_size: min((i + 1) * chunk_size, len(sent_ids))]
        args.append((_sent_ids, _sents))

    # docs = nlp.pipe(sents)
    # data = {}
    # for i, doc in tqdm(enumerate(docs), total=len(sent_ids)):
    #     tokens = list(doc)
    #     for token in tokens:
    #         if token.dep_ == 'ROOT':
    #             root = token
    #             break
    #     tree = get_subtree(root)
    #     data[sent_ids[i]] = tree

    pool = multiprocessing.Pool(4)
    results = pool.map(process_sents, args)
    print('merge...')
    data = {}
    for i in results:
        data.update(i)

    count = 0
    for sent_id, dep_tree in data.items():
        if tree_to_seq(dep_tree) != sents_dict[sent_id]['tokens']:
            print('id:', sent_id)
            print('original:', sents_dict[sent_id]['tokens'])
            print('restored:', tree_to_seq(dep_tree))
            count += 1
    print(count, 'out of', len(list(data.items())))

    print('save results...')
    with open(os.path.join(data_path, dataset_name, 'dep_tree_{}_spacy.json'.format(dataset_name)), 'w') as f:
        json.dump(data, f)


def process_sents_split(arg):
    sent_ids, sents = arg
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = get_custom_tokenizer(nlp)

    docs = nlp.pipe(sents)

    split_sents = []
    split_sent_ids = []
    for i, doc in tqdm(enumerate(docs), total=len(sents)):
        for _s in doc.sents:
            s = str(_s).replace('.', '').lower()
            if len(s.strip()) == 0: continue
            split_sents.append(s)
            split_sent_ids.append(sent_ids[i])

    nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sdb', before='parser')
    docs = nlp.pipe(split_sents, disable=['ner'])

    data = defaultdict(list)
    for i, doc in tqdm(enumerate(docs), total=len(split_sents)):
        sent_id = split_sent_ids[i]
        tokens = list(doc)
        for token in tokens:
            if token.dep_ == 'ROOT':
                data[sent_id].append(get_subtree(token))
    # _data = {}
    # for sent_id, sub_tree_list in data.items():
    #     tree = {'w': '<s>', 'r': sub_tree_list}
    #     _data[sent_id] = tree

    return data


def preprocess_dep_tree_split(dataset_name):
    with open(os.path.join(data_path, dataset_name, 'annotation_{}.json'.format(dataset_name)), 'r') as f:
        annotation = load_custom(f)

    sent_ids = []
    sents = []
    sents_dict = {}
    for img in annotation['images']:
        for sent in img['sentences']:
            sent_id = sent['sentid']
            sents_dict[sent_id] = sent
            sent = sent['raw']
            sent_ids.append(sent_id)
            sents.append(sent)

    n_threads = 4
    chunk_size = math.ceil(len(sents) / n_threads)
    print('chunk size:', chunk_size)

    args = []
    for i in range(n_threads):
        _sent_ids = sent_ids[i * chunk_size : min((i + 1) * chunk_size, len(sent_ids))]
        _sents = sents[i * chunk_size: min((i + 1) * chunk_size, len(sent_ids))]
        args.append((_sent_ids, _sents))

    data = {}

    # for _args in args:
    #     data.update(process_sents_split(_args))

    pool = multiprocessing.Pool(4)
    results = pool.map(process_sents_split, args)
    print('merge...')
    for i in results:
        data.update(i)

    # count = 0
    # for sent_id, dep_tree in data.items():
    #     seq = []
    #     for sub_tree in dep_tree:
    #         seq.append(tree_to_seq(sub_tree))
    #     if seq != sents_dict[sent_id]['tokens']:
    #         print('id:', sent_id)
    #         print('original:', sents_dict[sent_id]['tokens'])
    #         print('restored:', tree_to_seq(dep_tree))
    #         count += 1
    # print(count, 'out of', len(list(data.items())))

    print('save results...')
    with open(os.path.join(data_path, dataset_name, 'dep_tree_{}_spacy.json'.format(dataset_name)), 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    # preprocess_dep_tree('coco')
    # preprocess_dep_tree('flickr30k')
    # preprocess_dep_tree('msvd')
    # preprocess_dep_tree('msrvtt')
    # preprocess_dep_tree_split('charades')
    preprocess_dep_tree('activitynet')