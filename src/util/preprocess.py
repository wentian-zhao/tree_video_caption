import os
import string
import sys
from collections import Counter, defaultdict

xrange = range

_table = str.maketrans(dict.fromkeys(string.punctuation))


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def preprocess_ngrams(image_annotations, split, vocab, eos_word='</s>', unk_word='<unk>'):
    """
    :param image_annotations: list
    [
        {'filename': '', 'split': 'train', 'imgid': 0, 'sentences': ['raw': '...', 'tokens': ['a', 'little', 'boy', ...]]}
    ]
    :param split: one of 'train', 'val', 'test', 'all'
    :param vocab: a `set` containing all words
    :return: {'document_frequency': df, 'ref_len': ref_len}
    """
    count_imgs = 0

    refs_words = []

    # for caption_item in caption_items:
    #     if split == caption_item.split or split == 'all':
    #         ref_words = []
    #         for sent in caption_item.sentences:
    #             tmp_tokens = [w if w in vocab else unk_word for w in sent.words]   # filter unknown words
    #             tmp_tokens = tmp_tokens + [eos_word]        # must add <eos> token
    #             ref_words.append(' '.join(tmp_tokens))
    #         refs_words.append(ref_words)
    #         count_imgs += 1

    for image_annotation in image_annotations:
        if split == image_annotation['split'] or split == 'all':
            ref_words = []
            for sent in image_annotation['sentences']:
                tmp_tokens = [w if w in vocab else unk_word for w in sent['tokens']]   # filter unknown words
                tmp_tokens = tmp_tokens + [eos_word]        # must add <eos> token
                ref_words.append(' '.join(tmp_tokens))
            refs_words.append(ref_words)
            count_imgs += 1

    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    return {'document_frequency': ngram_words, 'ref_len': count_imgs}