# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:05
# @Author : Lingo
# @File : prepro_ngrams.py
from pyciderevalcap.ciderD.ciderD_scorer import CiderScorer
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from six.moves import cPickle
import json
import argparse
import six


def get_doc_freq(refs):
    tmp = CiderScorer(df_mode="corpus")
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)


def build_dict(imgs, params):
    count_imgs = 0

    refs_words = []
    for img in tqdm(imgs):
        if (params['split'] == img['split']) or \
                (params['split'] == 'train' and img['split'] == 'restval') or \
                (params['split'] == 'all'):
            # (params['split'] == 'val' and img['split'] == 'restval') or \
            ref_words = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<eos>']
                ref_words.append(' '.join(tmp_tokens))
            refs_words.append(ref_words)
            count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words, count_refs = get_doc_freq(refs_words)
    print('count_refs:', count_refs)
    return ngram_words, count_refs


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    return cPickle.dump(obj, f)


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))

    imgs = imgs['images']

    ngram_words, ref_len = build_dict(imgs, params)

    pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl'] + '-words.p', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/dataset/caption/karpathy_split/dataset_coco.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--output_pkl', default='/dataset/caption/karpathy_split/words/train', help='output pickle file')
    parser.add_argument('--split', default='train', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
