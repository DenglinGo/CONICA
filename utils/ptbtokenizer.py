# -*- coding: utf-8 -*-
# @Time : 2023/4/3 下午4:34
# @Author : Lingo
# @File : ptbtokenizer.py
#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import string
import sys
import subprocess
import tempfile
import itertools
# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""
    @classmethod
    def tokenize(self, captions):
        cmd = ['java', '-cp', 'stanford-corenlp-3.4.1.jar','edu.stanford.nlp.process.PTBTokenizer','-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokens = [[] for _ in captions]
        image_id = [k for k, v in enumerate(captions) for _ in range(len(v))]
        sentences = '\n'.join([c.replace('\n', ' ') for v in captions for c in v])
        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname='/home/smart-solution-server-001/.conda/envs/caption/lib/python3.8/site-packages/pycocoevalcap/tokenizer'
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE,stderr=open(os.devnull,'r'))
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split('\n')
        os.remove(tmp_file.name)
        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in enumerate(lines):
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                    if w not in PUNCTUATIONS])

            final_tokens[image_id[k]].append(tokenized_caption)

        return final_tokens

