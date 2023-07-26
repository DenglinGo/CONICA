# -*- coding: utf-8 -*-
# @Time : 2023/7/2 下午4:26
# @Author : Lingo
# @File : meteor.py
# !/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading
import numpy as np
# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        path_to_jar_dirname = '/home/smart-solution-server-001/.conda/envs/caption/lib/python3.8/site-packages/pycocoevalcap/meteor/'

        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=path_to_jar_dirname, \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert (len(gts) == len(res))
        imgIds = [id for id in range(len(gts))]
        scores = []
        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0, len(gts)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        scores = np.array(score)
        return scores.mean(), scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
