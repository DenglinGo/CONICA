# -*- coding: utf-8 -*-
# @Time : 2023/7/3 下午5:04
# @Author : Lingo
# @File : gender.py
import re

import pandas as pd
import json
import numpy as np
from pattern.text import singularize

female_word_list = ["woman", "girl", "female", "lady", "groom", "mother", "sister", "wife", "daughter"]
male_word_list = ["man", "boy", "male", "cowboy", "bride", "father", "gentleman", "brother", "husband", "son"]
neutral_word_list = ["people", "person", "human", "baby"]


def gender_for_caption(caption):
    is_female = False
    is_male = False
    is_neutral = False
    caption = " ".join([singularize(word) for word in caption.split(" ")])
    for word in female_word_list:
        if re.findall("\W" + word + "\W", caption):
            is_female = True
            break
    for word in male_word_list:
        if re.findall("\W" + word + "\W", caption):
            is_male = True
            break

    for word in neutral_word_list:
        if re.findall("\W" + word + "\W", caption):
            is_neutral = True
            break
    if is_female and not is_male:
        return 0
    elif not is_female and is_male:
        return 1
    elif is_female and is_male:
        return 2
    elif not is_female and not is_male and is_neutral:
        return 2
    else:
        return 3


dataset = pd.read_csv("/dataset/caption/mscoco.csv")

test_set = dataset[dataset.split.isin(["test"])]
results_list = ["result_xe_bs.json",
                "result_xe_scd.json",
                "result_rl_bs.json",
                "result_rl_scd.json"]
for results in results_list:

    print(results)
    results = json.load(open(results, "r"))
    male = np.zeros(3)
    female = np.zeros(3)

    for result in results:

        image_id = result["image_id"]
        captions = test_set[test_set.cocoid == image_id].sentences.to_list()
        captions = eval(captions[0])
        gender_predict = gender_for_caption(result["caption"])
        gender_targets = np.array([gender_for_caption(caption) for caption in captions])
        values, counts = np.unique(gender_targets, return_counts=True)
        gender_target = values[counts.argmax()]
        if gender_target == 0:
            if gender_predict == 0:
                female[0] += 1
            elif gender_predict == 1:
                female[1] += 1
            else:
                female[2] += 1
        elif gender_target == 1:
            if gender_predict == 1:
                male[0] += 1
            elif gender_predict == 0:
                male[1] += 1
            else:
                male[2] += 1
    print(male)
    print(female)
    print(male / male.sum())
    print(female / female.sum())
    print((male+female)/(male.sum()+female.sum()))
