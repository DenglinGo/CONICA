# -*- coding: utf-8 -*-
# @Time : 2022/6/2 上午12:55
# @Author : Lingo
# @File : prepro_datasets.py

from tqdm import tqdm
import pandas as pd
import string
import warnings
import argparse


def json_to_csv(jsonfile, outputfile):
    header = ["id", "filepath", "filename", "sentences", "split"]
    df = pd.DataFrame(columns=header)
    for image in tqdm(jsonfile["images"]):
        item = [image["cocoid"], image["filepath"], image["filename"],
                [" ".join(sentence["tokens"]) for sentence in image["sentences"]], image["split"]]

        df.loc[len(df) + 1] = item
    df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-karpathy_split_json', type=str, default="/dataset/caption/karpathy_split/dataset_coco.json",
                        help="path of karpathy split json")
    parser.add_argument('-output_file', type=str, default="/dataset/caption/karpathy_split/mscoco.csv",
                        help="path of output_file")
    args = parser.parse_args()
    transtab = str.maketrans({key: None for key in string.punctuation})

    json_to_csv(args.karpathy_split_json, args.output_file)
