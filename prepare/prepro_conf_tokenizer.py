# -*- coding: utf-8 -*-
# @Time : 2023/3/1 下午4:38
# @Author : Lingo
# @File : register_conf_tokenizer.py
# -*- coding: utf-8 -*-
# @Time : 2022/6/2 上午3:06
# @Author : Lingo
# @File : register.py
import os

from conica.configuration_conica import ConicaConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import json
import warnings
import argparse

from tokenizers.trainers import WordLevelTrainer
import string

from transformers import AutoTokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import Regex

import json
from tokenizers.trainers import BpeTrainer
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tokenizers.normalizers import NFC, Replace
from tokenizers import normalizers, pre_tokenizers

from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-karpathy_split_json', type=str, default="/dataset/caption/karpathy_split/dataset_coco.json",
                        help="path of karpathy split json")
    parser.add_argument('-output_file', type=str, default="../conica-base",
                        help="path of output_file")
    parser.add_argument('-min_frequency', type=int, default=5,
                        help="The minimum frequency a word should appear in corpus")
    parser.add_argument('-tokenizer_name', type=str, default=None,
                        help="the name of tokenizer, e.g, 'openai/clip-vit-large-patch14' ")
    parser.add_argument('-pad_token', type=str, default=None)
    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_align", type=int, default=128)
    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-vision_encoder_layers", type=int, default=6)
    parser.add_argument("-language_encoder_layers", type=int, default=3)
    parser.add_argument("-multimodal_decoder_layers", type=int, default=3)

    args = parser.parse_args()
    if args.tokenizer_name != None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if args.pad_token != None:
            tokenizer.add_special_tokens({"pad_token": args.pad_token})
        config = ConicaConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              n_head=args.n_head,
                              ffn_ratio=4,
                              vision_encoder_layers=args.vision_encoder_layers,
                              language_encoder_layers=args.language_encoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              bos_token_id=tokenizer.bos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              forced_eos_token_id=tokenizer.eos_token_id,
                              vocab_size=len(tokenizer))
        config.save_pretrained(args.output_file)

        tokenizer.save_pretrained(args.output_file)
    else:
        # train a BPE tokenizer from scratch on MSCOCO corpus
        imgs = json.load(open(args.karpathy_split_json, 'r'))
        imgs = imgs['images']

        gts = {}
        for img in imgs:
            if img["split"] not in ["val", "test"]:
                imgid = img["imgid"]
                for sent in img['sentences']:
                    lst = []
                    sentid = sent["sentid"]
                    if gts.get(imgid) is not None:
                        lst = gts[imgid]
                    lst.append({"image_id": imgid, "id": sentid, "caption": sent["raw"]})
                    gts[imgid] = lst
        ptbTokenizer = PTBTokenizer()
        tokens = ptbTokenizer.tokenize(gts)

        with open("temp_sentences.raw", "w") as file:
            for key in tokens:
                for sent in tokens[key]:
                    file.write(
                        sent.strip().replace("crow \\ d", "crowd").replace("\xa0", " ").replace("\\", " ") + "\n")
        file.close()

        # initialize tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
        # initialize normalizer
        tokenizer.normalizer = normalizers.Sequence([
            NFC(),
            Replace(pattern=Regex("\\s+"), content=" ")
        ])
        # initialize trainer

        trainer = BpeTrainer(min_frequency=5, end_of_word_suffix="</w>",
                             special_tokens=["[BOS]", "[EOS]", "[PAD]", "[UNK]"])

        from tokenizers.pre_tokenizers import Split, ByteLevel, Digits

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Split(pattern=Regex("'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"),
                  behavior="removed",
                  invert=True),
            Digits(individual_digits=True),
            ByteLevel(add_prefix_space=False, use_regex=True)]
        )

        tokenizer.train(["temp_sentences.raw"], trainer)
        os.remove("temp_sentences.raw")
        tokenizer.add_special_tokens(["[BOS]", "[EOS]"])

        tokenizer.decoder = BPEDecoder()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[

                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )

        config = ConicaConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              n_head=args.n_head,
                              ffn_ratio=4,
                              vision_encoder_layers=args.vision_encoder_layers,
                              language_encoder_layers=args.language_encoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              bos_token_id=tokenizer.token_to_id("[BOS]"),
                              pad_token_id=tokenizer.token_to_id("[PAD]"),
                              eos_token_id=tokenizer.token_to_id("[EOS]"),
                              forced_eos_token_id=tokenizer.token_to_id("[EOS]"),
                              vocab_size=tokenizer.get_vocab_size())

        config.save_pretrained(args.output_file)
        tokenizer.save('temp_tokenizer.json')
        from transformers import PreTrainedTokenizerFast

        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="temp_tokenizer.json")

        fast_tokenizer.add_special_tokens(
            {'pad_token': '[PAD]', 'unk_token': '[UNK]', 'bos_token': "[BOS]", "eos_token": "[EOS]"})
        fast_tokenizer.save_pretrained(args.output_file)
        os.remove("temp_tokenizer.json")
