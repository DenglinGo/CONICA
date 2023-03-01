# -*- coding: utf-8 -*-
# @Time : 2023/3/1 下午4:38
# @Author : Lingo
# @File : register_conf_tokenizer.py
# -*- coding: utf-8 -*-
# @Time : 2022/6/2 上午3:06
# @Author : Lingo
# @File : register.py
import os

from conica.configuration_conica import CONICAConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import json
import warnings
import argparse

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
from tokenizers.trainers import WordLevelTrainer
import string

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-karpathy_split_json', type=str, default="/dataset/caption/karpathy_split/dataset_coco.json",
                        help="path of karpathy split json")
    parser.add_argument('-output_file', type=str, default="../config",
                        help="path of output_file")
    parser.add_argument('-min_frequency', type=int, default=5,
                        help="The minimum frequency a word should appear in corpus")
    parser.add_argument('-tokenizer_name', type=str, default=None,
                        help="the name of tokenizer, e.g, 'openai/clip-vit-large-patch14' ")
    parser.add_argument('-pad_token', type=str, default=None)
    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_align", type=int, default=128)
    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-encoder_layers", type=int, default=6)
    parser.add_argument("-unimodal_decoder_layers", type=int, default=3)
    parser.add_argument("-multimodal_decoder_layers", type=int, default=3)

    args = parser.parse_args()
    if args.tokenizer_name != None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if args.pad_token != None:
            tokenizer.add_special_tokens({"pad_token": args.pad_token})
        config = CONICAConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              decoder_attention_heads=args.n_head,
                              encoder_attention_heads=args.n_head,
                              encoder_layers=args.encoder_layers,
                              unimodal_decoder_layers=args.unimodal_decoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              decoder_ffn_dim=args.d_model * 4,
                              encoder_ffn_dim=args.d_model * 4,
                              bos_token_id=tokenizer.bos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              forced_eos_token_id=tokenizer.eos_token_id,
                              vocab_size=len(tokenizer))
        config.save_pretrained(args.output_file)

        tokenizer.save_pretrained(args.output_file)
    else:
        # train a word level tokenizer from scratch on MSCOCO corpus
        imgs = json.load(open(args.karpathy_split_json, 'r'))
        imgs = imgs['images']

        transtab = str.maketrans({key: None for key in string.punctuation})

        trainer = WordLevelTrainer(min_frequency=args.min_frequency, special_tokens=["[PAD]", "[UNK]"])
        with open("temp_sentences.raw", "w") as file:
            for img in imgs:
                if img["split"] != "test":
                    for sent in img['sentences']:
                        s = ' '.join(sent["tokens"]).translate(transtab).strip()
                        file.write(s + "\n")
        file.close()
        tokenizer.train(["temp_sentences.raw"], trainer)
        os.remove("temp_sentences.raw")

        tokenizer.add_special_tokens(["[BOS]", "[EOS]"])

        from tokenizers.processors import TemplateProcessing

        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )

        config = CONICAConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              decoder_attention_heads=args.n_head,
                              encoder_attention_heads=args.n_head,
                              encoder_layers=args.encoder_layers,
                              unimodal_decoder_layers=args.unimodal_decoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              decoder_ffn_dim=args.d_model * 4,
                              encoder_ffn_dim=args.d_model * 4,
                              bos_token_id=tokenizer.token_to_id("[BOS]"),
                              pad_token_id=tokenizer.token_to_id("[PAD]"),
                              eos_token_id=tokenizer.token_to_id("[EOS]"),
                              forced_eos_token_id=tokenizer.token_to_id("[EOS]"),
                              vocab_size=tokenizer.get_vocab_size())

        config.save_pretrained(args.output_file)
        tokenizer.save('temp_tokenizer.json')
        from transformers import PreTrainedTokenizerFast

        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="temp_tokenizer.json")
        fast_tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        fast_tokenizer.save_pretrained(args.output_file)
        os.remove("temp_tokenizer.json")
