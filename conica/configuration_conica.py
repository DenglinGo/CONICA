# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:37
# @Author : Lingo
# @File : configuration_conica.py

from transformers.configuration_utils import PretrainedConfig


class ConicaConfig(PretrainedConfig):
    model_type = "conica"

    def __init__(
            self,
            vocab_size=49408,
            d_model=512,
            d_vision_local=2048,
            d_vision_global=2048,
            d_align=128,
            vision_global_pool = False,
            max_positions=64,
            vision_encoder_layers=3,
            text_encoder_layers=3,
            n_head=8,
            multimodal_decoder_layers=3,
            ffn_ratio=4,
            activation_function="gelu",
            dropout=0.1,
            droppath=0.1,
            init_std=0.02,
            scale_embedding=False,
            use_cache=True,
            pad_token_id=0,
            queue_size=8192,
            momentum=0.995,
            bos_token_id=49406,
            eos_token_id=49407,
            is_encoder_decoder=True,
            forced_eos_token_id=49407,
            top_k=0,
            max_length=32,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            tau=0.07,
            output_hidden_states=True,
            alpha=0.5,
            vision_dropout=0.1,
            predict_dropout=0.1,
            xe_weight=1,
            co_weight=1,
            label_smoothing=0.1,
            count_similarity=True,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_vision_local = d_vision_local
        self.d_vision_global = d_vision_global
        self.d_model = d_model
        self.d_align = d_align
        self.d_ffn = d_model*ffn_ratio
        self.n_head = n_head
        self.tau = tau
        self.vision_global_pool = vision_global_pool
        self.max_positions = max_positions
        self.vision_encoder_layers = vision_encoder_layers
        self.text_encoder_layers = text_encoder_layers
        self.multimodal_decoder_layers = multimodal_decoder_layers

        self.predict_dropout = predict_dropout
        self.dropout = dropout
        self.droppath = droppath
        self.vision_dropout = vision_dropout
        self.activation_function = activation_function
        self.queue_size = queue_size
        self.momentum = momentum
        self.init_std = init_std
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.alpha = alpha
        self.xe_weight = xe_weight
        self.co_weight = co_weight
        self.count_similarity = count_similarity
        self.label_smoothing = label_smoothing
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            max_length=max_length,
            top_k=top_k,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            do_sample=do_sample,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
