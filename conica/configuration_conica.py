# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:37
# @Author : Lingo
# @File : configuration_conica.py

from transformers.configuration_utils import PretrainedConfig


class CONICAConfig(PretrainedConfig):
    model_type = "conica"

    def __init__(
            self,
            vocab_size=49408,
            d_visual=2304,
            d_align=128,
            max_position_embeddings=64,
            encoder_layers=3,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            unimodal_decoder_layers=3,
            multimodal_decoder_layers=3,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            activation_function="gelu",
            layer_norm_eps=1e-5,
            pre_layer_norm=False,
            d_model=512,
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
            xe_weight=2,
            co_weight=1,
            count_similarity=False,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_visual = d_visual
        self.tau = tau
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.d_align = d_align
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.unimodal_decoder_layers = unimodal_decoder_layers
        self.multimodal_decoder_layers = multimodal_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.droppath = droppath
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.pre_layer_norm = pre_layer_norm
        self.queue_size = queue_size
        self.momentum = momentum
        self.init_std = init_std
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.alpha = alpha
        self.xe_weight = xe_weight
        self.co_weight = co_weight
        self.count_similarity = count_similarity
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
