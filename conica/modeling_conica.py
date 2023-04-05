# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:11
# @Author : Lingo
# @File : modeling_conica.py
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
from torch.distributed import get_world_size, all_gather
from .configuration_conica import ConicaConfig
from typing import Optional, Tuple
from torch import nn
from transformers.activations import ACT2FN
from torch.nn.functional import softmax, dropout, cross_entropy, pad
from transformers.utils import logging
from transformers import PreTrainedModel
import torch
import math
import os
from copy import deepcopy

logger = logging.get_logger(__name__)
from dataclasses import dataclass


def droppath(x, p: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if p == 0. or not training:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

@dataclass
class ConicaForContrastiveGenerationModelOutput(ModelOutput):
    """
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
        Class defining the outputs of [`CONICAForContrastiveGenerationModelOutput`].

        Args:
            loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
                Language modeling loss from the language model.
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head of the language model.
            unimodal_vision_outputs (`torch.FloatTensor`):
                Outputs of the unimocal vision encoder.
            unimodal_language_outputs (`torch.FloatTensor`):
                Outputs of the unimocal language encoder.
        """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    unimodal_vision_outputs: Optional[torch.FloatTensor] = None
    unimodal_language_outputs: Optional[torch.FloatTensor] = None


@dataclass
class ConicaForUnimocalVisionOutput(ModelOutput):
    unimodal_vision_global_outputs: Optional[torch.FloatTensor] = None
    unimodal_vision_local_outputs: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ConicaForUnimocalTextOutput(ModelOutput):
    unimodal_text_global_outputs: Optional[torch.FloatTensor] = None
    unimodal_text_local_outputs: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None,
    attn_mask: Optional[Tuple[torch.FloatTensor]] = None


class ConicaLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, max_position: int, d_modal: int):
        super().__init__(max_position, d_modal)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


class ConicaVisionEmbedding(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        self.local_embeds = nn.Linear(config.d_vision_local, config.d_model)
        self.global_embeds = nn.Linear(config.d_vision_global, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.dropout = config.vision_dropout

    def forward(self, g_feats, l_feats):
        hidden_states = torch.cat((self.global_embeds(g_feats), self.local_embeds(l_feats)), dim=1)
        hidden_states = self.layer_norm(hidden_states)
        return dropout(hidden_states, p=self.dropout, training=self.training)


class ConicaAttention(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        self.config = config
        d_modal = config.d_model
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        self.scale = self.d_head ** -0.5
        self.dropout = config.dropout

        self.q_proj = nn.Linear(d_modal, d_modal)
        self.k_proj = nn.Linear(d_modal, d_modal)
        self.v_proj = nn.Linear(d_modal, d_modal)
        self.out_proj = nn.Linear(d_modal, d_modal)

    def transpose_for_scores(self, x):
        B, L, C = x.shape
        return x.transpose(0, 1).reshape(L, B * self.n_head, -1).transpose(0, 1)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False
    ):
        B, L, C = hidden_states.shape
        is_cross_attention = encoder_hidden_states is not None
        q = self.transpose_for_scores(self.q_proj(hidden_states)) * self.scale
        if is_cross_attention and past_key_value is not None:
            k = past_key_value[0]
            v = past_key_value[1]
        elif is_cross_attention:
            k = self.transpose_for_scores(self.k_proj(encoder_hidden_states))
            v = self.transpose_for_scores(self.v_proj(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            k = self.transpose_for_scores(self.k_proj(hidden_states))
            v = self.transpose_for_scores(self.v_proj(hidden_states))
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        else:
            k = self.transpose_for_scores(self.k_proj(hidden_states))
            v = self.transpose_for_scores(self.v_proj(hidden_states))
        past_key_value = (k, v)
        attention_scores = torch.bmm(q, k.transpose(-1, -2))
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)
            attention_scores = attention_scores + attention_mask
        attention_probs = softmax(attention_scores, dim=-1)
        attention_probs = dropout(attention_probs, p=self.dropout, training=self.training)
        hidden_states = torch.bmm(attention_probs, v)
        if head_mask is not None:
            hidden_states = hidden_states.view(B, self.n_head, L, C) * head_mask
            hidden_states = hidden_states.view(-1, L, C)
        hidden_states = hidden_states.transpose(0, 1).reshape(L, B, C).transpose(0, 1)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        outputs = (hidden_states, attention_probs) if output_attentions else (hidden_states,)
        outputs = outputs + (past_key_value,)
        return outputs


class ConicaFFN(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.activation_function]
        self.in_proj = nn.Linear(config.d_model, config.d_ffn)
        self.out_proj = nn.Linear(config.d_ffn, config.d_model)
        self.dropout = self.config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ConicaLayer(nn.Module):
    def __init__(self, config: ConicaConfig, cross_attention=False):
        super().__init__()
        self.config = config
        self.self_attn = ConicaAttention(config)
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        self.cross_attn = None
        if cross_attention:
            self.cross_attn = ConicaAttention(config)
            self.cross_attn_norm = nn.LayerNorm(config.d_model)
        self.ffn = ConicaFFN(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        self_attn_weights, cross_attn_weights = None, None
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = droppath(hidden_states, p=self.droppath, training=self.training)
        hidden_states = self.self_attn_norm(residual + hidden_states)
        present_key_value = self_attn_outputs[-1]
        if output_attentions:
            self_attn_weights = self_attn_outputs[1]
        if self.cross_attn is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be given for cross-attention layers")
            cross_attn_outputs = self.cross_attn(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = cross_attn_outputs[0]
            cross_present_key_value = cross_attn_outputs[-1]

            hidden_states = droppath(hidden_states, p=self.droppath, training=self.training)
            hidden_states = self.cross_attn_norm(residual + hidden_states)

            present_key_value = present_key_value + cross_present_key_value

            if output_attentions:
                cross_attn_weights = cross_attn_outputs[1]
            # add cross-attn to positions 3,4 of present_key_value tuple
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = droppath(hidden_states, p=self.droppath, training=self.training)
        hidden_states = self.ffn_norm(residual + hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class ConicaPreTrainedModel(PreTrainedModel):
    config_class = ConicaConfig
    base_model_prefix = "conica"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = None

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ConicaUnimodalVisionEncoder(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.embed_vision = ConicaVisionEmbedding(config)
        self.layers = nn.ModuleList([ConicaLayer(config, False) for _ in range(config.vision_encoder_layers)])
        self.gradient_checkpointing = False
        self.global_pool = config.vision_global_pool
        self.vision_proj = nn.Linear(config.d_model, config.d_align)
        self.post_init()

    def _expand_mask(self, attention_mask):
        bsz, src_len = attention_mask.size()
        attn_mask = attention_mask.view(bsz, 1, 1, src_len). \
            expand(-1, self.config.n_head, -1, -1).reshape(bsz * self.config.n_head, 1, src_len)
        inverted_mask = 1.0 - attn_mask
        attn_mask = inverted_mask.masked_fill(inverted_mask > 0, float('-inf'))
        return attn_mask

    def forward(
            self,
            cls_feats=None,
            local_feats=None,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz*num_heads, tgt_seq_len, src_seq_len]
            attention_mask = self._expand_mask(attention_mask, local_feats.dtype)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # check if head_mask has a correct number of layers specified if desired\

        hidden_states = self.embed_visual(cls_feats, local_feats)
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if self.global_pool:
            global_outputs, local_outputs = self.vision_proj(hidden_states.mean(1)), hidden_states
        else:
            global_outputs, local_outputs = self.vision_proj(hidden_states[:, 0, :]), hidden_states[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [global_outputs, local_outputs, hidden_states, all_hidden_states, all_attentions] if
                         v is not None)

        return ConicaForUnimocalVisionOutput(
            unimodal_vision_global_outputs=global_outputs,
            unimodal_vision_local_outputs=local_outputs,
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class ConicaUnimodalTextEncoder(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.embed_token = nn.Embedding(config.vocab_size, config.d_model)
        self.cls_embedding = nn.Parameter(config.init_std * torch.randn(config.d_model))
        self.pos_embedding = ConicaLearnedPositionalEmbedding(config.max_positions, config.d_model)
        self.layers = nn.ModuleList([ConicaLayer(config, False) for _ in range(config.vision_encoder_layers)])
        self.text_proj = nn.Linear(config.d_model, config.d_align)
        self.post_init()

    def generate_decoder_attn_mask(self, attn_mask=None, add_cls=False, tgt_len=None, device="cpu"):
        if attn_mask is None and tgt_len is None:
            raise ValueError("You have to specify either attn_mask or tgt_len")
        if attn_mask is not None:
            if add_cls:
                attn_mask = pad(attn_mask, (1, 0), value=1.)
            bsz, src_len = attn_mask.size()
            attn_mask = attn_mask.view(bsz, 1, 1, src_len). \
                expand(-1, self.config.n_head, -1, -1).reshape(bsz * self.config.n_head, 1, src_len)
            inverted_mask = 1.0 - attn_mask
            attn_mask = inverted_mask.masked_fill(inverted_mask > 0, float('-inf'))
        else:
            if add_cls:
                tgt_len += 1
            src_len = tgt_len
        if tgt_len is None:
            tgt_len = src_len
        causal_mask = torch.triu(torch.full((tgt_len, src_len), float('-inf')), diagonal=1).to(device)
        if add_cls:
            causal_mask[..., 0, :] = 0
            causal_mask[..., 1:, 0] = float('-inf')
        causal_mask = causal_mask.unsqueeze(0)
        return attn_mask + causal_mask if attn_mask is not None else causal_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_token(input_ids)
            inputs_embeds = torch.cat(
                (self.cls_emb.reshape(1, 1, -1).repeat(inputs_embeds.size(0), 1, 1)), dim=1)
