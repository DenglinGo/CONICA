# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:11
# @Author : Lingo
# @File : modeling_conica.py

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from conica.generation_conica import CONICAGeneration
from torch.distributed import get_world_size, all_gather
from .configuration_conica import CONICAConfig
from typing import Optional, Tuple
from torch import nn
from transformers.activations import ACT2FN
from torch.nn.functional import softmax, dropout, cross_entropy
from transformers.utils import logging
from transformers import PreTrainedModel
import torch
import math
import os
from copy import deepcopy

logger = logging.get_logger(__name__)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class CONICALearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, max_position: int, embedding_dim: int):
        super().__init__(max_position, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


class CONICAAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # decoder cross_attention
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # decoder cross_attention without pre-computed k,v
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # decoder self_attention
            # reuse k, v
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # encoder/decoder self_attention without pask key and value
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can
        # be partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class CONICAEncoderLayer(nn.Module):
    def __init__(self, config: CONICAConfig):
        super().__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.embed_dim = config.d_model
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

        self.self_attn = CONICAAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.dropout,
            bias=config.bias
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.fc_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.fc_proj = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.out_proj = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if self.pre_layer_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.pre_layer_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        if self.pre_layer_norm:
            hidden_states = self.fc_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc_proj(hidden_states))
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.pre_layer_norm:
            hidden_states = self.fc_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CONICADecoderLayer(nn.Module):
    def __init__(self, config: CONICAConfig, cross_attention=True):
        super().__init__()

        self.pre_layer_norm = config.pre_layer_norm
        self.embed_dim = config.d_model
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.self_attn = CONICAAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout
        self.droppath = config.droppath
        self.activation_fn = ACT2FN[config.activation_function]

        self.cross_attn = None
        if cross_attention:
            self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

            self.cross_attn = CONICAAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.dropout,
                is_decoder=True,
            )
        self.fc_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

        self.fc_proj = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.out_proj = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

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
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        if self.pre_layer_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.pre_layer_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None and self.cross_attn is not None:
            residual = hidden_states

            if self.pre_layer_norm:
                hidden_states = self.cross_attn_layer_norm(hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            if not self.pre_layer_norm:
                hidden_states = self.cross_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value

            # add cross-attn to positions 3,4 of present_key_value tuple
        residual = hidden_states
        if self.pre_layer_norm:
            hidden_states = self.fc_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc_proj(hidden_states))
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.pre_layer_norm:
            hidden_states = self.fc_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class CONICAPretrainedModel(PreTrainedModel):
    config_class = CONICAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]
    main_input_name = 'inputs_embeds'

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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CONICAEncoder, CONICADecoder)):
            module.gradient_checkpointing = value


class CONICAEncoder(CONICAPretrainedModel):
    def __init__(self, config: CONICAConfig):
        super().__init__(config)

        self.embed_visual = nn.Linear(config.d_visual, config.d_model, bias=config.bias)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.layers = nn.ModuleList([CONICAEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim, config.layer_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.final_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps) \
                               if config.pre_layer_norm else nn.Identity()

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_visual

    def set_input_embeddings(self, value):
        self.embed_visual = value

    def forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size,num_regions/num_grids,hidden_size)`, *optional*):
                Directly pass an embedded representation of images.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # check if head_mask has a correct number of layers specified if desired\
        hidden_states = self.embed_visual(inputs_embeds)
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
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
        hidden_states = self.final_layer_norm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class CONICADecoder(CONICAPretrainedModel):
    def __init__(self, config: CONICAConfig):
        super().__init__(config)

        self.dropout = config.dropout
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.layers = nn.ModuleList(
            [CONICADecoderLayer(config, cross_attention=False) for _ in range(config.unimodal_decoder_layers)] + [
                CONICADecoderLayer(config, cross_attention=True) for _ in range(config.multimodal_decoder_layers)])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = CONICALearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.embed_layer_norm = nn.LayerNorm(config.d_model,
                                             config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.d_model,
                                             config.layer_norm_eps) if config.pre_layer_norm else nn.Identity()

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_unimodal_feature_only: bool = False
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all ``decoder_input_ids``` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor`
                of shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
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
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.embed_layer_norm(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
            if return_unimodal_feature_only and idx == self.config.unimodal_decoder_layers - 1:
                break

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_unimodal_feature_only:
            hidden_states = self.final_layer_norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class CONICAModel(CONICAPretrainedModel):
    def __init__(self, config: CONICAConfig):
        super().__init__(config)
        self.encoder = CONICAEncoder(config)
        self.decoder = CONICADecoder(config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.decoder.embed_tokens = new_embeddings

    def forward(
            self,
            inputs_embeds,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                "`decoder_input_ids` and `decoder_inputs_embeds` "
                "can not be None at the same time "
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        expand_size = 1
        if encoder_outputs is None:
            #
            if decoder_input_ids is None:
                expand_size = decoder_inputs_embeds.size(0) // inputs_embeds.size(0)
            elif decoder_inputs_embeds is None:
                expand_size = decoder_input_ids.size(0) // inputs_embeds.size(0)
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            expanded_return_idx = torch.arange(inputs_embeds.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(
                inputs_embeds.device)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        encoder_hidden_states = encoder_outputs[0].index_select(0, expanded_return_idx.to(
            encoder_outputs.last_hidden_state.device)) if expand_size > 1 else encoder_outputs[0]
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if self.training:
            encoder_hidden_states = encoder_hidden_states[:, 1:]
            if attention_mask is not None:
                attention_mask = attention_mask[..., 1:]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask if attention_mask is not None else None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class CONICAModelWithCONICA(CONICAPretrainedModel, CONICAGeneration):
    def __init__(self, config: CONICAConfig,
                 seq_per_image: int,
                 ):
        super().__init__(config)
        self.model = CONICAModel(config)
        self.predict_dropout = config.predict_dropout
        self.encoder_proj = nn.Sequential(
            nn.LayerNorm(config.d_model, config.layer_norm_eps) if config.pre_layer_norm else nn.Identity(),
            nn.Linear(config.d_model, config.d_align)
        )
        self.decoder_proj = nn.Sequential(
            nn.LayerNorm(config.d_model, config.layer_norm_eps) if config.pre_layer_norm else nn.Identity(),
            nn.Linear(config.d_model, config.d_align)
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.tau = nn.Parameter(torch.ones([]) * config.tau)

        self.post_init()

        self.momentum = config.momentum

        config_momentum = deepcopy(config)
        config_momentum.multimodal_decoder_layers = 0

        self.model_m = CONICAModel(config_momentum)
        self.encoder_proj_m = nn.Sequential(
            nn.LayerNorm(config.d_model, config.layer_norm_eps) if config.pre_layer_norm else nn.Identity(),
            nn.Linear(config.d_model, config.d_align)
        )
        self.decoder_proj_m = nn.Sequential(
            nn.LayerNorm(config.d_model, config.layer_norm_eps) if config.pre_layer_norm else nn.Identity(),
            nn.Linear(config.d_model, config.d_align)
        )

        self.model_pair = [[self.model_m, self.model],
                           [self.encoder_proj_m, self.encoder_proj],
                           [self.decoder_proj_m, self.decoder_proj]]

        self.copy_params()

        self.register_buffer("image_queue", torch.randn(config.queue_size, config.d_align))
        self.register_buffer("text_queue", torch.randn(config.queue_size * seq_per_image, config.d_align))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=1)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=1)
        self.seq_per_image = seq_per_image

    def copy_params(self):
        for model_m, model in self.model_pair:
            for name_m, parameter_m in model_m.named_parameters():
                parameter_m.data.copy_(model.state_dict()[name_m].data)
                parameter_m.requires_grad = False

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, *args, **kwargs):
        is_generate = kwargs.get('is_generate', False)
        if is_generate:
            kwargs.pop("is_generate")
            return super().generate.__wrapped__(self, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def get_image_features(
            self,
            inputs_embeds=None,
            attention_mask=None,
            head_mask=None):
        return self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask
        )[2][-1]

    def get_text_features(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,

    ):
        return self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_unimodal_feature_only=True
        )[2][-1]

    def _dequeue_and_enqueue(self, image_embeds, text_embeds):
        image_embeds = self.concat_all_gather(image_embeds)
        text_embeds = self.concat_all_gather(text_embeds)

        image_embeds = torch.cat((image_embeds, self.image_queue.clone().detach()), dim=0)
        text_embeds = torch.cat((text_embeds, self.text_queue.clone().detach()), dim=0)
        self.image_queue = image_embeds[:len(self.image_queue)]
        self.text_queue = text_embeds[:len(self.text_queue)]

    @torch.no_grad()
    def _momentum_update(self):
        for model_m, model in self.model_pair:
            for name_m, parameter_m in model_m.named_parameters():
                parameter = model.state_dict()[name_m]
                parameter_m.data = parameter_m.data * self.momentum + parameter.data * (1 - self.momentum)

    @torch.no_grad()
    @staticmethod
    def concat_all_gather(tensor):
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def _forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = labels
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(dropout(outputs[0],p=self.predict_dropout,training=self.training))
        loss = None

        if labels is not None:
            if output_hidden_states:
                image_embeds = self.encoder_proj(outputs.encoder_hidden_states[-1][:, 0])
                text_embeds = self.decoder_proj(
                    outputs.decoder_hidden_states[self.config.unimodal_decoder_layers][
                        torch.arange(decoder_input_ids.size(0)), decoder_input_ids.argmax(dim=-1)]
                )

                image_embeds = nn.functional.normalize(image_embeds, dim=-1)
                text_embeds = nn.functional.normalize(text_embeds, dim=-1)
                with torch.no_grad():
                    self.tau.clamp_(min=0.01, max=0.5)
                    self._momentum_update()

                    outputs_m = self.model_m(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        head_mask=head_mask,
                        decoder_head_mask=decoder_head_mask,
                        cross_attn_head_mask=cross_attn_head_mask,
                        encoder_outputs=encoder_outputs,
                        past_key_values=past_key_values,
                        decoder_inputs_embeds=decoder_inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    image_embeds_m = self.encoder_proj_m(outputs_m.encoder_hidden_states[-1][:, 0])
                    text_embeds_m = self.decoder_proj_m(
                        outputs_m.decoder_hidden_states[self.config.unimodal_decoder_layers][
                            torch.arange(decoder_input_ids.size(0)), decoder_input_ids.argmax(dim=-1)])
                    image_embeds_m = nn.functional.normalize(image_embeds_m, dim=-1)
                    text_embeds_m = nn.functional.normalize(text_embeds_m, dim=-1)
                    image_embeds_all = torch.cat([image_embeds_m, self.image_queue.clone().detach()], dim=0)
                    text_embeds_all = torch.cat([text_embeds_m, self.text_queue.clone().detach()], dim=0)

                expand_size = 1
                if decoder_input_ids is None:
                    expand_size = decoder_inputs_embeds.size(0) // inputs_embeds.size(0)
                elif decoder_inputs_embeds is None:
                    expand_size = decoder_input_ids.size(0) // inputs_embeds.size(0)
                sim_i2t = torch.div(torch.matmul(image_embeds, text_embeds_all.t()), self.tau)
                sim_t2i = torch.div(torch.matmul(text_embeds, image_embeds_all.t()), self.tau)
                sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
                sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
                for i in range(len(sim_i2t)):
                    sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
                    sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1
                co_loss = (cross_entropy(sim_i2t, sim_i2t_target) + cross_entropy(sim_t2i,
                                                                                  sim_t2i_target)) / 2 * self.config.co_weight
                loss = co_loss
                self._dequeue_and_enqueue(image_embeds_m, text_embeds_m)

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = cross_entropy(shift_logits.view(-1, self.config.vocab_size),
                                        shift_labels.view(-1),
                                        ignore_index=self.config.pad_token_id) * self.config.xe_weight
                loss += lm_loss

        if not return_dict:
            output = (loss,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def concat_all_gather(tensor):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            return tensor
        tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
        all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @staticmethod
    def expand_cache(past, next_indices, batch_size, num_beams):
        next_indices = next_indices + torch.arange(0, batch_size, device=next_indices.device).unsqueeze(1) * num_beams
        # print(next_indices)
        next_indices = next_indices.view(-1)
        # return past
        new_past = []
        if past is None:
            return None
        for layer in past:
            items = []
            for item in layer:
                item = item.index_select(0, next_indices)
                items.append(item)
            new_past.append(items)

        return new_past
