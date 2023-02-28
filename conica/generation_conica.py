# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:34
# @Author : Lingo
# @File : generation_conica.py
import string
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import torch_int_div, validate_stopping_criteria, GenerationMixin, SampleOutput, \
    SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, BeamSearchScorer, BeamSearchOutput, \
    BeamSearchEncoderDecoderOutput, \
    BeamSearchDecoderOnlyOutput
from conica.generation_contrastive_beam_search import ContrastiveBeamSearchScorer

from transformers.models.bart import BartModel


@dataclass
class COCASampleEncoderDecoderOutput(SampleEncoderDecoderOutput):
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None


@dataclass
class COCASamleDecoderOnlyOutput(SampleDecoderOnlyOutput):
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None


@dataclass
class CocaBeamSearchEncoderDecoderOutput(BeamSearchEncoderDecoderOutput):
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class CocaBeamSearchDecoderOnlyOutput(BeamSearchDecoderOnlyOutput):
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None


class CONICAGeneration(GenerationMixin):

    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            count_similarity: Optional[bool] = True,
            **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]
        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        image_embeds, text_embeds = None, None
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (nn.functional.log_softmax(next_token_scores, dim=-1),)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            if count_similarity:
                if image_embeds is None:
                    image_embeds = self.encoder_proj(outputs.encoder_hidden_states[-1][:, 0])
                    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
                    expanded_return_idx = torch.arange(image_embeds.shape[0]).view(-1, 1).repeat(1, input_ids.size(
                        0) // image_embeds.size(0)).view(-1).to(image_embeds.device)
                    image_embeds = image_embeds.index_select(0, expanded_return_idx)
                    text_embeds = torch.zeros((input_ids.size(0), image_embeds.size(1)), dtype=image_embeds.dtype,
                                              device=image_embeds.device)
                _inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                _text_embeds = self.decoder_proj(self.model.decoder(
                    input_ids=_inputs["decoder_input_ids"],
                    attention_mask=_inputs["attention_mask"],
                    head_mask=_inputs["head_mask"],
                    past_key_values=_inputs["past_key_values"],
                    use_cache=_inputs["use_cache"],
                    return_unimodal_feature_only=True
                )['hidden_states'][self.config.unimodal_decoder_layers][:, -1])
                _text_embeds = nn.functional.normalize(_text_embeds, dim=-1)
                text_embeds[next_tokens == eos_token_id] += _text_embeds[next_tokens == eos_token_id]

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        if return_dict_in_generate:

            # forward pass to get next token

            if count_similarity:
                similarity = torch.einsum("ij,ij->i", image_embeds, text_embeds)
            else:
                similarity = None

            if self.config.is_encoder_decoder:
                return COCASampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    similarity=similarity,
                    image_embeds=image_embeds,
                    text_embeds=text_embeds
                )
            else:
                return COCASamleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    similarity=similarity,
                    image_embeds=image_embeds,
                    text_embeds=text_embeds
                )
        else:
            return input_ids

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        beam_scorer = ContrastiveBeamSearchScorer(len(beam_scorer._beam_hyps), beam_scorer.num_beams,
                                                  beam_scorer.device,
                                                  beam_scorer.length_penalty, beam_scorer.do_early_stopping,
                                                  beam_scorer.num_beam_hyps_to_keep,
                                                  beam_scorer.num_beam_groups,
                                                  alpha=self.config.alpha)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        image_embeds = None
        this_peer_finished = False  # used by synced_gpus only
        logprobs = None
        count_similarity = self.config.count_similarity

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_logits = outputs.logits[:, -1, :]
            next_logits = self.adjust_logits_during_generation(next_logits, cur_len=cur_len)
            next_logprobs = nn.functional.log_softmax(
                next_logits, dim=-1,
            )  # (batch_size * num_beams, vocab_size)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_logprobs,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            next_token_scores_processed = logits_processor(input_ids, next_logprobs.clone())
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_logprobs)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_tokens_logprob = next_logprobs.view(batch_size, -1).gather(1, next_tokens.clone())
            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            if image_embeds is None:
                image_embeds = self.encoder_proj(outputs.encoder_hidden_states[-1][:, 0])
                image_embeds = nn.functional.normalize(image_embeds, dim=-1)
                expanded_return_idx = torch.arange(image_embeds.shape[0]).view(-1, 1).repeat(1, num_beams).view(
                    -1).to(image_embeds.device)
                image_embeds = image_embeds.index_select(0, expanded_return_idx)
            _model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            _input_ids = torch.cat((input_ids, torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
                                                           dtype=torch.long).fill_(eos_token_id)), dim=1)
            _inputs = self.prepare_inputs_for_generation(_input_ids, **_model_kwargs)
            text_embeds = self.decoder_proj(self.model.decoder(
                input_ids=_inputs["decoder_input_ids"],
                attention_mask=_inputs["attention_mask"],
                head_mask=_inputs["head_mask"],
                past_key_values=_inputs["past_key_values"],
                use_cache=_inputs["use_cache"],
                return_unimodal_feature_only=True
            )['hidden_states'][self.config.unimodal_decoder_layers][:, -1])
            text_embeds = nn.functional.normalize(text_embeds, dim=-1)

            similarity = torch.einsum("ij,ij->i", image_embeds, text_embeds)
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                logprobs,
                next_tokens_logprob,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                similarity=similarity,
                count_similarity=count_similarity
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_logprobs = beam_outputs["next_beam_logprobs"]
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            logprobs = beam_logprobs.unsqueeze(-1) if logprobs is None else torch.cat([logprobs[beam_idx, :],
                                                                                        beam_logprobs.unsqueeze(-1)],
                                                                                        dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        _model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
        _input_ids = torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
                                 dtype=torch.long).fill_(eos_token_id)
        _inputs = self.prepare_inputs_for_generation(_input_ids, **_model_kwargs)
        text_embeds = self.decoder_proj(self.model.decoder(
            input_ids=_inputs["decoder_input_ids"],
            attention_mask=_inputs["attention_mask"],
            head_mask=_inputs["head_mask"],
            past_key_values=_inputs["past_key_values"],
            use_cache=_inputs["use_cache"],
            return_unimodal_feature_only=True
        )['hidden_states'][self.config.unimodal_decoder_layers][:, -1])
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)
        similarity = torch.einsum("ij,ij->i", image_embeds, text_embeds)
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            logprobs,
            next_tokens_logprob,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            similarity=similarity,
            count_similarity=self.config.count_similarity
        )
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return CocaBeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    similarity=sequence_outputs["similarity"],
                    image_embeds=image_embeds,
                    text_embeds=text_embeds,
                    logprobs=sequence_outputs["logprobs"],
                )
            else:
                return CocaBeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    similarity=sequence_outputs["similarity"],
                    image_embeds=image_embeds,
                    text_embeds=text_embeds
                )
        else:
            return sequence_outputs["sequences"]

#todo: implement group contrastive beam search
    # def group_beam_search(
    #         self,
    #         input_ids: torch.LongTensor,
    #         beam_scorer,
    #         logits_processor: Optional[LogitsProcessorList] = None,
    #         stopping_criteria: Optional[StoppingCriteriaList] = None,
    #         max_length: Optional[int] = None,
    #         pad_token_id: Optional[int] = None,
    #         eos_token_id: Optional[int] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         output_scores: Optional[bool] = None,
    #         return_dict_in_generate: Optional[bool] = None,
    #         synced_gpus: Optional[bool] = False,
    #         count_similarity: Optional[bool] = True,
    #         **model_kwargs,
    # ):
    #     beam_scorer = ContrastiveBeamSearchScorer(beam_scorer)
    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    #     if max_length is not None:
    #         warnings.warn(
    #             "`max_length` is deprecated in this function, use"
    #             " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
    #             UserWarning,
    #         )
    #         stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    #     pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    #     eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    #     output_scores = output_scores if output_scores is not None else self.config.output_scores
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict_in_generate = (
    #         return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    #     )
    #
    #     batch_size = len(beam_scorer._beam_hyps)
    #     num_beams = beam_scorer.num_beams
    #     num_beam_groups = beam_scorer.num_beam_groups
    #     num_sub_beams = num_beams // num_beam_groups
    #     device = input_ids.device
    #
    #     batch_beam_size, cur_len = input_ids.shape
    #
    #     if return_dict_in_generate and output_scores:
    #         beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
    #     else:
    #         beam_indices = None
    #
    #     if num_beams * batch_size != batch_beam_size:
    #         raise ValueError(
    #             f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
    #         )
    #
    #     # init attention / hidden states / scores tuples
    #     scores = () if (return_dict_in_generate and output_scores) else None
    #     decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    #     cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    #     decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    #
    #     # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    #     if return_dict_in_generate and self.config.is_encoder_decoder:
    #         encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
    #         encoder_hidden_states = (
    #             model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
    #         )
    #
    #     beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    #     # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
    #     # the same group don't produce same tokens everytime.
    #     beam_scores[:, ::num_sub_beams] = 0
    #     beam_scores = beam_scores.view((batch_size * num_beams,))
    #
    #     this_peer_finished = False
    #     image_embeds = None
    #     while True:
    #
    #         if synced_gpus:
    #             # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
    #             # The following logic allows an early break if all peers finished generating their sequence
    #             this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
    #             # send 0.0 if we finished, 1.0 otherwise
    #             dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
    #             # did all peers finish? the reduced sum will be 0.0 then
    #             if this_peer_finished_flag.item() == 0.0:
    #                 break
    #
    #         # predicted tokens in cur_len step
    #         current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)
    #
    #         # indices which will form the beams in the next time step
    #         reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)
    #
    #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
    #         outputs = self(
    #             **model_inputs,
    #             return_dict=True,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #         )
    #         if count_similarity:
    #             if image_embeds is None:
    #                 image_embeds = self.encoder_proj(outputs.encoder_hidden_states[-1][:, 0])
    #                 image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    #                 expanded_return_idx = torch.arange(image_embeds.shape[0]).view(-1, 1).repeat(1, num_beams).view(
    #                     -1).to(
    #                     image_embeds.device)
    #                 image_embeds = image_embeds.index_select(0, expanded_return_idx)
    #             cls_model_kwargs = self._update_model_kwargs_for_generation(
    #                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #             )
    #             _input_ids = torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
    #                                      dtype=torch.long).fill_(eos_token_id)
    #             # _input_ids = torch.cat((input_ids, _input_ids), dim=1)
    #             _inputs = self.prepare_inputs_for_generation(_input_ids, **cls_model_kwargs)
    #
    #             text_embeds = self.decoder_proj(self.model.decoder(
    #                 input_ids=_inputs["decoder_input_ids"],
    #                 attention_mask=_inputs["attention_mask"],
    #                 head_mask=_inputs["head_mask"],
    #                 past_key_values=_inputs["past_key_values"],
    #                 use_cache=_inputs["use_cache"],
    #                 return_unimodal_feature_only=True
    #             )['hidden_states'][self.config.unimodal_decoder_layers][:, -1])
    #
    #             text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    #             similarity = torch.einsum("ij,ij->i", image_embeds, text_embeds)
    #         else:
    #             similarity = None
    #         if synced_gpus and this_peer_finished:
    #             cur_len = cur_len + 1
    #             continue  # don't waste resources running the code we don't need
    #
    #         if output_scores:
    #             processed_score = torch.zeros_like(outputs.logits[:, -1, :])
    #             output_score = torch.zeros_like(outputs.logits[:, -1, :])
    #
    #         for beam_group_idx in range(num_beam_groups):
    #             group_start_idx = beam_group_idx * num_sub_beams
    #             group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
    #             group_size = group_end_idx - group_start_idx
    #
    #             # indices of beams of current group among all sentences in batch
    #             batch_group_indices = []
    #
    #             for batch_idx in range(batch_size):
    #                 batch_group_indices.extend(
    #                     [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
    #                 )
    #             group_input_ids = input_ids[batch_group_indices]
    #             next_token_logits = outputs.logits[batch_group_indices, -1, :]
    #
    #             # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
    #             # cannot be generated both before and after the `nn.functional.log_softmax` operation.
    #             next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
    #             next_token_scores = nn.functional.log_softmax(
    #                 next_token_logits, dim=-1
    #             )  # (batch_size * group_size, vocab_size)
    #             if output_scores:
    #                 output_score[batch_group_indices] = next_token_scores.clone()
    #             vocab_size = next_token_scores.shape[-1]
    #
    #             next_token_scores_processed = logits_processor(
    #                 group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
    #             )
    #             next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
    #             next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
    #
    #             # reshape for beam search
    #             next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)
    #
    #             next_token_scores, next_tokens = torch.topk(
    #                 next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
    #             )
    #
    #             next_indices = torch_int_div(next_tokens, vocab_size)
    #             next_tokens = next_tokens % vocab_size
    #
    #             # stateless
    #             process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    #             beam_outputs = beam_scorer.process(
    #                 group_input_ids,
    #                 next_token_scores,
    #                 next_tokens,
    #                 next_indices,
    #                 pad_token_id=pad_token_id,
    #                 eos_token_id=eos_token_id,
    #                 beam_indices=process_beam_indices,
    #                 similarity=similarity
    #             )
    #             beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
    #             beam_next_tokens = beam_outputs["next_beam_tokens"]
    #             beam_idx = beam_outputs["next_beam_indices"]
    #             if return_dict_in_generate and output_scores:
    #                 beam_indices[beam_group_idx] = tuple(
    #                     beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
    #                 )
    #
    #             input_ids[batch_group_indices] = group_input_ids[beam_idx]
    #             group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    #             current_tokens[batch_group_indices] = group_input_ids[:, -1]
    #             reordering_indices[batch_group_indices] = (
    #                     num_beams * torch_int_div(beam_idx, group_size) + group_start_idx + (beam_idx % group_size)
    #             )
    #         if return_dict_in_generate:
    #             if output_scores:
    #                 scores += (output_score,)
    #             if output_attentions:
    #                 decoder_attentions += (
    #                     (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
    #                 )
    #                 if self.config.is_encoder_decoder:
    #                     cross_attentions += (outputs.cross_attentions,)
    #
    #             if output_hidden_states:
    #                 decoder_hidden_states += (
    #                     (outputs.decoder_hidden_states,)
    #                     if self.config.is_encoder_decoder
    #                     else (outputs.hidden_states,)
    #                 )
    #
    #         input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
    #
    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         )
    #         if model_kwargs["past"] is not None:
    #             model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], reordering_indices)
    #
    #         # increase cur_len
    #         cur_len = cur_len + 1
    #         if beam_scorer.is_done or stopping_criteria(input_ids, scores):
    #             if not synced_gpus:
    #                 break
    #             else:
    #                 this_peer_finished = True
    #     if count_similarity:
    #         cls_model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         )
    #         _input_ids = torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
    #                                  dtype=torch.long).fill_(eos_token_id)
    #         _inputs = self.prepare_inputs_for_generation(_input_ids, **cls_model_kwargs)
    #
    #         text_embeds = self.decoder_proj(self.model.decoder(
    #             input_ids=_inputs["decoder_input_ids"],
    #             attention_mask=_inputs["attention_mask"],
    #             head_mask=_inputs["head_mask"],
    #             past_key_values=_inputs["past_key_values"],
    #             use_cache=_inputs["use_cache"],
    #             return_unimodal_feature_only=True
    #         )['hidden_states'][self.config.unimodal_decoder_layers][:, -1])
    #         text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    #
    #         similarity = torch.einsum("ij,ij->i", image_embeds, text_embeds)
    #     else:
    #         similarity = None
    #         image_embeds = None
    #         text_embeds = None
    #     final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    #     sequence_outputs = beam_scorer.finalize(
    #         input_ids,
    #         beam_scores,
    #         next_tokens,
    #         next_indices,
    #         pad_token_id=pad_token_id,
    #         eos_token_id=eos_token_id,
    #         max_length=stopping_criteria.max_length,
    #         beam_indices=final_beam_indices,
    #         similarity=similarity
    #     )
    #     if return_dict_in_generate:
    #         if not output_scores:
    #             sequence_outputs["sequence_scores"] = None
    #
    #         if self.config.is_encoder_decoder:
    #             return CocaBeamSearchEncoderDecoderOutput(
    #                 sequences=sequence_outputs["sequences"],
    #                 sequences_scores=sequence_outputs["sequence_scores"],
    #                 scores=scores,
    #                 beam_indices=sequence_outputs["beam_indices"],
    #                 encoder_attentions=encoder_attentions,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 decoder_attentions=decoder_attentions,
    #                 cross_attentions=cross_attentions,
    #                 decoder_hidden_states=decoder_hidden_states,
    #                 similarity=sequence_outputs["similarity"],
    #                 image_embeds=image_embeds,
    #                 text_embeds=text_embeds
    #             )
    #         else:
    #             return CocaBeamSearchDecoderOnlyOutput(
    #                 sequences=sequence_outputs["sequences"],
    #                 sequences_scores=sequence_outputs["sequence_scores"],
    #                 scores=scores,
    #                 beam_indices=sequence_outputs["beam_indices"],
    #                 attentions=decoder_attentions,
    #                 hidden_states=decoder_hidden_states,
    #                 similarity=sequence_outputs["similarity"],
    #                 image_embeds=image_embeds,
    #                 text_embeds=text_embeds
    #             )
    #     else:
    #         return sequence_outputs["sequences"]
