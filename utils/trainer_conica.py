# -*- coding: utf-8 -*-
# @Time : 2023/4/5 下午6:40
# @Author : Lingo
# @File : trainer_conica.py

import torch.nn.functional
from torch import nn
from transformers.trainer import *
from transformers.trainer_utils import PredictionOutput
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Union, Any, Tuple
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import EvalLoopOutput
from torch.nn.utils.rnn import pad_sequence
from utils.ptbtokenizer import PTBTokenizer
from utils.cider import Cider
from utils.bleu import Bleu
from utils.rouge import Rouge
from utils.meteor import Meteor
from utils.spice import Spice


class ConicaTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 train_cached_cider=None,
                 scst=False,
                 scst_num_sample_sequences=None,
                 scst_baseline_type=None,
                 add_mean_cls=True,
                 init_tau=False):
        super().__init__(model,
                         args,
                         None,
                         train_dataset,
                         eval_dataset, tokenizer,
                         model_init, None,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics)
        self.train_cider = Cider(cider_cached=train_cached_cider)
        self.data_collator = self.caption_collator
        self.scst = scst
        self.count_similarity = model.config.count_similarity
        self.compute_metrics = self.compute_cider_rouge_and_bleu
        self.add_mean_cls = add_mean_cls
        self.init_tau = False
        if self.scst:
            self.scst_num_sample_sequences = scst_num_sample_sequences
            self.scst_baseline_type = scst_baseline_type
            self.init_tau = init_tau
        self.ptb_tokenizer = PTBTokenizer()

    def caption_collator(self, batch):
        feats, caps, gts = zip(*batch)
        sentences = []
        for cap in caps:
            sentences.extend(cap)
        _gts = []
        for gt in gts:
            _gts.append(gt)
        gts = _gts

        outputs = self.tokenizer.batch_encode_plus(sentences, padding=True, return_tensors='pt', max_length=62,
                                                   truncation=True)
        feats = pad_sequence([torch.from_numpy(feat) for feat in feats], batch_first=True)
        del sentences
        return {
            'vision_feats': feats,
            'attention_mask': (feats.mean(-1) != 0).to(torch.float32),
            'labels': outputs['input_ids'],
            'decoder_attention_mask': outputs['attention_mask'],
            'gts': gts}

    def compute_cider_rouge_and_bleu(self, inputs: EvalPrediction):
        predictions, gts = inputs
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        res = self.ptb_tokenizer.tokenize([[_] for _ in predictions])
        results = {}
        bleu, bleus = Bleu(4).compute_score(gts, res)
        for i, bleu_i in enumerate(bleu):
            results['bleu_' + str(i + 1)] = bleu_i
        rouge, rouges = Rouge().compute_score(gts, res)
        results['rouge'] = rouge
        cider, ciders = Cider(cider_cached=None).compute_score(gts, res)
        results["cider"] = ciders
        meteor, meteors = Meteor().compute_score(gts, res)
        results["meteor"] = meteor
        spice, spices = Spice().compute_score(gts, res)
        results["spice"] = spice
        return results

    def compute_training_cider(self, inputs: EvalPrediction):
        predictions, gts = inputs
        sample_num = len(predictions)
        batch_size = len(gts)
        sample_per_image = sample_num // batch_size

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)
        _gts = []
        for _ in gts:
            _ = self.tokenizer.batch_encode_plus(_, add_special_tokens=False, max_length=62, truncation=True)
            _ = self.tokenizer.batch_decode(_.input_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

            _gts.append(_)
        res_ = []
        gts_ = []
        for i in range(sample_num):
            gts_.append([_gts[i // sample_per_image][j] + " " + self.tokenizer.eos_token for j in
                         range(len(gts[i // sample_per_image]))])
            res_.append([predictions[i].replace(self.tokenizer.bos_token, "").replace(self.tokenizer.pad_token, "")])
        cider, ciders = self.train_cider.compute_score(gts_, res_)
        return {'cider': ciders}

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = None
        if self.scst:
            labels = inputs.pop("labels")
            gts = inputs.pop("gts")
        elif self.label_smoother is not None and "labels" in inputs:
            ids = inputs.pop("labels")
            inputs['decoder_input_ids'] = ids
            labels = ids[:, 1:]
            labels.masked_fill(labels == self.model.config.pad_token_id, -100)
        else:
            gts = inputs.pop("gts")
        if not self.scst:
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            if labels is not None and self.label_smoother is not None:
                if isinstance(outputs, dict):
                    outputs["logits"] = outputs["logits"][:, :-1, :]

                else:
                    outputs[0] = outputs[0][:, :-1, :]
                loss = self.label_smoother(outputs, labels)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            batch_size = len(inputs["vision_feats"])

            if self.model.config.do_sample:
                gen_kwargs = {
                    "max_length": None if self.model.config.max_length is None else self.model.config.max_length,
                    "use_cache": self.model.config.use_cache,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                    "do_sample": True,
                    "num_return_sequences": self.scst_num_sample_sequences,
                    'is_generate': True
                }
            else:
                gen_kwargs = {
                    "max_length": None if self.model.config.max_length is None else self.model.config.max_length,
                    "num_beams": self.scst_num_sample_sequences,
                    "use_cache": self.model.config.use_cache,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                    "do_sample": False,
                    "num_return_sequences": self.scst_num_sample_sequences,
                    'is_generate': True
                }

            outputs = model(**inputs, **gen_kwargs)
            logprobs = outputs["logprobs"]

            num_sample = len(outputs["sequences"]) // batch_size
            # for i in range (len(outputs["sequences"])):
            #     print(outputs["sequences"][i])
            #     print(outputs["logprobs"][i])

            similarity = outputs["similarity"].view(-1, num_sample)
            with torch.no_grad():

                sample_rewards = torch.as_tensor(
                    self.compute_training_cider(EvalPrediction(predictions=outputs["sequences"], label_ids=gts))[
                        'cider'].reshape(batch_size, num_sample), device=logprobs.device, dtype=torch.float32)
                if self.scst_baseline_type == 'greedy':
                    greedy_kwargs = {
                        "max_length": self.model.config.max_length,
                        "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                        'use_cache': self.model.config.use_cache,
                        'num_return_sequences': 1,
                        'num_beams': 1,
                        'do_sample': False,
                        'is_generate': True,
                    }
                    model.eval()
                    greedy_outputs = model(**inputs, **greedy_kwargs)
                    model.train()
                    base_rewards = \
                        torch.as_tensor(self.compute_training_cider(
                            EvalPrediction(predictions=greedy_outputs['sequences'], label_ids=gts))[
                                            'cider'].reshape(batch_size, 1), device=logprobs.device,
                                        dtype=torch.float32)

                elif self.scst_baseline_type == "avg_rest":
                    base_rewards = (sample_rewards.sum(1, keepdim=True) - sample_rewards) / (num_sample - 1)
                else:
                    base_rewards = sample_rewards.mean(1, keepdim=True)
                reward = (sample_rewards - base_rewards).view(-1, 1)

            loss = -(logprobs * reward).sum(-1) / model.config.max_length
            loss = loss.mean()
            # ListMLE
            # sorted_reward, sorted_idx = sample_rewards.sort(-1, descending=True)
            # sorted_similarity = similarity.gather(-1, sorted_idx)
            # similarity_max,_ = similarity.max(-1, keepdim=True)
            # similarity_logits = sorted_similarity - similarity_max
            # logcumsumexp = torch.logcumsumexp(similarity_logits.flip(-1),1).flip(-1)
            # loss += ((logcumsumexp) - similarity_logits).sum(-1).mean()

            # ListNet
            similarity_label = torch.softmax(reward.view(-1, num_sample), dim=-1)
            loss += torch.nn.functional.cross_entropy(similarity, similarity_label)

            # similarity = torch.div(similarity, model.tau)
            # reward = reward.view(-1, num_sample)
            # loss += torch.nn.functional.softplus(torch.logsumexp(-reward * similarity, dim=1)).mean()

            # similarity_label = torch.ones_like(reward, device=reward.device).masked_fill_(reward <= 0, 0)
            # similarity_label /= similarity_label.sum(-1, keepdims=True)
            # loss += torch.nn.functional.cross_entropy(similarity, similarity_label)
            # sorted_similarity = similarity.gather(-1, sorted_idx)
            # for i in range(0, num_sample - 1):
            #     similarity_logits = sorted_similarity[:, i:]
            #     similarity_label = sorted_reward[:, i:]
            #     similarity_label = similarity_label-similarity_label[:,-1:]
            #     print(similarity_label)
            #     loss += torch.nn.functional.cross_entropy(similarity_logits, similarity_label)

            #     ranking_loss = torch.max(torch.zeros_like(pos_similarity),
            #                              pos_reward - neg_reward + neg_similarity - pos_similarity)
            #     loss += ranking_loss.sum() / batch_size
            # same_mask = torch.abs(sorted_reward[:, :-i] - sorted_reward[:, i:] > 0.01).float()
            # ones = torch.ones_like(pos_similarity, device=pos_similarity.device)
            # margin_loss = torch.nn.functional.margin_ranking_loss(pos_similarity, neg_similarity, ones,
            #                                                       margin=0.01 * i, reduction="none")
            # if same_mask.sum() > 0:
            #     loss += (margin_loss * same_mask).sum() / batch_size
        #
        return (loss, outputs) if return_outputs else loss

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and args.deepspeed is None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model
        if self.init_tau:
            self.model.init_tau()
        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def training_step(self, model, inputs):
        model.config.count_similarity = False
        return super().training_step(model, inputs)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = 5,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.model.config.max_length
        self._num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.config.count_similarity = self.count_similarity
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        metrics_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_metrics = None
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:

                if labels_host is None:
                    labels_host = labels
                else:
                    labels_host.extend(labels)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=0)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=0)
                if labels_host is not None:
                    labels = labels_host

                    if all_labels is None:
                        all_labels = labels
                    else:
                        all_labels.extend(labels)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, metrics_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=0)
        if labels_host is not None:
            labels = labels_host
            if all_labels is None:
                all_labels = labels
            else:
                all_labels.extend(labels)
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = all_labels[:num_samples]

        all_metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = {k: all_metrics[k].mean().item() for k in all_metrics.keys()}
        metrics = denumpify_detensorize(metrics)
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = 5,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.model.config.max_length
        self._num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
               Perform an evaluation step on `model` using `inputs`.

               Subclass and override to inject custom behavior.

               Args:
                   model (`nn.Module`):
                       The model to evaluate.
                   inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                       The inputs and targets of the model.

                       The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                       argument `labels`. Check your model's documentation for all accepted arguments.
                   prediction_loss_only (`bool`):
                       Whether or not to return the loss only.

               Return:
                   Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
                   labels (each being optional).
               """

        # if prediction_loss_only:
        #     return super().prediction_step(
        #         model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        #     )
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams,
            "use_cache": self.model.config.use_cache,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "do_sample": False,
            "num_return_sequences": 1,
            'is_generate': True,
            "output_hidden_states": False,
            "output_attentions": False
        }
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        outputs = self.model(
            vision_feats=inputs["vision_feats"],
            **gen_kwargs,
        )
        generated_tokens = outputs['sequences']

        # _outputs = self.model(
        #     vision_feats=inputs["vision_feats"],
        #     attention_mask=gen_kwargs["attention_mask"],
        #     decoder_input_ids=generated_tokens,
        #     decoder_attention_mask=generated_tokens != self.model.config.pad_token_id,
        # )
        # v_embeds, t_embeds = _outputs.pooler_output
        # similarity = torch.einsum("ij,ij->i", v_embeds, t_embeds)
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)
        labels = inputs['gts']

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
