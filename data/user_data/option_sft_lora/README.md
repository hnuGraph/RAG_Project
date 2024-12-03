---
library_name: peft
license: other
base_model: /media/hnu/LLM/qwen2.5-7B/
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft_dev_option_round2_top5_train_and_valid
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_dev_option_round2_top5_train_and_valid

This model is a fine-tuned version of [/media/hnu/LLM/qwen2.5-7B/](https://huggingface.co//media/hnu/LLM/qwen2.5-7B/) on the train_option dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.45.2
- Pytorch 2.4.0+cu121
- Datasets 2.19.0
- Tokenizers 0.20.1