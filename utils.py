"""Useful functions for the fine-tuning process of T5 on SuperGLUE benchmark"""
import collections
import re

from evaluate import load
import numpy as np
from peft import (
    PeftModel,
    IA3Config,
    LoraConfig,
    TaskType,
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    PromptEncoderConfig
)
from config import SUPER_GLUE_DATASETS_INFOS

TASK: str | None = None
TOKENIZER = None
DATASET_INFOS = None
MODEL_NAME = None
DATASET_NAME = None

def setup_configs(dataset_name, task, tokenizer, model_checkpoint):
  global TASK, TOKENIZER, DATASET_INFOS
  TASK = task
  DATASET_INFOS = SUPER_GLUE_DATASETS_INFOS[TASK]
  TOKENIZER = tokenizer
  MODEL_NAME = model_checkpoint
  DATASET_NAME = dataset_name

def _mark_span(text, span_str, span_idx, mark):
  pattern_tmpl = r'^((?:\S+\s){N})(W)'
  pattern = re.sub('N', str(span_idx), pattern_tmpl)
  pattern = re.sub('W', span_str, pattern)
  return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

def correct_inputs_targets(samples):
  global TASK, DATASET_INFOS

  new_samples = collections.defaultdict(list)
  keys = samples.keys()
  label_key = DATASET_INFOS.label_key
  for values in zip(*samples.values()):
    num_answers, num_duplicates = 1, 1
    sample = {k: v for k, v in zip(keys, values)}
    sentences = [TASK]
    if TASK == "wsc":
      text = sample["text"]
      text = _mark_span(text, sample["span1_text"], sample["span1_index"], '*')
      span2_index_corrector = 1 if sample['span1_index'] < sample['span2_index'] else 0
      span2_index = sample["span2_index"] + 2 * span2_index_corrector
      text = _mark_span(text, sample["span2_text"], span2_index, '#')
      sentences.append(text)
    else:
      for feature_key in DATASET_INFOS.feature_keys:
        sentences.append(f"{feature_key}:")
        text = sample[feature_key]
        if not isinstance(text, str):
          text = ', '.join(sample[feature_key])
        sentences.append(text)
    sample["input"] = " ".join(sentences)
    if TASK == "record":
      sample["input"] = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', sample["input"])
      sample["input"] = re.sub(r'\n@highlight\n', '. ', sample["input"])
      num_answers = len(sample[label_key])
      num_duplicates = np.maximum(1, num_answers)
    elif TASK == "multirc":
      sample["input"] = re.sub(r"<br>", " ", sample["input"])
      sample["input"] = re.sub(r"<(/)?b>", "", sample["input"])   
    elif TASK == "wsc":
      sample["input"] = re.sub(r"<br>", " ", sample["input"])
      sample["input"] = re.sub(r"<(/)?b>", "", sample["input"])  
    new_samples["input"].extend([sample["input"]] * num_duplicates) 
    original_label = sample[label_key]
    if original_label == -1 or num_answers <= 0:
      new_samples["target"].extend(["<unk>"])
    elif DATASET_INFOS.label_names is not None:
      text_label = DATASET_INFOS.label_names[int(original_label)]
      new_samples["target"].extend([text_label])
    elif isinstance(original_label, list):
      new_samples["target"].extend(sample[label_key])
    else:
      new_samples["target"].extend([original_label])
  return new_samples

def tokenizer_function(samples):
  global TOKENIZER
  model_embeedings = TOKENIZER(samples["input"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
  targets_embeedings = TOKENIZER(samples["target"], max_length=2, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
  targets_embeedings[targets_embeedings == TOKENIZER.pad_token_id] = -100
  model_embeedings["label"] = targets_embeedings
  return model_embeedings

def get_model(model, peft_method: str):
  global MODEL_NAME
  peft_config = None
  match peft_method:
    case "lora":
      peft_config = LoraConfig(
          task_type=TaskType.SEQ_2_SEQ_LM
      )
    case "prompt_tuning":
      peft_config = PromptTuningConfig(
          task_type=TaskType.SEQ_2_SEQ_LM,
          num_virtual_tokens=20,
          prompt_tuning_init=PromptTuningInit.TEXT,
          prompt_tuning_init_text="Answer the yes/no question about the passage:",
          inference_mode=False,
          tokenizer_name_or_path=MODEL_NAME,
      )
    case "prefix_tuning":
      peft_config = PrefixTuningConfig(
          task_type=TaskType.SEQ_2_SEQ_LM,
          inference_mode=False,
          num_virtual_tokens=20
      )
    case "p_tuning":
      peft_config = PromptEncoderConfig(
          task_type=TaskType.SEQ_2_SEQ_LM,
          inference_mode=False,
          num_virtual_tokens=20
      )
    case _:
      peft_config = IA3Config(
          task_type=TaskType.SEQ_2_SEQ_LM
      )
  return get_peft_model(model, peft_config)

def custom_compute_metrics(pred):
    global DATASET_NAME, DATASET_INFOS
    metric = load(DATASET_NAME, DATASET_INFOS.task_name)
    logits = pred.predictions[0]
    labels = pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds[:, 0], references=labels[:, 0])
