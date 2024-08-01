import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
from transformers import default_data_collator
import collections
from tqdm.auto import tqdm
import numpy as np
model_checkpoint = "Distill_bert_Peft_1"
max_length = 512
doc_stride  = 128









def get_answer(data):
    n_best_size = 20
    data = dict()
    
