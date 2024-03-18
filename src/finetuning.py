from pynvml import *
import torch
from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer,TrainingArguments
from datasets import load_dataset
import transformers
from trl import SFTTrainer
from peft import LoraConfig


# for metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss
from torch.nn import CrossEntropyLoss
import numpy as np
from copy import deepcopy
from transformers import TrainerCallback


model_id = "/mnt/o/genai/models/gemma-7b"
output_dir = "./output/gemma7b-monitoring"
train_data = "/mnt/o/genai/datasets/Abirate_english_quotes"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()    


print_gpu_utilization()


# load model with quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True
                                )

model = AutoModelForCausalLM.from_pretrained(model_id,  device_map="auto",quantization_config=bnb_config)
print_gpu_utilization()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# test prompt before finetuning

# text = "Quote: Imagination is more"
text = "Quote: Two things are infinite"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print_gpu_utilization()
# prepare dataset
data = load_dataset(train_data)
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


# define metric collectors
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train@en")
            return control_copy

def compute_metrics(pred):
    global num_labels
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    loss_fct = CrossEntropyLoss()
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(labels)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return {
        'accuracy@en': acc,
        'f1@en': f1,
        'precision@en': precision,
        'recall@en': recall,
        'loss@en': loss,
    }

#train

def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"Quote: {example['quote'][i]}\nAuthor: {example['author'][i]}"
        output_texts.append(text)
    return output_texts

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,    
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    overwrite_output_dir=True,
    save_total_limit=1,

)


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_func,
    compute_metrics=compute_metrics
    
)
result =  trainer.train()

