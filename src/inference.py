import torch
from transformers import pipeline


model_path = "/mnt/o/genai/models/"

model_id = "/mnt/o/genai/models/gemma-2b-it"
output_dir = "./output"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")
