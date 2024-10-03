from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#Set pad_token_id
tokenizer.pad_token = tokenizer.eos_token #Set pad token to be equal to the eos token
model.config.pad_token_id = tokenizer.pad_token_id