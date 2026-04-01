#!/usr/bin/env python
import sys  
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModel, AutoTokenizer


def save_pretrained_model(model_name, save_dir):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python save_pretrained.py <model_name> <save_dir>")
        sys.exit(1)
    model_name = sys.argv[1]
    save_dir = sys.argv[2]
    save_pretrained_model(model_name, save_dir)

    config = AutoConfig.from_pretrained(llm_model_path,
                                            device_map="auto",
                                            quantization_config=quantization_config,
                                            torch_dtype=torch_dtype,
                                            force_download=force_download,
                                            output_hidden_states=True,)
        language_model = AutoModelForCausalLM.from_config(config)
        language_model = language_model.to(torch_dtype)
        language_model = language_model.to("cuda" if torch.cuda.is_available() else "cpu")
        language_model = language_model.eval()
