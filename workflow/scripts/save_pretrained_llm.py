#!/usr/bin/env python
import sys
from transformers import AutoModel, AutoTokenizer

def save_pretrained_model(model_name, save_dir):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python workflow/scripts/save_pretrained_llm.py <model_name> <save_dir>")
        sys.exit(1)
    model_name = snakemake.wildcards.model_name # sys.argv[1]
    save_dir = snakemake.output[0]
    save_pretrained_model(model_name, save_dir)