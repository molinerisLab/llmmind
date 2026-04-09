#!/usr/bin/env python
import os
import pandas as pd
import argparse

import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Embed script skeleton")
    parser.add_argument("--input_dir", 
                        required=True, 
                        help="Path to the input directory containing txt files to embed"
                        )
    parser.add_argument("--output",
                        required=True,
                        help="Path to the output file where embeddings will be saved"
                        )
    parser.add_argument("--model_path", 
                        required=True, 
                        help="Path to the language model"
                        )
    parser.add_argument("--quantization_method", 
                        choices=["4bit", "8bit"], 
                        help="Quantization method"
                        )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = None
    if args.quantization_method == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif args.quantization_method == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None
    )
    
    # load txt in input dir in a list of stirngs
    stimuli = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(args.input_dir, filename), "r") as f:
                stimuli.append(f.read())

    tokens = tokenizer(
        stimuli,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    records = []
    for i, stimulus in enumerate(stimuli):
        with torch.no_grad():
            llm_output = model(
                input_ids=tokens['input_ids'][i].unsqueeze(0),
                attention_mask=tokens['attention_mask'][i].unsqueeze(0),
            )

        embedding = llm_output.last_hidden_state.mean(dim=1).squeeze(0).cpu().tolist()

        records.append({
            "stimulus": stimulus,
            "embedding": embedding,
        })

    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)

if __name__ == "__main__":
    main()