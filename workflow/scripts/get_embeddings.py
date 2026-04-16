import os
import re
import pandas as pd
import argparse

import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Embed script skeleton")
    parser.add_argument("--input_dir",
                        required=True,
                        help="Path to the input directory containing txt files to embed")
    parser.add_argument("--output",
                        required=True,
                        help="Path to the output file where embeddings will be saved")
    parser.add_argument("--model_path",
                        required=True,
                        help="Path to the language model")
    parser.add_argument("--quantization_method",
                        choices=["4bit", "8bit"],
                        help="Quantization method")
    parser.add_argument("--excluded_stimuli",
                        nargs="*",
                        default=[],
                        help="List of stimuli to exclude from embedding")
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
    model.eval()

    # Determine device for non-quantized models.
    # For quantized models with device_map="auto", inputs can usually stay on the
    # device of the embedding layer / first parameter.
    device = next(model.parameters()).device

    # Load txt files and keep both task name and stimulus text
    items = []
    pattern = re.compile(r"^(.+)_transcript\.txt$")

    for filename in sorted(os.listdir(args.input_dir)):
        match = pattern.match(filename)
        if not match:
            continue

        task = match.group(1)
        if task in args.excluded_stimuli:
            continue

        filepath = os.path.join(args.input_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            stimulus = f.read()

        items.append({
            "task": task,
            "stimulus": stimulus,
        })

    records = []
    with torch.no_grad():
        for item in items:
            # Tokenize one stimulus at a time on CPU
            tokens = tokenizer(
                item["stimulus"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Move only this stimulus to GPU
            tokens = {k: v.to(device) for k, v in tokens.items()}

            llm_output = model(**tokens)

            embedding = (
                llm_output.last_hidden_state
                .mean(dim=1)
                .squeeze(0)
                .cpu()
                .tolist()
            )

            records.append({
                "task": item["task"],
                "embedding": embedding,
            })

    df = pd.DataFrame(records)
    df = df.set_index("task")
    print(df)
    df.to_parquet(args.output, engine="pyarrow", index=True)

if __name__ == "__main__":
    main()