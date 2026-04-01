#!/usr/bin/env python
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")

def main():
    parser = argparse.ArgumentParser(description="Embed script skeleton")
    parser.add_argument("--model-path", required=True, help="Path to the language model")
    parser.add_argument("--quantization", choices=["4bit", "8bit"], help="Quantization method")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    config = AutoConfig.from_pretrained(args.model_path,
                                        output_hidden_states=True,)
    language_model = AutoModelForCausalLM.from_config(config)

    # TODO: implement embedding logic
    print(f"Loaded model from: {args.model_path}")
    print(f"Embedding input: {args.input}")
    print(f"Writing embeddings to: {args.output}")
    print(f"Batch size: {args.batch_size}")


if __name__ == "__main__":
    main()
