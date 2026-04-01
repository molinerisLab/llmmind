#!/usr/bin/env python
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Embed script skeleton")
    parser.add_argument("--model-path", required=True, help="Path to the language model")
    parser.add_argument("--input-dir", required=True, help="Path to the input directory containing txt files to embed")
    parser.add_argument("--quantization", choices=["4bit", "8bit"], help="Quantization method")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    #load txt in input dir in a list of stirngs
    
    stimuli = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(args.input_dir, filename), "r") as f:
                stimuli.append(f.read())

    tokens = tokenizer(stimuli, padding="longest", return_tensors="pt")

    for i, stimulus in enumerate(stimuli):
        #see ../platonic_representation/workflow/scripts/extract_features.py line 56
        with torch.no_grad():
            llm_output = model(
                input_ids=tokens['input_ids'][i],
                attention_mask=tokens['attention_mask'][i],
            )
        print(f"Stimulus {i} embedding: {llm_output.last_hidden_state.mean(dim=1)}")

if __name__ == "__main__":
    main()
