#!/usr/bin/env python
import os
#import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModel

## define a function that get the embedding for a given text, cache the results on a json file, and avoid redundant calls to the API
#def get_embedding(embedding_cache, embedding_cache_file, text, client, model = "text-embedding-3-small"):
#    # check if the embedding of 'text' wrt the chosen model is already cached
#    if text in embedding_cache and embedding_cache[text].get('model') == model:
#        return embedding_cache[text]['embedding']
#    # if not cached, call the OpenAI API to get the embedding
#    response = client.embeddings.create(
#        input = text,
#        model = model
#    )
#    embedding = response.data[0].embedding
#    # cache the embedding with model info
#    embedding_cache[text] = {'embedding': embedding, 'model': model}
#    # save the updated cache to the json file
#    with open(embedding_cache_file, "w") as f: # open the file in "w"rite mode
#        json.dump(embedding_cache, f)          # convert the python object into a json string and write it into the file
#    return embedding                           # return the embedding "vector" for 'text'

def main():
    parser = argparse.ArgumentParser(description="Embed script skeleton")
    parser.add_argument("--model-path", 
                        required=True, 
                        help="Path to the language model"
                        )
    parser.add_argument("--input-dir", 
                        required=True, 
                        help="Path to the input directory containing txt files to embed"
                        )
    parser.add_argument("--quantization", 
                        choices=["4bit", "8bit"], 
                        help="Quantization method"
                        )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_path)
    
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

    for i in range(len(stimuli)):
        with torch.no_grad():
            llm_output = model(
                input_ids=tokens['input_ids'][i].unsqueeze(0),
                attention_mask=tokens['attention_mask'][i].unsqueeze(0),
            )
        # pooling strategy: mean pooling along sequence dimension
        print(f"Stimulus {i} embedding: {llm_output.last_hidden_state.mean(dim=1)}")

if __name__ == "__main__":
    main()