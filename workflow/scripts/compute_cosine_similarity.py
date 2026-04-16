import numpy as np
import pandas as pd

import argparse

# define a function that normalise a vector wrt the l2 norm
def normalize_l2(x):
    x = np.array(x) # convert input to NumPy array
    # check if x is a one-dimensional array
    if x.ndim == 1:
        norm = np.linalg.norm(x) # compute the l2 norm of the vector
        if norm == 0:
            return x
        return x/norm
    # if x is a higher-dimensional array, compute the l2 norm along the columns
    norm = np.linalg.norm(x, 2, axis = 1, keepdims = True)
    return np.where(norm == 0, x, x/norm) # short-hand for what has been done for one-dimnensional arrays

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dataframe", 
                      type = str, 
                      help = "Path to dataframe of embeddings")
    parser.add_argument("--cosine_similarity", 
                      type = str, 
                      help = "Path to dataframe of computed cosine similarities")
    args = parser.parse_args()

    embedding_dataframe = args.embedding_dataframe
    cosine_similarity = args.cosine_similarity

    # load the dataframe
    embedding_df = pd.read_parquet(embedding_dataframe, engine = "pyarrow")
    
    # compute cosine similarities for all the concepts in the embedding dataframe
    X = normalize_l2(embedding_df["embedding"].tolist())
    result = X @ X.T

    # remove cosine self-similarities
    np.fill_diagonal(result, -np.inf)

    result_df = pd.DataFrame(
        result, 
        index = embedding_df.index, 
        columns = embedding_df.index
    )

    # save the pandas dataframe as a parquet file
    result_df.to_parquet(cosine_similarity, engine = "pyarrow", index = True)

if __name__ == "__main__":
    main()