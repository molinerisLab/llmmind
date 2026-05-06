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
    norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
    return np.where(norm == 0, x, x/norm) # short-hand for what has been done for one-dimnensional arrays

def pearson_normalize(x):
    """
    Row-wise Pearson normalization.

    Pearson correlation between two vectors is equivalent to cosine similarity
    after subtracting each vector's mean.
    """
    x = np.asarray(x, dtype=np.float64)

    # mean-center each embedding vector
    x = x - np.mean(x, axis=1, keepdims=True)

    # l2-normalize each centered vector
    x = normalize_l2(x)

    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dataframe", 
                      type = str, 
                      help = "Path to dataframe of embeddings")
    parser.add_argument("--similarity_dataframe", 
                      type = str, 
                      help = "Path to dataframe of computed similarities")
    args = parser.parse_args()

    embedding_dataframe = args.embedding_dataframe
    similarity_dataframe = args.similarity_dataframe

    # load the dataframe
    embedding_df = pd.read_parquet(embedding_dataframe, engine="pyarrow")
    
#    # compute cosine similarities for all the concepts in the embedding dataframe
#    X = normalize_l2(embedding_df.values)
    # compute Pearson similarities for all the concepts in the embedding dataframe
    X = pearson_normalize(embedding_df.values)
    result = X @ X.T

    # remove self-similarities
    np.fill_diagonal(result, -np.inf)

    result_df = pd.DataFrame(
        result, 
        index = embedding_df.index, 
        columns = embedding_df.index
    )
    result_df = result_df.sort_index()

    # save the pandas dataframe as a parquet file
    result_df.to_parquet(similarity_dataframe, engine="pyarrow", index=True)
    
#    # print the alignment score dataframe
#    with pd.option_context("display.max_rows", None, "display.max_columns", None):
#        print(result_df)

if __name__ == "__main__":
    main()