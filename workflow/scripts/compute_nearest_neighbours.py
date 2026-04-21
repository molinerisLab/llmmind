import numpy as np
import pandas as pd

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_neighbours", 
                      type = int, 
                      help = "Set the number of neighbours to compute")
    parser.add_argument("--cosine_similarity", 
                      type = str, 
                      help = "Path to the file containing computed cosine similarities")
    parser.add_argument("--nearest_neighbours", 
                      type = str, 
                      help = "Path to the file containing the nearest neighbours of the primary concepts")
    args = parser.parse_args()

    primary_concept = args.primary_concept
    number_of_neighbours = args.number_of_neighbours
    cosine_similarity = args.cosine_similarity
    nearest_neighbours = args.nearest_neighbours

    # load the dataframe
    cosine_similarity_df = pd.read_parquet(cosine_similarity, engine = "pyarrow")

    X = cosine_similarity_df.values
    concepts = cosine_similarity_df.index.to_numpy()
    neighbours = cosine_similarity_df.columns.to_numpy()
    k_eff = min(number_of_neighbours, X.shape[1] - 1)

    # row by row, partition it storing the indices of largest values in the last positions, then keep only those
    idx_part = np.argpartition(X, -k_eff, axis = 1)[:, -k_eff:]
    # row by row, take the values corresponding to the indices selected above
    scores_part = np.take_along_axis(X, idx_part, axis = 1)
    
    # row by row, returns the indices that would sort that row in ascending order, then reverse it
    order = np.argsort(scores_part, axis = 1)[:, ::-1]
    # row by row, take the indices according to the order selected above
    idx_topk = np.take_along_axis(idx_part, order, axis = 1)
    # row by row, take the values according to the indices selected above
    scores_topk = np.take_along_axis(X, idx_topk, axis = 1)

    # create a pandas dataframe to store the nearest neighbours of each concept
    nearest_neighbours_df = pd.DataFrame({
    "concept": np.repeat(concepts, k_eff), 
    "neighbour": neighbours[idx_topk.reshape(-1)], 
        "cosine_similarity": scores_topk.reshape(-1)
    })
    
    # print dataframe of nearest neighbours
    print(nearest_neighbours_df.to_string())

    # save the nearest neighbours to primary concept as a parquet file
    nearest_neighbours_df.to_parquet(nearest_neighbours, engine = "pyarrow", index = True)

if __name__ == "__main__":
    main()