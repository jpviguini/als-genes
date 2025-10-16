import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src.modeling import get_embeddings



def calculate_ranking_cosine(genes, model, model_type: str, known_als_genes=None):



    # default known ALS genes if none provided
    if known_als_genes is None:
        known_als_genes = {
            "ANXA11", "C9ORF72", "CHCHD10", "EPHA4", "FUS", "HNRNPA1", "KIF5A", "NEK1",
            "OPTN", "PFN1", "SOD1", "TARDBP", "TDP-43", "TDP43", "TBK1", "UBQLN2",
            "UNC13A", "VAPB", "VCP"
        }

    known_als_genes = {g.upper() for g in known_als_genes}

    # get embeddings for known ALS genes
    known_embeddings, valid_known_genes = get_embeddings(
        list(known_als_genes), model, model_type
    )

    if len(valid_known_genes) == 0:
        print("No known genes found in model.")
        return pd.DataFrame()

    known_embeddings = normalize(known_embeddings, axis=1) # normalize it

    # get embeddings for candidate genes
    #candidates = [gene.upper() for gene in gene_score_dict]
    candidates = [gene.upper() for gene in genes]

    candidate_embeddings, valid_candidates = get_embeddings(
        candidates, model, model_type
    )

    if len(valid_candidates) == 0:
        print("No candidate embeddings generated.")
        return pd.DataFrame()

    candidate_embeddings = normalize(candidate_embeddings, axis=1) # normalize it

    # cosine similarity between candidates and known genes
    similarity_matrix = cosine_similarity(candidate_embeddings, known_embeddings)


    # # here we are preventing a known gene to be compared to itself (if it's also a candidate)
    # known_gene_to_idx = {gene.upper(): i for i, gene in enumerate(valid_known_genes)}


    # for i, candidate_gene in enumerate(valid_candidates):

    #     if candidate_gene.upper() in known_gene_to_idx:

    #         j = known_gene_to_idx[candidate_gene.upper()]

    #         similarity_matrix[i, j] = 0.0 # if the candidate is a known gene, we prevent it from comparing to itself


    max_similarities = np.max(similarity_matrix, axis=1) # take the MAX score

    # create results dataframe with combined scores
    results_df = pd.DataFrame({
        'gene': [g.upper() for g in valid_candidates],
        'sim_raw': max_similarities
    })


    # normalize scores before combining them with alpha (just testing)
    for col in ['sim_raw']:
        min_val, max_val = results_df[col].min(), results_df[col].max()

        results_df[f'{col}_norm'] = (results_df[col] - min_val) / (max_val - min_val + 1e-9)

    return results_df.sort_values('sim_raw', ascending=False)



def calculate_ranking_dot_product(genes, model, model_type: str):


    # normalize gene names
    candidates = [g.lower() for g in genes]

    # get candidate embeddings
    candidate_embeddings, valid_candidates = get_embeddings(candidates, model, model_type)
    if len(valid_candidates) == 0:
        print("No candidate gene found in model.")
        return pd.DataFrame()

    candidate_embeddings = normalize(candidate_embeddings, axis=1)

    # "ALS" embedding
    als_word = "als"
    if model_type == 'fasttext':
        if model.get_word_id(als_word) == -1:
            print("'ALS' not found in fastText vocab")
            return pd.DataFrame()
        als_embedding = model.get_word_vector(als_word)

    elif model_type == 'word2vec':
        if als_word not in model.wv:
            print("'ALS' not found in word2vec vocab")
            return pd.DataFrame()
        als_embedding = model.wv[als_word]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    als_embedding = als_embedding / np.linalg.norm(als_embedding)  # normalize

    # calculate dot product with ALS
    dot_scores = candidate_embeddings @ als_embedding

    # returns the results in a df
    results_df = pd.DataFrame({
        'gene': [g.upper() for g in valid_candidates],
        'dot_with_als': dot_scores
    })

    results_df = results_df.sort_values('dot_with_als', ascending=False).reset_index(drop=True)

    return results_df


