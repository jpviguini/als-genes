import numpy as np
import pandas as pd
import config
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

def get_embeddings(gene_list, model, model_type):
    """
        Gets embeddings for gene list depending on the model
    """

    # convert genes to lowercase for embedding lookup
    lowercase_genes = [g.lower() for g in gene_list]

    if model_type == 'fasttext':
        # filter genes that exist in fasttext vocabulary
        valid_genes = [g for g in lowercase_genes if model.get_word_id(g) != -1]
        embeddings = np.array([model.get_word_vector(g) for g in valid_genes])

        # return original case genes with their embeddings
        original_case_genes = [gene_list[i] for i, g in enumerate(lowercase_genes) if g in valid_genes]
        return embeddings, original_case_genes

    elif model_type == 'word2vec':
        valid_genes = [g for g in lowercase_genes if g in model.wv]
        embeddings = np.array([model.wv[g] for g in valid_genes])

        original_case_genes = [gene_list[i] for i, g in enumerate(lowercase_genes) if g in valid_genes]
        return embeddings, original_case_genes

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

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
    als_word = "als_disease_token"
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



genes_df = pd.read_csv("../data/genes_extracted_validated_general_pmc3.csv")

genes = genes_df["gene"].tolist()

model_path = "./models/general_pmc3/model_ALS_v200_a0p05_n15.model"
model_type = "word2vec"


if os.path.exists(model_path):
    print(f"Loading existing Word2Vec model from {model_path}...")
    model = Word2Vec.load(model_path)


ranking_df = calculate_ranking_dot_product(genes, model, model_type)


print(ranking_df.head(40))

print(model.wv.similarity("sod1", "als_disease_token"))

#print(model.wv.most_similar("als", topn=20))

with open("../data/corpus_als_general_pmc_preprocessed3.csv") as f:
    text = f.read().lower()
print(text.count("als_disease_token"), text.count("amyotrophic lateral sclerosis"))

