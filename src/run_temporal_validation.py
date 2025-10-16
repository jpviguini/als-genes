import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt


GENE_TO_QUERY = "KIF5A"
MODEL_TO_USE = 'fasttext'

VALIDATION_GENES = {
    "ANXA11", "C9ORF72", "CHCHD10", "EPHA4", "FUS", "HNRNPA1", "KIF5A", "NEK1",
    "OPTN", "PFN1", "SOD1", "TARDBP", "TBK1", "UBQLN2",
    "UNC13A", "VAPB", "VCP"
}
VALIDATION_GENES = {g.upper() for g in VALIDATION_GENES}


print(f"Loading the model... '{MODEL_TO_USE}'...")
embedding_model = None


if MODEL_TO_USE == 'fasttext':
    model_filepath = f"./drive/MyDrive/IC_2025/fasttext_model_{YEAR_END}.bin"
    embedding_model = fasttext.load_model(model_filepath)

df_full = pd.read_csv('als_articles_with_genes_2010.csv')
df_full['genes'] = df_full['genes'].apply(lambda x: eval(x))
all_genes_in_corpus = set(g.upper() for gene_list in df_full['genes'] for g in gene_list)



def find_similar_genes_within_set(input_gene, model, model_type, comparison_set):

    print(f"\nCalculating similarity of '{input_gene.upper()}' with reference set...")
    input_gene_upper = input_gene.upper()
    comparison_set_upper = {g.upper() for g in comparison_set}
    comparison_set_upper.discard(input_gene_upper)
    genes_to_embed = [input_gene_upper] + list(comparison_set_upper)
    all_embeddings, valid_genes_list = get_embeddings(genes_to_embed, model, model_type)

    if input_gene_upper not in valid_genes_list:
        print(f"ERROR: Candidate gene '{input_gene_upper}' was not found in the model's vocabulary.")
        return None

    all_embeddings = normalize(all_embeddings)
    input_embedding = all_embeddings[0].reshape(1, -1)
    comparison_embeddings = all_embeddings[1:]
    comparison_genes = valid_genes_list[1:]
    similarities = cosine_similarity(input_embedding, comparison_embeddings)
    results = sorted(zip(comparison_genes, similarities[0]), key=lambda item: item[1], reverse=True)
    print(f"\n--- Similarity ranking with '{input_gene_upper}' (Model: {model_type.capitalize()}) ---")
    for gene, score in results:
        print(f"{gene:<12} | Similarity: {score:.4f}")
    return results


if embedding_model:
    if GENE_TO_QUERY.upper() in all_genes_in_corpus:

        # reference set should be both in validation genes and in model's vocab
        final_comparison_set = VALIDATION_GENES.intersection(all_genes_in_corpus)

        print(f"\nIniciando análise. Comparando com {len(final_comparison_set)} genes validados que estão presentes no corpus.")

        similarity_results = find_similar_genes_within_set(GENE_TO_QUERY, embedding_model, MODEL_TO_USE, final_comparison_set)


        if similarity_results:

            top_10_results = similarity_results[:10]
            df_plot = pd.DataFrame(top_10_results, columns=['gene', 'similarity'])
            df_plot = df_plot.sort_values('similarity', ascending=True)
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(10, 7), dpi=120)
            plt.barh(df_plot['gene'], df_plot['similarity'], color='skyblue', edgecolor='black')
            plt.xlabel('Cosine Similarity', fontsize=12)
            plt.title(f'Top Most Similar ALS Genes to "{GENE_TO_QUERY.upper()}"', fontsize=16, weight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'similarity_plot_{GENE_TO_QUERY.upper()}.png', dpi=300)
            plt.show()

    else:
        print(f"\nERROR: Gene '{GENE_TO_QUERY.upper()}' not found on the list of genes of the corpus..")
else:
    print(f"Model '{MODEL_TO_USE}' could not be loaded.")