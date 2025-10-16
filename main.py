import pandas as pd
import requests
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
from collections import defaultdict
import os
import ast
from gensim.models import Word2Vec
from io import StringIO




from src import config
from src import data_loader
from src import text_processing
from src import modeling
from src import ranking
from src import evaluation 


def load_or_fetch_data(csv_path):
    """
        Loads a dataframe if it exists, or do the search in Pubmed
    """
    df = pd.DataFrame()
    
    response = requests.get(csv_path)


    if response.status_code == 200: # os.path.exists() only works with local files
        
        try:
            df = pd.read_csv(StringIO(response.text))
            if not df.empty and isinstance(df['genes'].iloc[0], str):
                 df['genes'] = df['genes'].apply(lambda x: eval(x))

                # convert string to list
                #  df['genes'] = df['genes'].apply(ast.literal_eval)

                # get only articles that contain genes --> filtering articles with no genes
                 df = df[df['genes'].apply(len) > 0]

            print(f"Existing data loaded from {csv_path} ({len(df)} articles).")
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print("Error: the dataframe is empty or this is another unknown error")
            return df # returns empty df
    else:
        print("Existing file not found in google drive. Doing the search from scratch...\n")

        print("Fetching new articles from PubMed...")
        articles_2000_2001 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2000, 2001, max_articles=10000)
        articles_2001_2002 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2001, 2002, max_articles=10000)
        articles_2002_2003 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2002, 2003, max_articles=10000)
        articles_2003_2004 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2003, 2004, max_articles=10000)
        articles_2004_2005 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2004, 2005, max_articles=10000)
        articles_2005_2006 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2005, 2006, max_articles=10000)
        articles_2006_2007 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2006, 2007, max_articles=10000)
        articles_2007_2008 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2007, 2008, max_articles=10000)
        articles_2008_2009 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2008, 2009, max_articles=10000)
        articles_2009_2010 = data_loader.get_als_genetic_articles(config.BASE_QUERY, 2009, 2010, max_articles=10000)

        df = pd.DataFrame(
            articles_2000_2001 + articles_2001_2002 + articles_2002_2003 + articles_2003_2004 +
            articles_2004_2005 + articles_2005_2006 + articles_2006_2007 + articles_2007_2008 +
            articles_2008_2009 + articles_2009_2010
        )
        #df.to_csv("pubmed_articles.csv", index=False)
        df.to_csv(csv_path, index=False)
        print("Total collected:", len(df))
        return df

def process_and_extract_genes(df, csv_path):
    """
    Realiza a limpeza do texto e a extração/validação de genes.
    """
    
    extractor = text_processing.GeneExtractor()

    if 'clean_text' not in df.columns:
        df['text'] = df['text'].fillna('')
        df['clean_text'] = df['text'].apply(text_processing.clean_text)
    if 'genes' not in df.columns:
        df['genes'] = df['text'].apply(extractor.extract_genes_unbiased)

        if config.VALIDATE_WITH_MYGENE:
            valid_genes = extractor.validate_genes_with_mygene(set(g for gl in df['genes'] for g in gl))
            df['genes'] = df['genes'].apply(lambda genes: [g for g in genes if g in valid_genes])

    df.to_csv(csv_path, index=False)
    print(f"Processed data saved to '{csv_path}'.")
    return df

def prepare_embedding_model(df):
    """
    Carrega ou treina o modelo de embedding especificado no config.
    """
    embedding_model = None

    if config.MODEL_CHOICE == 'fasttext':
        print("Initializing fastText...")
        corpus_filepath = f"fasttext_corpus_{config.YEAR_END}.txt"
        #model_filepath = f"./drive/MyDrive/IC_2025/fasttext_model_2010_3173_biased_updated.bin"
        
        model_filepath = "./models/fasttext_model_2010_3173_biased_updated.bin"
        
        if not os.path.exists(corpus_filepath): 
            modeling.save_corpus_for_fasttext(df, filepath=corpus_filepath)
                                                                                            # model_path = model_filepath
        embedding_model = modeling.get_fasttext_model(corpus_path=corpus_filepath, model_path=model_filepath) 


     
    elif config.MODEL_CHOICE == 'word2vec':
        print("Initializing word2vec...")
        corpus_filepath = "fasttext_corpus.txt" # change for a general txt
        model_filepath = f"./drive/MyDrive/IC_2025/word2vec_model_2010_3173_biased_updated.bin"

        if not os.path.exists(corpus_filepath):
            modeling.save_corpus_for_fasttext(df, filepath=corpus_filepath)

        embedding_model = modeling.get_word2vec_model(df, corpus_path=corpus_filepath, model_path=model_filepath)

    return embedding_model


# Metrics
def evaluate_ranking(ranked_genes_df, validation_genes):

    y_true = ranked_genes_df['gene'].str.upper().isin(validation_genes).astype(int)
    y_score = ranked_genes_df['sim_raw_norm']
    
    map_score = average_precision_score(y_true, y_score)
    ndcg = ndcg_score([y_true], [y_score])
    
    precision_at_10 = y_true[:10].sum() / 10
    precision_at_50 = y_true[:50].sum() / 50
    
    return {
        "MAP": map_score,
        "NDCG": ndcg,
        "Precision@10": precision_at_10,
        "Precision@50": precision_at_50
    }



def rank_and_display_results(df, embedding_model):
    """
    Realiza o ranking dos genes e exibe/salva os resultados.
    """
    all_genes = list({g.upper() for gene_list in df['genes'] for g in gene_list})

    ranked_genes_full = ranking.calculate_ranking_cosine(
        genes=all_genes,
        model=embedding_model,
        model_type=config.MODEL_CHOICE,
        known_als_genes=config.VALIDATION_GENES
    )

    ranked_dot_product = ranking.calculate_ranking_dot_product(
        genes=all_genes,
        model=embedding_model,
        model_type=config.MODEL_CHOICE
    )
    
    metrics = evaluate_ranking(ranked_genes_full, config.VALIDATION_GENES)
    print(metrics)

    # # metrics calculation (on list with known genes)
    # print("\nCalculating performance metrics...")
    # metrics = evaluation.calculate_all_metrics(ranked_genes_full, config.VALIDATION_GENES)
    # metrics_df = pd.DataFrame([metrics], index=[config.MODEL_CHOICE.capitalize()])

    # filtering for novel candidates
    ranked_novel_genes = ranked_genes_full[~ranked_genes_full['gene'].isin(config.VALIDATION_GENES)]

    # to make temporal analysis --> use ranked_genes_full
    # to discover new candidate genes --> use ranked_novel_genes
    if not ranked_novel_genes.empty:
        print(f"\n--- TOP 20 NOVEL CANDIDATES (Model: {config.MODEL_CHOICE.upper()}) ---")
        for i, row in enumerate(ranked_novel_genes.head(20).itertuples(), 1):
            print(f"{i}. {row.gene.upper():<10} | Sim: {row.sim_raw_norm:.4f})")

        output_filename = f'als_novel_gene_candidates_{config.MODEL_CHOICE}.csv'
        ranked_novel_genes.to_csv(output_filename, index=False)
        print(f"\nResults with novel genes saved to '{output_filename}'")


    if not ranked_dot_product.empty:
        print(f"\n--- TOP 20 GENES RANKED BY DOT PRODUCT WITH 'ALS' (Model: {config.MODEL_CHOICE.upper()}) ---")
        for i, row in enumerate(ranked_dot_product.head(20).itertuples(), 1):
            print(f"{i}. {row.gene:<10} | Dot with ALS: {row.dot_with_als:.4f}")

        ranked_dot_validation = ranked_dot_product[
          ranked_dot_product['gene'].apply(lambda g: str(g).upper() in config.VALIDATION_GENES)
        ]

        if not ranked_dot_product.empty:
            print(f"\n--- TOP known genes GENES RANKED BY DOT PRODUCT WITH 'ALS' (Model: {config.MODEL_CHOICE.upper()}) ---")
            for i, row in enumerate(ranked_dot_product.itertuples(), 1):
                if row.gene.upper() in config.VALIDATION_GENES:
                    print(f"{i}. {row.gene:<10} | Dot with ALS: {row.dot_with_als:.4f}")
        else:
            print("\nNo validation genes were found.")


    # print("\n--- Performance Metrics Table ---")
    # print(metrics_df.round(4))






def main_pipeline():

    print("Starting pipeline...\n")
    
    # data loading and processing


    #csv_path = f"./drive/MyDrive/IC_2025/als_articles_2010_3173_biased_updated.csv"
    csv_path = "https://drive.google.com/uc?id=1qQYVG0KKD6J8OEVLCDElOvwn7V6fyR-a"
    df = load_or_fetch_data(csv_path)

    # extracting and validating genes
    if not df.empty:
        df = process_and_extract_genes(df, csv_path)

        # loading/training the model
        embedding_model = prepare_embedding_model(df)
        
        # ranking
        if embedding_model:
            rank_and_display_results(df, embedding_model)
    else:
        print("DataFrame is empty. Exiting pipeline.")





if __name__ == "__main__":
    main_pipeline()
