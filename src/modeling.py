import os
import fasttext
import numpy as np
import time
from src import config
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from torch.utils.tensorboard import SummaryWriter # teste


def get_embeddings(gene_list, model, model_type):

    """
      gets embeddings for gene list depending on the model
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

# txt for fasttext
def save_corpus_for_fasttext(df, filepath="fasttext_corpus.txt"):
    with open(filepath, "w", encoding="utf-8") as f:
        for text in df['clean_text']:
            f.write(text + "\n")

    print(f"Corpus saved to {filepath}")



def get_word2vec_model(df, corpus_path="word2vec_corpus.txt", model_path=f"word2vec_model{config.YEAR_END}.bin"):

    if model_path.status_code == 200:
        print(f"Loading existing Word2Vec model from {model_path}...")
        model = Word2Vec.load(model_path)

    else:
        print("Training new Word2Vec model...")

        # Build sentences for training
        sentences = [simple_preprocess(text) for text in df['clean_text']]

        model = Word2Vec(
            sentences=sentences,
            vector_size=300,  # embedding dimension
            window=5,         # context window
            min_count=2,      # ignore words that appear < 2 times
            workers=4,
            sg=1              # skip-gram
        )

        model.save(model_path)
        print(f"Word2Vec model trained and saved to {model_path}")

    return model


# train or load fastText model
def get_fasttext_model(corpus_path="fasttext_corpus.txt", model_path=f"fasttext_model_{config.YEAR_END}.bin", force_retrain=False):
    
    if not force_retrain and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        return fasttext.load_model(model_path)

    print("Training new FastText model...")
    model = fasttext.train_unsupervised(
        corpus_path,
        model='skipgram',
        dim=300,
        epoch=10,
        minn=3,
        maxn=6,
        ws=10,
        lr=0.05,
        minCount=2, # adjust to 2
        thread=4)
    
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    return model

