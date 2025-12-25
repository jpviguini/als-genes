import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA

def load_embeddings(model_path, model_type):
    """
    load embeddings from word2vec, fastText or BERT (numpy)
    returns: X (vectors), words (list), counts (frequencies)
    """

    if model_type == "word2vec":
        model = Word2Vec.load(model_path)
        words = list(model.wv.index_to_key)
        X = np.array([model.wv[w] for w in words])
        counts = np.array([model.wv.get_vecattr(w, "count") for w in words])

    elif model_type == "fasttext":
        model = FastText.load(model_path)
        words = list(model.wv.index_to_key)
        X = np.array([model.wv[w] for w in words])
        counts = np.array([model.wv.get_vecattr(w, "count") for w in words])

    elif model_type == "npz": 
        print("Loading from NPZ...")
        data = np.load(model_path)
        words = data['words']
        X = data['embeddings']
        counts = data['counts']
        
    else:
        raise ValueError("model_type must be 'word2vec', 'fasttext' or 'npz'")

    return X, words, counts

def pca_whitening(X, eps=1e-5):
    # center
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # covariance
    cov = np.cov(Xc, rowvar=False)
    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # numerical stability
    eigvals = np.maximum(eigvals, eps)
    # whitening matrix
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    Xw = Xc @ W.T
    return Xw, mean, W


def plot_pca_before_after(X_before, X_after, counts, output_path):
    pca = PCA(n_components=2)

    print("Fitting PCA...")
    Xb_2d = pca.fit_transform(X_before)
    Xa_2d = pca.fit_transform(X_after)

    # log normalization (to visualize)
    colors = np.log10(counts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # plot before
    sc1 = axes[0].scatter(Xb_2d[:, 0], Xb_2d[:, 1], 
                          c=colors, cmap='viridis', s=3, alpha=0.6)
    axes[0].set_title("Before Whitening (PCA 2D)")
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Log10(Frequency)')

    # plot after
    sc2 = axes[1].scatter(Xa_2d[:, 0], Xa_2d[:, 1], 
                          c=colors, cmap='viridis', s=3, alpha=0.6)
    axes[1].set_title("After Whitening (PCA 2D)")
    cbar2 = fig.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Log10(Frequency)')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    
    
    model_path = "./scores_biobert/scores_top10_ALS_1970_2026.npz"
    model_type = "npz" 

    output_dir = "./whitened_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading embeddings from {model_path}...")
    X, words, counts = load_embeddings(model_path, model_type)

    print(f"Vocabulary size: {X.shape[0]}")
    print(f"Embedding dim: {X.shape[1]}")
    print(f"Freq range: Min={counts.min()}, Max={counts.max()}")

    print("Applying PCA whitening...")
    X_whitened, mean, W = pca_whitening(X)

    print("Saving whitened embeddings...")
    
    np.savez(
        os.path.join(output_dir, "whitened_biobert_2026.npz"),
        embeddings=X_whitened,
        words=np.array(words),
        counts=counts,
        mean=mean,
        whitening_matrix=W
    )

    print("Generating PCA visualization...")
    plot_pca_before_after(
        X,
        X_whitened,
        counts,
        os.path.join(output_dir, "pca_whitened_biobert_2026.png")
    )

    print("Done.")