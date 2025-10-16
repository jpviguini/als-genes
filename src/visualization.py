
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_separate_score_charts(model_choice: str, num_genes: int = 20):
    """
      Loads model results and generates two separate bar charts for the TF-IDF
      and Semantic Similarity scores of the top N candidate genes.

    """

    # loads data
    results_filename = f'als_novel_gene_candidates_{model_choice}.csv'
    try:
        ranked_genes_df = pd.read_csv(results_filename)
        print(f"Loaded data from '{results_filename}'")
    except FileNotFoundError:
        print(f"Error: The results file '{results_filename}' was not found.")
        print("Please run the main pipeline cell first to generate the results.")
        return

    # select the top N genes based on the overall combined score
    top_genes = ranked_genes_df.head(num_genes)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(
        1, 2,  # 1 row, 2 columns
        figsize=(20, 10),
        dpi=150
    )


    # sort data by TF-IDF score for this plot
    tfidf_sorted = top_genes.sort_values('tfidf_raw_norm', ascending=True)

    axes[0].barh(
        tfidf_sorted['gene'].str.upper(),
        tfidf_sorted['tfidf_raw_norm'],
        color='skyblue'
    )
    axes[0].set_title('Top 20 Gene Candidates by TF-IDF Score', fontsize=16, weight='bold')
    axes[0].set_xlabel('TF-IDF Score', fontsize=12)
    axes[0].tick_params(axis='y', labelsize=11)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)



    # sort data by similarity score for this plot
    sim_sorted = top_genes.sort_values('sim_raw_norm', ascending=True)

    axes[1].barh(
        sim_sorted['gene'].str.upper(),
        sim_sorted['sim_raw_norm'],
        color='salmon'
    )
    axes[1].set_title('Top 20 Gene Candidates by Semantic Similarity Score', fontsize=16, weight='bold')
    axes[1].set_xlabel('Semantic Similarity Score ', fontsize=12)
    axes[1].tick_params(axis='y', labelsize=11) # Adjust y-axis label size
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)


    fig.suptitle(
        f'Score Analysis of Top Gene Candidates (fastText)',
        fontsize=20,
        weight='bold'
    )

    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # saving figure
    output_fig_filename = f'separate_scores_visualization_{model_choice}.png'
    plt.savefig(output_fig_filename, dpi=300)
    print(f"Figure saved as '{output_fig_filename}'")
    plt.show()



model_to_visualize = 'fasttext' # 'fasttext', 'biobert'

plot_separate_score_charts(model_choice=model_to_visualize)