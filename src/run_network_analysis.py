import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel


FINETUNED_MODEL_PATH = "./drive/MyDrive/IC_2025/biobert-finetuned-als"
SIMILARITY_THRESHOLD = 0.9  # similarity threshold


# loading and generating embeddings
print("Loading data and generating embeddings for all genes...")

df_full = pd.read_csv(f'als_articles_with_genes_{YEAR_END}.csv')
df_full['genes'] = df_full['genes'].apply(lambda x: eval(x))

all_genes_in_corpus = sorted(list(set(g.upper() for gene_list in df_full['genes'] for g in gene_list)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
model = AutoModel.from_pretrained(FINETUNED_MODEL_PATH).to(device)
model_components = {'tokenizer': tokenizer, 'model': model, 'device': device}

# getting embeddings
all_embeddings, all_genes_list = get_embeddings(all_genes_in_corpus, model_components, 'biobert')
all_embeddings = normalize(all_embeddings)

print(f"Generated embeddings for {len(all_genes_list)} unique genes.")


# creating the network
print(f"\nCreating similarity network with threshold: {SIMILARITY_THRESHOLD}...")

similarity_matrix = cosine_similarity(all_embeddings)

G = nx.Graph()
for i in range(len(all_genes_list)):
    G.add_node(all_genes_list[i])
    for j in range(i + 1, len(all_genes_list)):
        if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            G.add_edge(all_genes_list[i], all_genes_list[j], weight=similarity_matrix[i, j])

print(f"Initial network: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# removing isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Filtered network (no isolates): {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


# network metrics
print("\n--- Network Analysis Results ---")

avg_clustering = nx.average_clustering(G)
print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

print("\nTop 10 Genes by Betweenness Centrality:")
betweenness = nx.betweenness_centrality(G)
sorted_betweenness = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
for gene, score in sorted_betweenness[:10]:
    print(f"- {gene}: {score:.4f}")

print("\nDetecting community structure with Louvain algorithm...")
partition = community_louvain.best_partition(G)
modularity = community_louvain.modularity(partition, G)
num_communities = len(set(partition.values()))

print(f"Number of communities detected: {num_communities}")
print(f"Modularity Score: {modularity:.4f}")

print("\nGenes in the largest communities:")
community_sizes = pd.Series(partition).value_counts()
for i in range(min(5, num_communities)):
    community_id = community_sizes.index[i]
    community_genes = [node for node, com in partition.items() if com == community_id]
    print(f"\nCommunity {community_id} ({len(community_genes)} genes):")
    print(", ".join(community_genes[:15]) + ('...' if len(community_genes) > 15 else ''))


# full network
print("\nGenerating network visualization...")
plt.figure(figsize=(15, 15))

pos = nx.spring_layout(G, seed=42)

colors = [partition[node] for node in G.nodes()]
degrees = dict(G.degree())
node_sizes = [degrees[node] * 50 for node in G.nodes()]  # size is proportional to degree

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, cmap=plt.cm.jet)
nx.draw_networkx_edges(G, pos, alpha=0.3)

plt.title("Gene Similarity Network (Colored by Community, Node Size ~ Degree)", fontsize=20)
plt.show()


# plot top communities
print("\nPlotting top communities...")

partition = community_louvain.best_partition(G)
community_sizes = pd.Series(partition).value_counts()
num_top_communities_to_plot = 3

for i in range(min(num_top_communities_to_plot, len(community_sizes))):
    community_id = community_sizes.index[i]
    community_nodes = [node for node, com in partition.items() if com == community_id]
    community_graph = G.subgraph(community_nodes)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(community_graph, seed=42, k=0.5, iterations=100)

    degrees_sub = dict(community_graph.degree())
    node_sizes_sub = [degrees_sub[node] * 100 for node in community_graph.nodes()]

    nx.draw_networkx_nodes(community_graph, pos, node_size=node_sizes_sub, node_color='skyblue')
    nx.draw_networkx_edges(community_graph, pos, alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(community_graph, pos, font_size=10)

    plt.title(f"Community {community_id} ({community_graph.number_of_nodes()} Genes)", fontsize=20, weight='bold')
    output_fig_filename = f'network_community_{community_id}.png'
    plt.savefig(output_fig_filename, dpi=300, bbox_inches='tight')
    plt.show()
