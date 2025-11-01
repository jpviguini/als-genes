import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


MODELS = ["w2v"]
BASE_DIR = "./validation/per_gene/"
OUTPUT_DIR = "./validation/plots/"
COLUMN = "dot_minmax"
LATEX_FILE = os.path.join(OUTPUT_DIR, "report_gene_evolution.tex")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Source of discovery years:
# https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1170996/full

GENE_DISCOVERY_YEAR = {
    'ANXA11': 2017, 'C9ORF72': 2011, 'CHCHD10': 2014, 'EPHA4': 2012,
    'FUS': 2009, 'HNRNPA1': 2013, 'KIF5A': 2018, 'NEK1': 2016,
    'OPTN': 2010, 'PFN1': 2012, 'SOD1': 1993, 'TARDBP': 2008,
    'TBK1': 2015, 'UBQLN2': 2011, 'UNC13A': 2009, 'VAPB': 2004, 'VCP': 2010
}


WINDOW_SIZE = 3       
X_THRESHOLD = 0.02    # threshold for positive derivative
T_THRESHOLD = 0.04    # threshold for overall increase

def detect_latent_knowledge(df):
    """
        Detects year in which the notification could be released
    """

    years = df["year"].values
    values = df[COLUMN].values
    n = len(values)

    for i in range(n - WINDOW_SIZE + 1):
        window_years = years[i:i+WINDOW_SIZE]
        window_values = values[i:i+WINDOW_SIZE]

        # 1st criteria: positive derivative and small threshold
        derivadas = np.diff(window_values)
        derivada_media = np.mean(derivadas)
        crit1 = derivada_media >= X_THRESHOLD

        # 2nd criteria: overall increase and big threshold
        crit2 = (window_values[-1] - window_values[0]) >= T_THRESHOLD

        if crit1 or crit2:
        
            # year of notification is the following year after the window
            notification_year = window_years[-1] + 1
            return notification_year

    return None


# plots the temporal visualization
def plot_gene(df, gene, model_name, column):

    plt.figure(figsize=(8, 4))

    
    df = df[df[column] > 0] # only positive dot products

    if df.empty:
        print(f"No positive values for {gene}")
        return None

    # the discovery year is the last year of the plot
    discovery_year = GENE_DISCOVERY_YEAR.get(gene.upper())
    if discovery_year:
        df = df[df["year"] <= discovery_year]


    notification_year = detect_latent_knowledge(df)


    plt.plot(df["year"], df[column], marker="o", linewidth=2, label="Evolution")

   
    if discovery_year:
        plt.axvline(x=discovery_year, color="red", linestyle="--", linewidth=1.8,
                    label=f"Discovery ({discovery_year})")


    if notification_year:
        notif_value = df[df["year"] == notification_year][column].values
        if len(notif_value) > 0:
            plt.scatter(notification_year, notif_value,
                        color="green", s=100, zorder=5, label=f"Notification ({notification_year})")


    plt.title(f"{gene.upper()} ({model_name})", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel(column.replace("_", " ").capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"{gene}_{model_name}_{column}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filename

def process_model(model_name, latex_entries):
    folder = os.path.join(BASE_DIR, model_name)
    if not os.path.exists(folder):
        print(f"Folder {folder} not found.")
        return

    print(f"Generating plots for {model_name.upper()}...")
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files):
        gene = csv_file.replace(".csv", "")
        file_path = os.path.join(folder, csv_file)

        try:
            df = pd.read_csv(file_path)
            if COLUMN not in df.columns or "year" not in df.columns:
                continue

            img_name = plot_gene(df, gene, model_name.upper(), COLUMN)
            if img_name is None:
                continue

            latex_entries.append(f"""
\\section*{{{gene.upper()} ({model_name.upper()})}}
\\begin{{figure}}[h!]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{img_name}}}
    \\caption{{Temporal evolution of {gene.upper()} ({model_name.upper()}) till the year of discovery({GENE_DISCOVERY_YEAR.get(gene.upper(), '?')}).}}
\\end{{figure}}
\\newpage
""")
        except Exception as e:
            print(f"Error when processing {gene}: {e}")

def build_latex(entries):
    with open(LATEX_FILE, "w") as f:
        f.write(r"""
\documentclass[a4paper,12pt]{article}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
\title{Temporal Dot Product Evolution of ALS genes}
\author{Word2Vec and FastText}
\maketitle
""")
        f.writelines(entries)
        f.write("\n\\end{document}")

def main():
    latex_entries = []
    for model in MODELS:
        process_model(model, latex_entries)
    build_latex(latex_entries)
    print(f"\nVisualizations: {OUTPUT_DIR}")
    print(f"LaTeX file: {LATEX_FILE}")

if __name__ == "__main__":
    main()
