from Bio import Entrez

Entrez.email = "jpvviguini@gmail.com" # replace
API_KEY = "7edcb0657d8ffc045a7eec1068abad863b09"   # replace
YEAR_START = 2000
YEAR_END = 2025
MAX_ARTICLES = 100 # how many articles you want to retrieve
SLEEP_TIME = 0.37
MAX_RETRIES = 3
CHUNK_SIZE = 200
VALIDATE_WITH_MYGENE = True
MYGENE_BATCH_SIZE = 1000


# base_query = (
#     '("amyotrophic lateral sclerosis"[tiab] OR "motor neuron disease"[tiab] OR MND[tiab] OR ALS[tiab]) AND '
#     '("gene"[tiab] OR "genes"[tiab] OR genetic[tiab] OR mutation*[tiab] OR polymorphism*[tiab] OR "Genome-Wide Association Study"[Mesh] OR GWAS[tiab])'
# )
BASE_QUERY = (
  '( "gene[tiab]" OR "genes[tiab]" OR genetic[tiab] '
  'OR mutation*[tiab] OR polymorphism*[tiab] '
  'OR variant*[tiab] OR SNP[tiab] OR SNPs[tiab] '
  'OR loci[tiab] OR locus[tiab] '
  'OR GWAS[tiab] OR "genome-wide association"[tiab] '
  'OR expression[tiab] ) '
  'AND '
  '( association*[tiab] OR relationship*[tiab] '
  'OR correlation*[tiab] OR interaction*[tiab] '
  'OR linkage[tiab] OR "risk factor*"[tiab] '
  'OR susceptib*[tiab] OR regulat*[tiab] )'
)



GENE_STOPWORDS = set([
    "THE", "AND", "WITH", "FOR", "WAS", "WERE", "ARE", "OUR", "FROM",
    "THIS", "THAT", "THAN", "DISEASE", "PATIENT", "PATIENTS", "GENETIC", "RISK",
    "STUDY", "GENE", "GENES", "ANALYSIS", "RESULT", "RESULTS", "DATA", "MODEL",
    "MODELS", "TYPE", "CASE", "CASES", "ALS", "LATERAL", "SCLEROSIS", "MOTOR",
    "NEURON", "DNA", "RNA", "PROTEIN", "CELL", "CELLS", "TISSUE", "BRAIN",
    "NEURONS", "MOUSE", "MICE",

    "OF", "IN", "TO", "ON", "BY", "AS", "AN", "OR", "IS", "BE", "WE",
    "NOT", "THESE", "HAVE", "HAS", "WITHIN", "FOUND", "US", "INCREASE", "IMPACT"
])


# regex patterns as a fallback
REGEX_PATTERNS = [
    r"\bC\d+ORF\d+\b",           # ex: C9ORF72
    r"\bRS\d{3,9}\b",              # SNP ids: rs123456
    r"\b[A-Z]{2,4}-\d{1,3}\b",    # ex: ABC-1, TDP-43
    r"\b[A-Z]{3,6}[0-9]{0,3}\b"  # ex: SOD1, TP53
]


VALIDATION_GENES = {
    "ANXA11", "C9ORF72", "CHCHD10", "EPHA4", "FUS", "HNRNPA1", "KIF5A", "NEK1",
    "OPTN", "PFN1", "SOD1", "TARDBP", "TDP-43", "TDP43", "TBK1", "UBQLN2",
    "UNC13A", "VAPB", "VCP"
}


MODEL_CHOICE = 'fasttext' 

# file paths
DATA_PATH = f"./data/als_articles_with_genes_{YEAR_END}.csv"
CORPUS_PATH = "./fasttext_corpus.txt"
MODEL_PATH = f"./models/fasttext_model_{YEAR_END}.bin"
RESULTS_PATH = f'./results/als_novel_gene_candidates_{MODEL_CHOICE}.csv'