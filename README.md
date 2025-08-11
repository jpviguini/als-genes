<h1 align="center">🧬 Genetic Associations in ALS Through NLP and Network Analysis</h1>

<p align="center">
  <b>Author:</b> João Pedro Viguini<sup>1</sup><br>
  <b>Advisor:</b> Prof. Dr. Ricardo Cerri<sup>1</sup>
</p>
<p align="center">
  <sup>1</sup> University of São Paulo (USP)
</p>





<p align="center">
  <em>An NLP-based approach to identify and prioritize genes potentially associated with Amyotrophic Lateral Sclerosis (ALS) and other complex diseases.</em>
</p>

<p align="center">
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="#status">
    <img src="https://img.shields.io/badge/status-work%20in%20progress-orange" alt="Status">
  </a>
</p>


---


## 📝 Overview
This repository contains the current progress of a research project supported by FAPESP, aiming to explore and prioritize genes potentially linked to Amyotrophic Lateral Sclerosis (ALS). The approach combines Natural Language Processing (NLP) and network analysis to extract, represent, and analyze gene relationships from biomedical literature.  

**Current status:**  
- Collection and preprocessing of a large corpus of PubMed abstracts related to ALS (currently limited to 10,000).
- Training of _fastText_ and _BioBERT_ models to obtain gene embeddings.  
- Construction of semantic similarity networks for community detection and association analysis.  

**Note:** GWAS data will be integrated in a future stage.

## Prerequisites
Before using the application, you must download three **.tar.gz** files:
- Pre-trained BioBERT-Base v1.1 **(biobert_v1.1_pubmed.tar.gz)** 
- scispaCy for tokenization **(en_core_sci_sm-0.5.4.tar.gz)**
- scispaCy for NER **(en_ner_bionlp13cg_md-0.5.4.tar.gz)**

Useful links: [Pretrained BioBERT](https://github.com/naver/biobert-pretrained?tab=readme-ov-file)  |  [scispaCy models](https://allenai.github.io/scispacy/)


## ⚙️ Usage
Clone this repository:
   ```bash
   git clone https://github.com/jpviguini/als-genes.git
   cd als-genes
   ```

There is a **.ipynb notebook** containing the available code in this repository.


## Acknowledgements
This project is supported by FAPESP (2025/06512-0).








