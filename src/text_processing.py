import re
import spacy
from src import config
import gc

try:
    import mygene
    MYGENE_AVAILABLE = True
except Exception:
    print("mygene not available")
    MYGENE_AVAILABLE = False



def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    

    return text.lower()



class GeneExtractor: # spaCy NER
    def __init__(self):
        
        print("Loading NLP models...")
        

        self.nlp_ner = None
        try:
            self.nlp_ner = spacy.load("en_ner_bionlp13cg_md", disable=["tagger", "parser"])
        except Exception as e:
            print("Warning: could not load en_ner_bionlp13cg_md (NER). Error:", e)


    def extract_genes_unbiased(self, text):
        """
        - Extract gene candidates from a text (NER when available and regex as fallback).

        - Receive the original text (with capital letters/punctuation) to maximize NER retrieval.

        - Return an ordered list of gene-like symbols/tokens in capital letters.
        """

        if not text:
            return []
        text_str = str(text)
        text_upper = text_str.upper()
        genes = set()

        # NER in the original text
        if self.nlp_ner is not None:
            try:
                doc = self.nlp_ner(text_str)
                for ent in doc.ents:
                    label = getattr(ent, 'label_', '')

                    # NER labels (depends on the model)
                    if 'GENE' in label.upper() or 'PROTEIN' in label.upper() or 'GENE_PRODUCT' in label.upper():
                        norm = re.sub(r'[^A-Za-z0-9]', '', ent.text)
                        norm_up = norm.upper()
                        if 3 <= len(norm_up) <= 10 and norm_up not in config.GENE_STOPWORDS: # filter stopwords
                            genes.add(norm_up)
            except Exception:

                # if NER fails, ignore it and continue with regex
                pass


        # regex patterns
        for pattern in config.REGEX_PATTERNS:
            for match in re.findall(pattern, text_upper):

                # filter stopwords and tokens too short
                if match and match not in config.GENE_STOPWORDS and len(re.sub(r'[^A-Z0-9]', '', match)) >= 3:
                    genes.add(match)


        # removes tokens that are only numbers
        cleaned = set()
        for g in genes:
            if re.search(r'[A-Z]', g):
                cleaned.add(g)
        return sorted(cleaned)


    # validation with mygene
    def validate_genes_with_mygene(candidate_genes):
        """
        Validates a list of symbols using mygene (batch). Returns a set of validated symbols.

        """
        if not MYGENE_AVAILABLE:
            print("mygene not available; skipping validation.")
            return set()

        mg = mygene.MyGeneInfo()
        validated = set()
        candidates = list(candidate_genes)

        for i in range(0, len(candidates), config.MYGENE_BATCH_SIZE):
            batch = candidates[i:i+config.MYGENE_BATCH_SIZE]
            try:
                res = mg.querymany(batch, scopes=['symbol', 'alias', 'name'], fields='symbol,taxid', species='human', entrezonly=False)
                for r in res:
                    # r can signal notfound
                    if r is None:
                        continue

                    if isinstance(r, dict) and not r.get('notfound', False):
                        sym = r.get('symbol')
                        taxid = r.get('taxid')

                        # human (taxid 9606) or None (some results doesn't have taxid)
                        if sym and (taxid is None or int(taxid) == 9606):
                            validated.add(sym.upper())
            except Exception as e:
                print(f"mygene query batch failed: {e}")

                # in case of error, we just continue
                continue

        return validated