import re
import pandas as pd


CHAR_MAP = {
    '-': '-', '‒': '-', '–': '-', '—': '-', '¯': '-',
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
    'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o', 'ø': 'o', '×': 'x',
    'ú': 'u', 'ü': 'u', 'č': 'c', 'ğ': 'g', 'ł': 'l',
    'ń': 'n', 'ş': 's', 'ŭ': 'u', 'і': 'i', 'ј': 'j',
    'а': 'a', 'в': 'b', 'н': 'h', 'о': 'o', 'р': 'p', 'с': 'c',
    'т': 't', 'ӧ': 'o', '⁰': '0', '⁴': '4', '⁵': '5', '⁶': '6',
    '⁷': '7', '⁸': '8', '⁹': '9', '₀': '0', '₁': '1', '₂': '2',
    '₃': '3', '₅': '5', '₇': '7', '₉': '9',
}

UNITS_AND_SYMBOLS = [
    '/μm', '/mol', '°c', '≥', '≤', '<', '>', '±', '%', '/mumol',
    'day', 'month', 'year', '·', 'week', 'days', 'weeks', 'years',
    '/µl', 'μg', 'u/mg', 'mg/m', 'g/m', 'mumol/kg', '/week', '/day',
    'm²', '/kg', '®', 'ﬀ', 'ﬃ', 'ﬁ', 'ﬂ', '£', '¥', '©', '«', '¬',
    '°', '±', '²', '³', '´', '·', '¹', '»', '½', '¿', '‘', '’', '“',
    '”', '•', '˂', '˙', '˚', '˜', '…', '‰', '′', '″', '‴', '€', '™',
    '↑', '→', '↓', '∗', '∙', '∝', '∞', '∼', '≈', '≠', '≤', '≥', '≦',
    '≫', '⊘', '⊣', '⊿', '⋅', '═', '■', '▵', '⟶', '⩽', '⩾', '、',
    '气', '益', '粒', '肾', '补', '颗', '', '', '', '', '，'
]

ALS_SYNONYMS = [
    'amyotrophic[- ]lateral[- ]sclerosis',
    'lou[- ]gehrig[’\'`s]?[- ]disease',
    'motor[- ]neuron[- ]disease',
    'mnd'
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # replace special characters
    for k, v in CHAR_MAP.items():
        text = text.replace(k, v)

    # lowercase
    text = text.lower()

    # remove "et al"
    text = re.sub(r"\bet\s+al\b", " ", text)


    for sym in UNITS_AND_SYMBOLS:
        text = text.replace(sym, " ")

    # replace synonyms of ALS with "ALS"
    regex_als = r'(?i)(' + '|'.join(ALS_SYNONYMS) + r')'
    text = re.sub(regex_als, ' ALS ', text)

    # remove URLs
    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)

    # remove isolated numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # remove 1 char words
    text = re.sub(r"\b[a-zA-Z0-9]\b", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_corpus(df, text_column="text"):
    df = df.copy()
    df[text_column] = df[text_column].astype(str).apply(normalize_text)
    df = df[df[text_column].str.len() > 10]  # removes too short texts

    # remove articles that contains “mcr-als” (this is not about ALS)
    df = df[~df[text_column].str.contains(r"\bmcr[- ]?als\b", case=False, regex=True)]

    return df


if __name__ == "__main__":
    df = pd.read_csv("../data/corpus_als_with_year.csv")
    df_clean = preprocess_corpus(df)
    df_clean.to_csv("../data/corpus_als_preprocessed3.csv", index=False)
    print(f"Cleaned corpus saved in ../data/corpus_als_preprocessed3.csv with {len(df_clean)} articles.")
