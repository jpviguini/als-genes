from src import config
from tqdm import tqdm
import time



def safe_read_abstract(article):
    try:
        art = article['MedlineCitation']['Article']
        abstract_field = art.get('Abstract')
        if not abstract_field:
            return ''
        abstract_text = abstract_field.get('AbstractText')
        if not abstract_text:
            return ''
        abstract_parts = []
        for a in abstract_text:
            if isinstance(a, dict):
                txt = a.get('#text') or a.get('label') or a.get('Label') or ''
                abstract_parts.append(str(txt))
            else:
                abstract_parts.append(str(a))
        return ' '.join([p for p in abstract_parts if p])
    except Exception:
        return ''

def get_als_genetic_articles(query, start_year, end_year, max_articles=20000):
    all_articles = []
    try:
        handle = config.Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=0,
            mindate=str(start_year),
            maxdate=str(end_year),
            datetype="pdat",
            api_key=config.API_KEY
        )
        result = config.Entrez.read(handle)
        handle.close()
        total = int(result.get("Count", 0))
        print(f"Found {total} articles between {start_year}-{end_year}")

        if total == 0:
            return []

        for retstart in range(0, min(total, max_articles), 10000):
            for retry in range(config.MAX_RETRIES):
                try:
                    handle = config.Entrez.esearch(
                        db="pubmed",
                        term=query,
                        retmax=10000,
                        retstart=retstart,
                        mindate=str(start_year),
                        maxdate=str(end_year),
                        datetype="pdat",
                        api_key=config.API_KEY
                    )
                    search_result = config.Entrez.read(handle)
                    handle.close()
                    id_list = search_result.get("IdList", [])

                    for i in tqdm(range(0, len(id_list), config.CHUNK_SIZE)):
                        batch = id_list[i:i+config.CHUNK_SIZE]
                        fetch_handle = config.Entrez.efetch(
                            db="pubmed",
                            id=batch,
                            retmode="xml",
                            api_key=config.API_KEY
                        )
                        try:
                            data = config.Entrez.read(fetch_handle)
                        except Exception:
                            data = {}
                        finally:
                            fetch_handle.close()

                        for article in data.get('PubmedArticle', []):
                            try:
                                title = article['MedlineCitation']['Article'].get('ArticleTitle', '')
                                abstract = safe_read_abstract(article)
                                pmid = str(article['MedlineCitation']['PMID'])
                                text = f"{title} {abstract}".strip()
                                all_articles.append({
                                    "pmid": pmid,
                                    "title": title,
                                    "abstract": abstract,
                                    "text": text
                                })
                            except KeyError:
                                continue
                        time.sleep(config.SLEEP_TIME)
                    break
                except Exception as e:
                    print(f"Attempt {retry+1} failed: {e}")
                    time.sleep(2 ** retry)
    except Exception as e:
        print(f"Fatal error in PubMed query: {e}")
    return all_articles