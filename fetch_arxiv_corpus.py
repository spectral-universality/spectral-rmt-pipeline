"""
Fetch a clean, publicly available corpus from arXiv for spectral taxonomy discovery.
Multi-category physics/math abstracts — diverse enough to produce interesting spectral structure.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import os

CATEGORIES = [
    ("hep-th", "High Energy Physics - Theory", 100),
    ("cond-mat.stat-mech", "Statistical Mechanics", 100),
    ("quant-ph", "Quantum Physics", 100),
    ("math-ph", "Mathematical Physics", 100),
    ("nlin.CD", "Chaotic Dynamics", 80),
    ("physics.bio-ph", "Biological Physics", 80),
    ("math.NT", "Number Theory", 80),
    ("cs.LG", "Machine Learning", 80),
    ("stat.ML", "Statistical ML", 60),
    ("math.SP", "Spectral Theory", 60),
    ("physics.data-an", "Data Analysis", 60),
    ("math.PR", "Probability", 60),
]

BASE_URL = "http://export.arxiv.org/api/query"
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

def fetch_category(category, max_results):
    """Fetch abstracts from a single arXiv category."""
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "SpectralTaxonomy/1.0 (shadow@shdwcorp.cloud)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    
    root = ET.fromstring(data)
    docs = []
    for entry in root.findall("atom:entry", NAMESPACE):
        title = entry.find("atom:title", NAMESPACE).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NAMESPACE).text.strip().replace("\n", " ")
        arxiv_id = entry.find("atom:id", NAMESPACE).text.strip().split("/")[-1]
        
        # Get primary category
        primary_cat = entry.find("atom:primary_category", NAMESPACE)
        if primary_cat is not None:
            primary = primary_cat.get("term", category)
        else:
            primary = category
        
        # Get all categories
        categories = [c.get("term") for c in entry.findall("atom:category", NAMESPACE)]
        
        # Get authors
        authors = [a.find("atom:name", NAMESPACE).text for a in entry.findall("atom:author", NAMESPACE)]
        
        # Get published date
        published = entry.find("atom:published", NAMESPACE).text[:10]
        
        docs.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "primary_category": primary,
            "categories": categories,
            "authors": authors[:3],  # First 3 authors
            "published": published,
            "text": f"{title}. {abstract}",  # Combined for embedding
        })
    
    return docs

def main():
    all_docs = []
    seen_ids = set()
    
    for cat, label, n in CATEGORIES:
        print(f"Fetching {label} ({cat}): {n} papers...")
        try:
            docs = fetch_category(cat, n)
            new = 0
            for d in docs:
                if d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    d["query_category"] = cat
                    d["query_label"] = label
                    all_docs.append(d)
                    new += 1
            print(f"  Got {len(docs)}, {new} unique (total: {len(all_docs)})")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(3)  # Be polite to arXiv API
    
    # Save corpus
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, "arxiv_corpus.json")
    with open(outpath, "w") as f:
        json.dump(all_docs, f, indent=2)
    
    print(f"\nTotal unique documents: {len(all_docs)}")
    print(f"Saved to: {outpath}")
    
    # Summary by category
    from collections import Counter
    cats = Counter(d["query_category"] for d in all_docs)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
