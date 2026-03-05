"""
Spectral Taxonomy Discovery on arXiv corpus — sweep K values.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
import time

# ─── Load corpus ───
with open(Path(__file__).parent / "arxiv_corpus.json") as f:
    docs = json.load(f)

# ─── Load embeddings (reuse from previous run or re-embed) ───
print("Loading Nomic Embed v2 MoE...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

texts = [f"search_document: {d['text']}" for d in docs]
print(f"Embedding {len(texts)} documents...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

N, D = embeddings.shape
X = embeddings - embeddings.mean(axis=0)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
G = X @ X.T

# Full eigendecomposition
eigvals_full, eigvecs_full = np.linalg.eigh(G)
idx_sorted = np.argsort(eigvals_full)[::-1]
eigenvalues = eigvals_full[idx_sorted]
eigvecs_sorted = eigvecs_full[:, idx_sorted]

# MP edge
gamma = N / D
lambda_plus = (1 + np.sqrt(gamma))**2
signal_eigs = eigenvalues[eigenvalues > lambda_plus]
n_signal = len(signal_eigs)

def spacing_ratio(eigs):
    if len(eigs) < 3:
        return None
    spacings = np.diff(eigs[::-1])
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(ratios)

print(f"\nN={N}, D={D}, Signal eigenvalues: {n_signal}, MP edge: {lambda_plus:.3f}")
print(f"⟨r⟩ signal: {spacing_ratio(signal_eigs):.3f}")
print(f"⟨r⟩ full: {spacing_ratio(eigenvalues):.3f}")

# ─── Sweep K = 5, 8, 10, 12 ───
for K in [5, 8, 10, 12]:
    print(f"\n{'='*80}")
    print(f"K = {K}")
    print(f"{'='*80}")
    
    top_K_vecs = eigvecs_sorted[:, :K]
    norms = np.linalg.norm(top_K_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    top_K_vecs = top_K_vecs / norms
    
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = km.fit_predict(top_K_vecs)
    
    print(f"\n{'ID':<6} {'Top Categories':<45} {'N':>4} {'Coh':>6} {'⟨r⟩':>6} {'Regime':<8}")
    print("-"*85)
    
    results = []
    for c in range(K):
        mask = labels == c
        n_c = mask.sum()
        if n_c < 5:
            continue
        
        cluster_X = X[mask]
        G_c = cluster_X @ cluster_X.T
        eigs_c = np.linalg.eigvalsh(G_c)[::-1]
        cosines = G_c[np.triu_indices(n_c, k=1)]
        coherence = np.mean(cosines)
        r_c = spacing_ratio(eigs_c)
        
        cat_dist = Counter(docs[i]["query_category"] for i in np.where(mask)[0])
        top_cats = cat_dist.most_common(3)
        cats_str = ", ".join(f"{cat}({count})" for cat, count in top_cats)
        
        regime = "Poisson" if r_c and r_c < 0.50 else "GOE" if r_c and r_c < 0.56 else "GUE" if r_c else "N/A"
        
        results.append({
            "cluster": c, "n": n_c, "r": r_c, "coherence": coherence,
            "top_categories": top_cats, "regime": regime
        })
        
        print(f"C{c:02d}   {cats_str:<45} {n_c:4d} {coherence:6.3f} {r_c:6.3f} {regime:<8}")
    
    # Save K=10 results (matches original paper)
    if K == 10:
        output = {
            "corpus": "arXiv multi-category physics/math/CS abstracts (publicly available)",
            "corpus_license": "arXiv non-exclusive license (CC BY, CC BY-SA, CC BY-NC-SA, or arXiv license)",
            "corpus_url": "https://arxiv.org/",
            "corpus_size": len(docs),
            "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
            "embedding_dim": int(D),
            "N_signal": int(n_signal),
            "mp_edge": float(lambda_plus),
            "r_signal": float(spacing_ratio(signal_eigs)),
            "r_full": float(spacing_ratio(eigenvalues)),
            "K": K,
            "clusters": [{
                "id": f"C{r['cluster']:02d}",
                "n": r["n"],
                "r": float(r["r"]) if r["r"] else None,
                "coherence": float(r["coherence"]),
                "regime": r["regime"],
                "top_categories": [(cat, count) for cat, count in r["top_categories"]],
            } for r in results],
            "categories_in_corpus": dict(Counter(d["query_category"] for d in docs)),
        }
        with open(Path(__file__).parent / "arxiv_spectral_K10.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nK=10 results saved.")
