"""
Spectral Taxonomy Discovery on arXiv corpus.
Domain-agnostic replication of the QB corpus analysis.
"""
import json
import numpy as np
from pathlib import Path
import time

# ─── Load corpus ───
print("Loading corpus...")
with open(Path(__file__).parent / "arxiv_corpus.json") as f:
    docs = json.load(f)
print(f"Loaded {len(docs)} documents")

# ─── Embed with Nomic v2 MoE ───
print("Loading Nomic Embed v2 MoE...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

texts = [f"search_document: {d['text']}" for d in docs]
print(f"Embedding {len(texts)} documents...")
t0 = time.time()
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
embed_time = time.time() - t0
print(f"Embedded in {embed_time:.1f}s ({len(texts)/embed_time:.0f} docs/s)")
print(f"Embedding shape: {embeddings.shape}")

# ─── Spectral Decomposition ───
print("\n=== Spectral Decomposition ===")
t0 = time.time()

N, D = embeddings.shape
X = embeddings - embeddings.mean(axis=0)  # Center
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Row-normalize

# Gram matrix
G = X @ X.T
eigenvalues = np.linalg.eigvalsh(G)
eigenvalues = eigenvalues[::-1]  # Descending

# Marchenko-Pastur edge
gamma = N / D
sigma2 = 1.0
lambda_plus = sigma2 * (1 + np.sqrt(gamma))**2
signal_eigs = eigenvalues[eigenvalues > lambda_plus]
n_signal = len(signal_eigs)
print(f"N={N}, D={D}, γ={gamma:.3f}")
print(f"MP edge λ+ = {lambda_plus:.3f}")
print(f"Signal eigenvalues: {n_signal}")
print(f"Dominant eigenvalue: λ₁ = {eigenvalues[0]:.1f}")

# Mean spacing ratio (full spectrum above MP edge)
def spacing_ratio(eigs):
    """Compute mean spacing ratio ⟨r⟩ for a set of eigenvalues."""
    if len(eigs) < 3:
        return None
    spacings = np.diff(eigs[::-1])  # ascending order gaps
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(ratios)

r_bulk = spacing_ratio(eigenvalues[:n_signal])
r_full = spacing_ratio(eigenvalues)
print(f"⟨r⟩ signal eigenvalues: {r_bulk:.3f}")
print(f"⟨r⟩ full spectrum: {r_full:.3f}")

# ─── Eigengap analysis for K ───
print("\n=== Eigengap Analysis ===")
gaps = signal_eigs[:-1] / signal_eigs[1:]
best_gaps = np.argsort(gaps)[::-1][:5]
for i, idx in enumerate(best_gaps):
    print(f"  Gap {i+1}: K={idx+1}, ratio={gaps[idx]:.3f} (λ{idx+1}={signal_eigs[idx]:.2f} / λ{idx+2}={signal_eigs[idx+1]:.2f})")

# Choose K from largest eigengap (skip K=1 which is trivial)
K = best_gaps[0] + 1
if K == 1 and len(best_gaps) > 1:
    K = best_gaps[1] + 1
print(f"\nChosen K = {K}")

# ─── Spectral Clustering ───
print(f"\n=== Spectral Clustering (K={K}) ===")
from sklearn.cluster import KMeans

# Project onto top-K eigenvectors
eigvals_full, eigvecs_full = np.linalg.eigh(G)
idx_sorted = np.argsort(eigvals_full)[::-1]
top_K_vecs = eigvecs_full[:, idx_sorted[:K]]

# Row-normalize projections
norms = np.linalg.norm(top_K_vecs, axis=1, keepdims=True)
norms[norms == 0] = 1
top_K_vecs = top_K_vecs / norms

# K-means
km = KMeans(n_clusters=K, n_init=10, random_state=42)
labels = km.fit_predict(top_K_vecs)

# ─── Per-cluster analysis ───
print("\n=== Per-Cluster Spectral Analysis ===")
from collections import Counter

results = []
for c in range(K):
    mask = labels == c
    n_c = mask.sum()
    cluster_embeddings = X[mask]
    
    # Intra-cluster Gram matrix
    G_c = cluster_embeddings @ cluster_embeddings.T
    eigs_c = np.linalg.eigvalsh(G_c)[::-1]
    
    # Coherence (mean pairwise cosine similarity)
    cosines = G_c[np.triu_indices(n_c, k=1)]
    coherence = np.mean(cosines)
    
    # ⟨r⟩ for cluster
    r_c = spacing_ratio(eigs_c)
    
    # Category distribution
    cat_dist = Counter(docs[i]["query_category"] for i in np.where(mask)[0])
    top_cats = cat_dist.most_common(3)
    
    # Sample titles
    sample_titles = [docs[i]["title"][:80] for i in np.where(mask)[0][:3]]
    
    results.append({
        "cluster": c,
        "n": int(n_c),
        "r": float(r_c) if r_c else None,
        "coherence": float(coherence),
        "top_categories": [(cat, count) for cat, count in top_cats],
        "sample_titles": sample_titles,
    })
    
    regime = "Poisson" if r_c and r_c < 0.50 else "GOE" if r_c and r_c < 0.56 else "GUE" if r_c else "N/A"
    cats_str = ", ".join(f"{cat}({count})" for cat, count in top_cats)
    print(f"  C{c:02d}: N={n_c:3d}, ⟨r⟩={r_c:.3f} ({regime:7s}), coh={coherence:.3f} | {cats_str}")

decomp_time = time.time() - t0
print(f"\nSpectral analysis completed in {decomp_time:.1f}s")

# ─── Save results ───
output = {
    "corpus": "arXiv multi-category physics/math/CS abstracts",
    "corpus_size": len(docs),
    "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
    "embedding_dim": int(D),
    "embedding_time_s": embed_time,
    "spectral_analysis_time_s": decomp_time,
    "mp_edge": float(lambda_plus),
    "n_signal_eigenvalues": int(n_signal),
    "dominant_eigenvalue": float(eigenvalues[0]),
    "r_signal": float(r_bulk) if r_bulk else None,
    "r_full": float(r_full) if r_full else None,
    "K": int(K),
    "eigengaps": [(int(idx+1), float(gaps[idx])) for idx in best_gaps],
    "clusters": results,
    "category_counts": dict(Counter(d["query_category"] for d in docs)),
}

outpath = Path(__file__).parent / "arxiv_spectral_results.json"
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {outpath}")

# ─── Summary table ───
print("\n" + "="*80)
print("SUMMARY TABLE FOR PAPER")
print("="*80)
print(f"{'ID':<6} {'Dominant Category':<30} {'N':>4} {'Coherence':>10} {'⟨r⟩':>8} {'Regime':<10}")
print("-"*80)
for r in sorted(results, key=lambda x: x["r"] or 0):
    top = r["top_categories"][0][0] if r["top_categories"] else "unknown"
    regime = "Poisson" if r["r"] and r["r"] < 0.50 else "GOE" if r["r"] and r["r"] < 0.56 else "GUE" if r["r"] else "N/A"
    print(f"C{r['cluster']:02d}   {top:<30} {r['n']:4d} {r['coherence']:10.3f} {r['r']:8.3f} {regime:<10}")
