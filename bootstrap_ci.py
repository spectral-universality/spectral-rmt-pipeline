"""
Bootstrap confidence intervals for per-cluster ⟨r⟩ values.
Also checks K-robustness of regime assignments.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
import time

# ─── Load and embed ───
with open(Path(__file__).parent / "arxiv_corpus.json") as f:
    docs = json.load(f)

print("Loading model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
texts = [f"search_document: {d['text']}" for d in docs]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

N, D = embeddings.shape
X = embeddings - embeddings.mean(axis=0)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
G = X @ X.T

eigvals_full, eigvecs_full = np.linalg.eigh(G)
idx_sorted = np.argsort(eigvals_full)[::-1]
eigenvalues = eigvals_full[idx_sorted]
eigvecs_sorted = eigvecs_full[:, idx_sorted]

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
    return float(np.mean(ratios))

def bootstrap_r(eigs, n_boot=10000):
    """Bootstrap CI for ⟨r⟩."""
    if len(eigs) < 5:
        return None, None, None
    spacings = np.diff(eigs[::-1])
    spacings = spacings[spacings > 0]
    if len(spacings) < 3:
        return None, None, None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    
    r_mean = np.mean(ratios)
    boot_means = np.zeros(n_boot)
    n_ratios = len(ratios)
    for b in range(n_boot):
        idx = np.random.randint(0, n_ratios, size=n_ratios)
        boot_means[b] = np.mean(ratios[idx])
    
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    sigma = np.std(boot_means)
    return r_mean, sigma, (ci_lo, ci_hi)

# ─── K=10 clustering + bootstrap ───
K = 10
top_K_vecs = eigvecs_sorted[:, :K]
norms = np.linalg.norm(top_K_vecs, axis=1, keepdims=True)
norms[norms == 0] = 1
top_K_vecs = top_K_vecs / norms
km = KMeans(n_clusters=K, n_init=10, random_state=42)
labels_K10 = km.fit_predict(top_K_vecs)

print(f"\n{'='*90}")
print(f"K=10 Bootstrap Confidence Intervals (10,000 resamples)")
print(f"{'='*90}")
print(f"{'ID':<6} {'Top Cat':<25} {'N':>4} {'⟨r⟩':>7} {'±σ':>7} {'95% CI':>18} {'Regime':<10}")
print("-"*90)

k10_results = []
for c in range(K):
    mask = labels_K10 == c
    n_c = mask.sum()
    if n_c < 5:
        continue
    
    cluster_X = X[mask]
    G_c = cluster_X @ cluster_X.T
    eigs_c = np.linalg.eigvalsh(G_c)[::-1]
    
    cosines = G_c[np.triu_indices(n_c, k=1)]
    coherence = float(np.mean(cosines))
    
    r_mean, sigma, ci = bootstrap_r(eigs_c)
    
    cat_dist = Counter(docs[i]["query_category"] for i in np.where(mask)[0])
    top_cat = cat_dist.most_common(1)[0][0]
    
    # Sample titles for C00 and C09
    sample = []
    if c in [0, 9]:
        sample = [docs[i]["title"][:100] for i in np.where(mask)[0][:5]]
    
    regime = "Poisson" if r_mean < 0.50 else "GOE" if r_mean < 0.56 else "GUE"
    
    k10_results.append({
        "cluster": c, "n": n_c, "r": r_mean, "sigma": sigma,
        "ci": ci, "coherence": coherence, "top_cat": top_cat,
        "regime": regime, "sample_titles": sample
    })
    
    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"C{c:02d}   {top_cat:<25} {n_c:4d} {r_mean:7.3f} {sigma:7.3f} {ci_str:>18} {regime:<10}")

# ─── K-robustness check ───
print(f"\n{'='*90}")
print(f"K-ROBUSTNESS: Regime stability across K ∈ {{5, 8, 10, 12}}")
print(f"{'='*90}")

for K_test in [5, 8, 10, 12]:
    top_vecs = eigvecs_sorted[:, :K_test]
    norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    top_vecs = top_vecs / norms
    km_test = KMeans(n_clusters=K_test, n_init=10, random_state=42)
    labels_test = km_test.fit_predict(top_vecs)
    
    r_values = []
    for c in range(K_test):
        mask = labels_test == c
        n_c = mask.sum()
        if n_c < 5:
            continue
        cluster_X = X[mask]
        G_c = cluster_X @ cluster_X.T
        eigs_c = np.linalg.eigvalsh(G_c)[::-1]
        r_c = spacing_ratio(eigs_c)
        if r_c:
            r_values.append(r_c)
    
    n_poisson = sum(1 for r in r_values if r < 0.50)
    n_goe = sum(1 for r in r_values if 0.50 <= r < 0.56)
    n_gue = sum(1 for r in r_values if r >= 0.56)
    r_range = f"[{min(r_values):.3f}, {max(r_values):.3f}]"
    
    print(f"  K={K_test:2d}: {len(r_values)} clusters, Poisson={n_poisson}, GOE={n_goe}, GUE={n_gue}, range={r_range}")

# ─── Print sample titles for C00 and C09 ───
print(f"\n{'='*90}")
print("SAMPLE TITLES: C00 (GUE, hep-th) vs C09 (GOE boundary, mixed)")
print(f"{'='*90}")
for r in k10_results:
    if r["cluster"] in [0, 9] and r["sample_titles"]:
        print(f"\nC{r['cluster']:02d} (⟨r⟩={r['r']:.3f}, {r['regime']}):")
        for t in r["sample_titles"]:
            print(f"  - {t}")

# ─── Eigengap data for figure ───
print(f"\n{'='*90}")
print("EIGENGAP RATIOS (for figure)")
print(f"{'='*90}")
gaps = signal_eigs[:-1] / signal_eigs[1:]
for i in range(min(15, len(gaps))):
    print(f"  K={i+1}: gap ratio = {gaps[i]:.3f} (λ{i+1}={signal_eigs[i]:.2f} → λ{i+2}={signal_eigs[i+1]:.2f})")
