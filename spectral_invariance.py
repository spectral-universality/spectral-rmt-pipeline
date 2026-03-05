#!/usr/bin/env python3
"""
Spectral Invariance Analysis v2
================================
Comprehensive multi-encoding RMT analysis with proper controls.

Encodings:
  1. TF-IDF (raw vocabulary)
  2. TF-IDF + SVD (multiple D: 50, 100, 200, 500, 768)
  3. NMF (100 topics)
  4. Hashing vectorizer (fixed D, no vocabulary)
  5. Character n-gram TF-IDF (sub-word, D=vocab)
  6. Random projection from TF-IDF (null control)
  7. Shuffled-text control (destroys structure, keeps statistics)
  8. Label-permuted control (same embeddings, random labels)

Per-encoding, per-cluster:
  - ⟨r⟩ with bootstrap 95% CI (10000 iterations)
  - Cluster size N
  - Coherence (mean pairwise cosine)
  - Regime classification (Poisson / GOE / GUE)

Finite-size calibration:
  - Synthetic GOE/GUE/Poisson matrices at matched N
  - Expected ⟨r⟩ ± σ at each sample size

Author: Shadow + Dreadbot 3.2.666
Date: 2026-03-01
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import eigvalsh


# ═══════════════════════════════════════════════════════════════
# RMT Utilities
# ═══════════════════════════════════════════════════════════════

def spacing_ratios(eigs):
    """Spacing ratios from eigenvalues (ascending sort)."""
    eigs = np.sort(np.real(eigs))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-15]  # numerical zero filter
    if len(spacings) < 2:
        return np.array([])
    return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])


def mean_r(eigs):
    r = spacing_ratios(eigs)
    return float(np.mean(r)) if len(r) > 2 else None


def bootstrap_ci(eigs, n_boot=10000, alpha=0.05):
    """Bootstrap CI on ⟨r⟩ from eigenvalues."""
    ratios = spacing_ratios(eigs)
    if len(ratios) < 3:
        return None, None, None
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(ratios, size=len(ratios), replace=True)
        boot_means[i] = np.mean(sample)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(ratios)), float(lo), float(hi)


def classify_regime(r_val, ci_lo=None, ci_hi=None):
    """Classify with honest uncertainty."""
    if r_val is None:
        return 'N/A'
    # Strict: use CI to determine if classification is robust
    if ci_lo is not None and ci_hi is not None:
        if ci_hi < 0.50:
            return 'Poisson'
        elif ci_lo > 0.56:
            return 'GUE'
        elif ci_lo > 0.50 and ci_hi < 0.56:
            return 'GOE'
        else:
            return 'ambiguous'
    # Fallback: point estimate only
    if r_val < 0.46:
        return 'Poisson'
    elif r_val < 0.50:
        return '~Poisson'
    elif r_val < 0.56:
        return 'GOE'
    elif r_val < 0.60:
        return '~GUE'
    else:
        return 'GUE'


# ═══════════════════════════════════════════════════════════════
# Finite-Size Calibration
# ═══════════════════════════════════════════════════════════════

def finite_size_calibration(sizes, n_trials=500):
    """Generate expected ⟨r⟩ ± σ for GOE, GUE, Poisson at each N."""
    rng = np.random.default_rng(42)
    results = {}

    for N in sizes:
        cal = {}
        for ensemble, label in [('GOE', 'GOE'), ('GUE', 'GUE'), ('Poisson', 'Poisson')]:
            r_vals = []
            for _ in range(n_trials):
                if ensemble == 'GOE':
                    A = rng.standard_normal((N, N))
                    M = (A + A.T) / 2
                    eigs = np.linalg.eigvalsh(M)
                elif ensemble == 'GUE':
                    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
                    M = (A + A.conj().T) / 2
                    eigs = np.linalg.eigvalsh(M)
                else:  # Poisson
                    eigs = np.sort(rng.uniform(0, N, size=N))

                r = mean_r(eigs)
                if r is not None:
                    r_vals.append(r)

            cal[label] = {
                'mean': float(np.mean(r_vals)),
                'std': float(np.std(r_vals)),
                'ci_lo': float(np.percentile(r_vals, 2.5)),
                'ci_hi': float(np.percentile(r_vals, 97.5)),
            }

        results[N] = cal

    return results


# ═══════════════════════════════════════════════════════════════
# Spectral Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def spectral_analysis(X, K=10, label=""):
    """Full spectral pipeline on embedding matrix X.
    Returns per-cluster ⟨r⟩ with bootstrap CIs."""
    N, D = X.shape
    gamma = N / D

    # Center and normalize
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X_c = X_c / norms

    # Gram matrix
    G = X_c @ X_c.T
    eigenvalues = np.linalg.eigvalsh(G)[::-1]

    # MP edge
    sigma2 = 1.0
    lambda_plus = sigma2 * (1 + np.sqrt(gamma))**2
    signal_eigs = eigenvalues[eigenvalues > lambda_plus]
    n_signal = len(signal_eigs)

    # Global ⟨r⟩
    r_global, r_global_lo, r_global_hi = bootstrap_ci(signal_eigs) if n_signal > 3 else (None, None, None)

    # Spectral clustering
    eigvals_full, eigvecs_full = np.linalg.eigh(G)
    idx_sorted = np.argsort(eigvals_full)[::-1]
    top_K_vecs = eigvecs_full[:, idx_sorted[:K]]
    row_norms = np.linalg.norm(top_K_vecs, axis=1, keepdims=True)
    row_norms[row_norms < 1e-10] = 1
    top_K_vecs = top_K_vecs / row_norms

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = km.fit_predict(top_K_vecs)

    return {
        'label': label,
        'N': int(N), 'D': int(D), 'gamma': float(gamma),
        'mp_edge': float(lambda_plus),
        'n_signal': int(n_signal),
        'r_global': r_global,
        'r_global_ci': [r_global_lo, r_global_hi] if r_global_lo else None,
        'K': K,
        'labels': labels,
        'eigenvalues_top50': eigenvalues[:50].tolist(),
    }


def per_cluster_analysis(X, labels, docs, K):
    """Compute per-cluster RMT statistics with bootstrap CIs."""
    N = X.shape[0]
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X_c = X_c / norms

    clusters = []
    for c in range(K):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < 5:
            clusters.append({'cluster': c, 'n': n_c, 'r': None,
                           'ci': None, 'regime': 'too_small'})
            continue

        G_c = X_c[mask] @ X_c[mask].T
        eigs_c = np.linalg.eigvalsh(G_c)[::-1]

        # Coherence
        cosines = G_c[np.triu_indices(n_c, k=1)]
        coherence = float(np.mean(cosines))

        # Bootstrap ⟨r⟩
        r_mean, ci_lo, ci_hi = bootstrap_ci(eigs_c)
        regime = classify_regime(r_mean, ci_lo, ci_hi)

        # Category distribution
        cat_dist = Counter(docs[i]['primary_category'] for i in np.where(mask)[0])
        top_cat = cat_dist.most_common(1)[0]

        clusters.append({
            'cluster': c,
            'n': n_c,
            'r': r_mean,
            'ci': [ci_lo, ci_hi] if ci_lo is not None else None,
            'sigma': float((ci_hi - ci_lo) / (2 * 1.96)) if ci_lo is not None else None,
            'coherence': coherence,
            'regime': regime,
            'top_cat': top_cat[0],
            'top_cat_count': int(top_cat[1]),
            'purity': float(top_cat[1] / n_c),
            'category_dist': {k: int(v) for k, v in cat_dist.items()},
        })

    return clusters


# ═══════════════════════════════════════════════════════════════
# Encoding Functions
# ═══════════════════════════════════════════════════════════════

def encode_tfidf_raw(texts):
    """TF-IDF with full vocabulary."""
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95,
                            stop_words='english', sublinear_tf=True)
    X = tfidf.fit_transform(texts).toarray()
    return X, f'TF-IDF raw (D={X.shape[1]})'


def encode_tfidf_svd(texts, n_components):
    """TF-IDF + SVD dimensionality reduction."""
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95,
                            stop_words='english', sublinear_tf=True)
    X_sparse = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X = svd.fit_transform(X_sparse)
    var_exp = svd.explained_variance_ratio_.sum()
    return X, f'TF-IDF+SVD({n_components}) [var={var_exp:.2f}]'


def encode_nmf(texts, n_components=100):
    """Non-negative Matrix Factorization."""
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95,
                            stop_words='english', sublinear_tf=True)
    X_sparse = tfidf.fit_transform(texts)
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    X = nmf.fit_transform(X_sparse)
    return X, f'NMF({n_components})'


def encode_hashing(texts, n_features=768):
    """Hashing vectorizer — no vocabulary, fixed dimensionality."""
    hv = HashingVectorizer(n_features=n_features, stop_words='english',
                           alternate_sign=False)
    X = hv.fit_transform(texts).toarray()
    return X, f'HashingVec({n_features})'


def encode_char_ngram(texts, ngram_range=(2, 4)):
    """Character n-gram TF-IDF — sub-word features."""
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range,
                            max_features=10000, min_df=2, max_df=0.95,
                            sublinear_tf=True)
    X = tfidf.fit_transform(texts).toarray()
    return X, f'CharNgram{ngram_range} (D={X.shape[1]})'


def encode_random_projection(texts, n_components=768):
    """Random Gaussian projection from TF-IDF — null control."""
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95,
                            stop_words='english', sublinear_tf=True)
    X_sparse = tfidf.fit_transform(texts)
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X = rp.fit_transform(X_sparse)
    return X, f'RandomProj({n_components}) [control]'


def encode_shuffled(texts):
    """Shuffle words within each document — destroys structure, keeps word stats."""
    rng = np.random.default_rng(42)
    shuffled = []
    for t in texts:
        words = t.split()
        rng.shuffle(words)
        shuffled.append(' '.join(words))
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95,
                            stop_words='english', sublinear_tf=True)
    X = tfidf.fit_transform(shuffled).toarray()
    return X, 'Shuffled-text TF-IDF [control]'


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Load corpus ──────────────────────────────────────────
    corpus_path = Path(__file__).parent / 'data' / 'arxiv_expanded_corpus.json'

    if not corpus_path.exists():
        # Fall back to original 747
        corpus_path = Path.home() / 'papers' / 'spectral-taxonomy' / 'arxiv_corpus.json'
        print(f'⚠ Expanded corpus not found, using original: {corpus_path}')

    if not corpus_path.exists():
        print('ERROR: No corpus found. Run fetch_expanded_corpus.py first.')
        sys.exit(1)

    with open(corpus_path) as f:
        docs = json.load(f)

    # Normalize field names (original corpus uses 'query_category')
    for d in docs:
        if 'primary_category' not in d and 'query_category' in d:
            d['primary_category'] = d['query_category']

    print(f'Loaded {len(docs)} documents from {corpus_path.name}')

    # Category summary
    cat_counts = Counter(d['primary_category'] for d in docs)
    print(f'\nCategories ({len(cat_counts)}):')
    for cat, n in cat_counts.most_common():
        print(f'  {cat:<25} {n:>5}')

    texts = [d['text'] for d in docs]
    K = 15 if len(cat_counts) >= 12 else 10

    # ── Finite-size calibration ──────────────────────────────
    unique_sizes = sorted(set(cat_counts.values()))
    # Also calibrate at standard sizes
    cal_sizes = sorted(set([20, 30, 50, 75, 100, 200, 500] + unique_sizes))
    cal_sizes = [s for s in cal_sizes if 10 <= s <= 1000]

    print(f'\nRunning finite-size calibration at N = {cal_sizes}...')
    t0 = time.time()
    calibration = finite_size_calibration(cal_sizes, n_trials=500)
    print(f'  Done in {time.time()-t0:.1f}s')

    # Print calibration table
    print(f'\n  {"N":>5} {"Poisson":>12} {"GOE":>12} {"GUE":>12}')
    print(f'  {"-"*42}')
    for N in cal_sizes:
        c = calibration[N]
        print(f'  {N:>5} {c["Poisson"]["mean"]:.3f}±{c["Poisson"]["std"]:.3f}'
              f'  {c["GOE"]["mean"]:.3f}±{c["GOE"]["std"]:.3f}'
              f'  {c["GUE"]["mean"]:.3f}±{c["GUE"]["std"]:.3f}')

    # ── Run all encodings ────────────────────────────────────
    encodings = [
        lambda t: encode_tfidf_raw(t),
        lambda t: encode_tfidf_svd(t, 50),
        lambda t: encode_tfidf_svd(t, 100),
        lambda t: encode_tfidf_svd(t, 200),
        lambda t: encode_tfidf_svd(t, 500),
        lambda t: encode_tfidf_svd(t, 768),
        lambda t: encode_nmf(t, 100),
        lambda t: encode_hashing(t, 768),
        lambda t: encode_char_ngram(t),
        lambda t: encode_random_projection(t, 768),
        lambda t: encode_shuffled(t),
    ]

    all_results = {}

    for encode_fn in encodings:
        print(f'\n{"="*70}')
        t0 = time.time()
        X, label = encode_fn(texts)
        encode_time = time.time() - t0
        print(f'  {label}  (N={X.shape[0]}, D={X.shape[1]}, encoded in {encode_time:.1f}s)')

        # Spectral analysis
        t0 = time.time()
        result = spectral_analysis(X, K=K, label=label)
        clusters = per_cluster_analysis(X, result['labels'], docs, K)
        result['clusters'] = clusters
        result['encode_time'] = encode_time
        result['analysis_time'] = time.time() - t0

        # Print cluster table
        print(f'\n  {"ID":<5} {"Dom Category":<22} {"N":>4} {"⟨r⟩":>7} {"CI":>15} {"σ":>6} {"Regime":<12}')
        print(f'  {"-"*75}')
        for c in sorted(clusters, key=lambda x: x['r'] if x['r'] else 0):
            if c['r'] is None:
                print(f'  C{c["cluster"]:<3} {"??":<22} {c["n"]:>4}     N/A')
                continue
            ci_str = f'[{c["ci"][0]:.3f},{c["ci"][1]:.3f}]' if c['ci'] else '       N/A'
            sigma_str = f'{c["sigma"]:.3f}' if c['sigma'] else '  N/A'
            print(f'  C{c["cluster"]:<3} {c["top_cat"]+"("+str(c["top_cat_count"])+"/"+str(c["n"])+")":22} '
                  f'{c["n"]:>4} {c["r"]:>7.3f} {ci_str:>15} {sigma_str:>6} {c["regime"]:<12}')

        # Strip numpy arrays for JSON
        result_clean = {k: v for k, v in result.items() if k != 'labels'}
        result_clean['labels'] = result['labels'].tolist()
        all_results[label] = result_clean

    # ── Label-permuted control ───────────────────────────────
    # Use TF-IDF raw embeddings but shuffle category labels
    print(f'\n{"="*70}')
    print(f'  Label-Permuted Control (TF-IDF raw, shuffled categories)')
    print(f'{"="*70}')

    X_raw, _ = encode_tfidf_raw(texts)
    result_perm = spectral_analysis(X_raw, K=K, label='Label-permuted [control]')

    # Shuffle category assignments
    rng = np.random.default_rng(42)
    docs_shuffled = [dict(d) for d in docs]
    cats_shuffled = [d['primary_category'] for d in docs_shuffled]
    rng.shuffle(cats_shuffled)
    for d, cat in zip(docs_shuffled, cats_shuffled):
        d['primary_category'] = cat

    clusters_perm = per_cluster_analysis(X_raw, result_perm['labels'], docs_shuffled, K)
    result_perm['clusters'] = clusters_perm
    result_perm_clean = {k: v for k, v in result_perm.items() if k != 'labels'}
    result_perm_clean['labels'] = result_perm['labels'].tolist()
    all_results['Label-permuted [control]'] = result_perm_clean

    # ── Cross-encoding summary ───────────────────────────────
    print(f'\n\n{"="*70}')
    print(f'  CROSS-ENCODING SUMMARY')
    print(f'{"="*70}')

    print(f'\n  {"Encoding":<35} {"γ":>6} {"Sig":>4} {"⟨r⟩_g":>7} '
          f'{"Min ⟨r⟩":>8} {"Max ⟨r⟩":>8} {"Spread":>8}')
    print(f'  {"-"*78}')

    for name, res in all_results.items():
        r_vals = [c['r'] for c in res['clusters'] if c['r'] is not None]
        if not r_vals:
            continue
        r_g = f'{res["r_global"]:.3f}' if res.get('r_global') else 'N/A'
        print(f'  {name[:35]:<35} {res["gamma"]:>6.2f} {res["n_signal"]:>4} {r_g:>7} '
              f'{min(r_vals):>8.3f} {max(r_vals):>8.3f} {max(r_vals)-min(r_vals):>8.3f}')

    # ── Save everything ──────────────────────────────────────
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    output = {
        'corpus_source': str(corpus_path),
        'n_documents': len(docs),
        'n_categories': len(cat_counts),
        'category_counts': dict(cat_counts),
        'K': K,
        'calibration': {str(k): v for k, v in calibration.items()},
        'encodings': {},
    }

    for name, res in all_results.items():
        # Don't save full label arrays (large)
        res_save = {k: v for k, v in res.items() if k != 'labels'}
        output['encodings'][name] = res_save

    outpath = output_dir / 'spectral_invariance_v2.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=1, default=str)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
