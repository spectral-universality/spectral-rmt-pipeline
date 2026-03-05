cat << 'EOF' > ~/spectral-universality-repo/README.md
# Spectral RMT Pipeline

RMT spectral analysis of neural network representations and embedding spaces.

## What This Does

Applies random matrix theory diagnostics to transformer weight matrices, activation covariances, and embedding similarity matrices. Three diagnostics per layer per domain:

- **⟨r⟩** — eigenvalue spacing ratio. Classifies Poisson/GOE/GUE regimes.
- **α** — spectral dimension (Liu-Paquette-Sous exponent). Tracks effective dimensionality.
- **Soft rank** — energy distribution across eigenvalue spectrum.

## What It Found

**Embedding level:** ⟨r⟩ classifies corpus information architecture into Poisson (self-contained) through GUE (entangled). Reproducible on 747 arXiv abstracts with bootstrap CIs. Stable across K.

**Then it broke:** 11 encoding methods (TF-IDF, SVD, NMF, hashing, char n-grams, random projection, shuffled text) all reproduce the same regime classifications. Encrypted inputs produce the same spectral patterns. The classification is real but it's classifying co-occurrence statistics, not language geometry.

**Internal spectroscopy (B200 burst):** Ran diagnostics inside Qwen 72B (80 layers) and Llama (24 layers) on manufacturing text, random text, and code.

- ⟨r⟩ is flat Poisson across all domains at every layer — the model doesn't differentiate.
- α diverges by domain across all layers — three distinct trajectories.
- Soft rank collapses to ~1.0 in middle layers then re-expands domain-dependently in deep layers. Code gets broadest distribution.

Transformers differentiate domains in spectral dimension and energy distribution, but not in eigenvalue spacing.

## Scripts

| File | What it does |
|------|-------------|
| `spectral_invariance.py` | 11-encoding pipeline with bootstrap CIs and finite-size calibration |
| `spectral_neural_map.py` | Internal spectroscopy — weight and activation RMT per layer |
| `spectral_scale_test.py` | Cross-model scale comparison (7B vs 72B) |
| `run_spectral_arxiv.py` | Embedding-level spectral taxonomy on arXiv corpus |
| `run_spectral_arxiv_multiK.py` | K-stability analysis |
| `bootstrap_ci.py` | Bootstrap confidence intervals for ⟨r⟩ |
| `fetch_arxiv_corpus.py` | Fetch corpus via arXiv API |
| `curate_corpus.py` | Build 10K corpus from Kaggle arXiv dump (25 categories) |

## Dependencies

numpy, scipy, scikit-learn. torch for internal spectroscopy only.

## Data

B200 burst results (Qwen 72B, Llama, Qwen 7B, Mistral 7B) available on request. ArXiv corpus IDs included for reproducibility.

## Author

Joseph Hayden — shadow@shdwcorp.cloud
EOF

cd ~/spectral-universality-repo
git add README.md
git commit -m "add README"
git push
