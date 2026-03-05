#!/usr/bin/env python3
"""
Spectral Neural Map — RMT Analysis of Transformer Weight Matrices & Activations

Analyzes pretrained transformer models using Random Matrix Theory:
  1. Weight matrix SVD per layer (Q, K, V, O, QK^T per head)
  2. Three-regime <r> spacing ratio (lower outliers, bulk, upper outliers)
  3. Jha-Reagen MP metrics (Gap, outlier count, outlier energy, soft rank)
  4. Liu-Paquette-Sous α exponents on activation covariance
  5. Domain-conditioned spectral analysis (manufacturing vs random vs code)

References:
  - Staats, Thamm & Rosenow (2410.17770) — small SVs matter, both-tail deviations
  - Jha & Reagen (2507.09394) — MP diagnostics for W_Q W_K^T
  - Liu, Paquette & Sous (OPT2025) — spectral dimension evolution across layers
  - Nait Saada, Naderi & Tanner (2410.07799) — spectral gap in attention matrices

Target hardware: NVIDIA B300 288GB SXM6
Target model: Llama 3.1 70B (FP8), ~35GB VRAM

Authors: Shadow + Dreadbot 3.2.666
Date: 2026-02-23
"""

import numpy as np
import json
import os
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# §0 — Hardware Onboarding & Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HardwareProfile:
    """Auto-detected hardware configuration for optimal pipeline execution."""
    gpu_name: str = "unknown"
    gpu_vram_gb: float = 0.0
    compute_capability: tuple = (0, 0)
    sm_count: int = 0
    cpu_count: int = 0
    ram_gb: int = 0
    disk_free_gb: int = 0

    # Derived optimal settings
    model_dtype: str = "float32"      # Load dtype for model weights
    compute_dtype: str = "float32"    # Matmul dtype
    eigvalsh_dtype: str = "float32"   # Always float32 (LAPACK minimum)
    max_model_params_b: float = 0.0   # Max model size in billions
    activation_batch_size: int = 2     # Batch size for activation capture
    sample_heads_per_layer: int = 4    # W_QK heads to sample
    use_gpu_eigvalsh: bool = False     # GPU LAPACK vs CPU fallback

    def __post_init__(self):
        """Auto-detect hardware and set optimal configuration."""
        self._detect_gpu()
        self._detect_system()
        self._configure()

    def _detect_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.gpu_name = props.name
                self.gpu_vram_gb = round(props.total_memory / 1e9, 1)
                self.compute_capability = (props.major, props.minor)
                self.sm_count = props.multi_processor_count
        except Exception:
            pass

    def _detect_system(self):
        self.cpu_count = os.cpu_count() or 1
        try:
            import subprocess
            mem = subprocess.check_output("free -g", shell=True).decode()
            for line in mem.split('\n'):
                if line.startswith('Mem:'):
                    parts = line.split()
                    self.ram_gb = int(parts[1])
            df = subprocess.check_output("df -BG / | tail -1", shell=True).decode().split()
            self.disk_free_gb = int(df[3].rstrip('G'))
        except Exception:
            pass

    def _configure(self):
        cc = self.compute_capability
        vram = self.gpu_vram_gb

        # Dtype selection based on compute capability
        if cc >= (8, 0):
            self.model_dtype = "bfloat16"
            self.compute_dtype = "bfloat16"
        elif cc >= (7, 0):
            self.model_dtype = "float16"
            self.compute_dtype = "float16"
        else:
            self.model_dtype = "float32"
            self.compute_dtype = "float32"

        # LAPACK always needs float32 minimum
        self.eigvalsh_dtype = "float32"

        # GPU eigvalsh available on most CUDA devices
        self.use_gpu_eigvalsh = cc >= (7, 0) and vram > 4.0

        # Model size budget: 85% of VRAM, 2 bytes per param for bf16
        bytes_per_param = 2 if self.model_dtype in ("bfloat16", "float16") else 4
        self.max_model_params_b = round(vram * 0.85 / bytes_per_param, 1)

        # Batch size for activation capture
        if vram > 150:
            self.activation_batch_size = 10
        elif vram > 80:
            self.activation_batch_size = 6
        elif vram > 40:
            self.activation_batch_size = 4
        else:
            self.activation_batch_size = 2

        # Head sampling budget
        self.sample_heads_per_layer = 8 if vram > 100 else 4

    def report(self) -> str:
        lines = [
            "=" * 60,
            "HARDWARE PROFILE",
            "=" * 60,
            f"  GPU:         {self.gpu_name}",
            f"  VRAM:        {self.gpu_vram_gb} GB",
            f"  Compute:     sm_{self.compute_capability[0]}{self.compute_capability[1]} ({self.sm_count} SMs)",
            f"  CPU:         {self.cpu_count} cores",
            f"  RAM:         {self.ram_gb} GB",
            f"  Disk free:   {self.disk_free_gb} GB",
            "",
            "  OPTIMAL CONFIG:",
            f"  Model dtype:     {self.model_dtype}",
            f"  Compute dtype:   {self.compute_dtype}",
            f"  Eigvalsh dtype:  {self.eigvalsh_dtype} (LAPACK minimum)",
            f"  GPU eigvalsh:    {self.use_gpu_eigvalsh}",
            f"  Max model:       {self.max_model_params_b}B params",
            f"  Batch size:      {self.activation_batch_size}",
            f"  Heads/layer:     {self.sample_heads_per_layer}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# §1 — Marchenko-Pastur Reference Distribution
# ═══════════════════════════════════════════════════════════════════

def marchenko_pastur_edges(eigenvalues: np.ndarray, gamma: float) -> tuple:
    """
    Compute Marchenko-Pastur distribution edges.

    For a random matrix with aspect ratio γ = N/M:
      λ₊ = σ² (1 + √γ)²
      λ₋ = σ² (1 - √γ)²

    We estimate σ² from the bulk of the spectrum (median-based,
    robust to outliers at both tails per Staats et al.).

    Returns: (lambda_minus, lambda_plus, sigma_sq, gamma)
    """
    # Robust variance estimate from bulk (median of eigenvalues)
    # This avoids contamination from outliers at both ends
    bulk_median = np.median(eigenvalues[eigenvalues > 0])
    # For MP distribution, median ≈ σ²(1 - γ)² + correction
    # Simpler: use trimmed mean of middle 50%
    sorted_eigs = np.sort(eigenvalues[eigenvalues > 0])
    n = len(sorted_eigs)
    q1, q3 = n // 4, 3 * n // 4
    sigma_sq = np.mean(sorted_eigs[q1:q3]) / (1.0 + gamma)  # approximate

    lambda_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma_sq * (1 - np.sqrt(gamma)) ** 2

    return lambda_minus, lambda_plus, sigma_sq, gamma


def partition_eigenvalues(eigenvalues: np.ndarray, gamma: float):
    """
    Partition eigenvalues into three regimes (Staats et al.):
      - lower outliers: λ < λ₋  (learned structure in small SVs)
      - bulk:           λ₋ ≤ λ ≤ λ₊  (random/noise)
      - upper outliers: λ > λ₊  (learned structure in large SVs)

    Returns dict with arrays and MP edges.
    """
    eigs = np.sort(eigenvalues[eigenvalues > 1e-10])  # remove numerical zeros
    lam_minus, lam_plus, sigma_sq, _ = marchenko_pastur_edges(eigs, gamma)

    lower = eigs[eigs < lam_minus]
    bulk = eigs[(eigs >= lam_minus) & (eigs <= lam_plus)]
    upper = eigs[eigs > lam_plus]

    return {
        "lower_outliers": lower,
        "bulk": bulk,
        "upper_outliers": upper,
        "lambda_minus": float(lam_minus),
        "lambda_plus": float(lam_plus),
        "sigma_sq": float(sigma_sq),
    }


# ═══════════════════════════════════════════════════════════════════
# §2 — Spacing Statistics (<r> ratio)
# ═══════════════════════════════════════════════════════════════════

def mean_spacing_ratio(eigenvalues: np.ndarray) -> Optional[float]:
    """
    Compute mean spacing ratio <r> (Oganesyan-Huse).

    For sorted eigenvalues {λᵢ}, spacing sᵢ = λᵢ₊₁ - λᵢ,
    ratio rᵢ = min(sᵢ, sᵢ₊₁) / max(sᵢ, sᵢ₊₁).

    Reference values:
      Poisson: <r> ≈ 0.386 (independent, crystallized)
      GOE:     <r> ≈ 0.531 (correlated, real symmetric)
      GUE:     <r> ≈ 0.603 (maximally correlated, complex Hermitian)

    Returns None if insufficient eigenvalues (<4).
    """
    if len(eigenvalues) < 4:
        return None

    sorted_eigs = np.sort(eigenvalues)
    spacings = np.diff(sorted_eigs)

    # Remove zero spacings (degenerate eigenvalues)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 3:
        return None

    r_values = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_values))


def three_regime_spacing(eigenvalues: np.ndarray, gamma: float) -> dict:
    """
    Compute <r> for each of the three spectral regimes.
    """
    parts = partition_eigenvalues(eigenvalues, gamma)

    return {
        "r_lower": mean_spacing_ratio(parts["lower_outliers"]),
        "r_bulk": mean_spacing_ratio(parts["bulk"]),
        "r_upper": mean_spacing_ratio(parts["upper_outliers"]),
        "n_lower": len(parts["lower_outliers"]),
        "n_bulk": len(parts["bulk"]),
        "n_upper": len(parts["upper_outliers"]),
        "lambda_minus": parts["lambda_minus"],
        "lambda_plus": parts["lambda_plus"],
    }


# ═══════════════════════════════════════════════════════════════════
# §3 — Jha-Reagen MP Metrics
# ═══════════════════════════════════════════════════════════════════

def mp_diagnostics(eigenvalues: np.ndarray, gamma: float) -> dict:
    """
    Four MP diagnostic metrics from Jha & Reagen (2507.09394):
      1. MP-Gap:            λ_max / λ₊  (distance from largest to MP edge)
      2. Outlier count:     # eigenvalues outside [λ₋, λ₊]
      3. Outlier energy:    Σ λ_outlier / Σ λ_all  (fraction of spectral energy)
      4. Soft rank:         (Σ λᵢ)² / Σ λᵢ²  (effective dimensionality)
    """
    eigs = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]  # descending
    lam_minus, lam_plus, _, _ = marchenko_pastur_edges(eigs, gamma)

    # MP-Gap
    mp_gap = float(eigs[0] / lam_plus) if lam_plus > 0 else float('inf')

    # Outlier count (both tails)
    outlier_mask = (eigs > lam_plus) | (eigs < lam_minus)
    outlier_count = int(np.sum(outlier_mask))

    # Outlier energy fraction
    total_energy = np.sum(eigs)
    outlier_energy = np.sum(eigs[outlier_mask])
    outlier_energy_frac = float(outlier_energy / total_energy) if total_energy > 0 else 0.0

    # Soft rank (effective dimensionality)
    soft_rank = float(np.sum(eigs) ** 2 / np.sum(eigs ** 2)) if np.sum(eigs ** 2) > 0 else 0.0

    # Stable rank (for comparison)
    stable_rank = float(np.sum(eigs) / eigs[0]) if eigs[0] > 0 else 0.0

    return {
        "mp_gap": mp_gap,
        "outlier_count": outlier_count,
        "outlier_energy_fraction": outlier_energy_frac,
        "soft_rank": soft_rank,
        "stable_rank": stable_rank,
        "lambda_max": float(eigs[0]),
        "lambda_plus": float(lam_plus),
        "lambda_minus": float(lam_minus),
    }


# ═══════════════════════════════════════════════════════════════════
# §4 — Heavy-Tail Exponent (α) — Liu, Paquette & Sous
# ═══════════════════════════════════════════════════════════════════

def heavy_tail_exponent(eigenvalues: np.ndarray, fit_range: tuple = (0.1, 0.9)) -> dict:
    """
    Estimate power-law heavy-tail exponent α from eigenvalue distribution.

    For eigenvalues sorted descending, if they follow a power law:
      λ_k ∝ k^(-1/α)

    Then log(λ_k) vs log(k) has slope -1/α.

    Liu et al. conjecture α increases monotonically across layers
    (shallow ~0.70, deep ~0.90 after fine-tuning), tracking the
    GUE→Poisson transition as representations compress.

    fit_range: fraction of eigenvalue ranks to fit (exclude head/tail artifacts)
    """
    eigs = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
    n = len(eigs)
    if n < 10:
        return {"alpha": None, "r_squared": None}

    # Rank indices (1-based)
    ranks = np.arange(1, n + 1)

    # Fit in middle range to avoid edge effects
    i_start = max(1, int(n * fit_range[0]))
    i_end = min(n, int(n * fit_range[1]))

    log_ranks = np.log(ranks[i_start:i_end])
    log_eigs = np.log(eigs[i_start:i_end])

    # Linear regression: log(λ) = -1/α * log(k) + const
    # Slope = -1/α → α = -1/slope
    coeffs = np.polyfit(log_ranks, log_eigs, 1)
    slope = coeffs[0]

    alpha = -1.0 / slope if abs(slope) > 1e-10 else float('inf')

    # R² for fit quality
    predicted = np.polyval(coeffs, log_ranks)
    ss_res = np.sum((log_eigs - predicted) ** 2)
    ss_tot = np.sum((log_eigs - np.mean(log_eigs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": float(alpha),
        "slope": float(slope),
        "r_squared": float(r_squared),
    }


# ═══════════════════════════════════════════════════════════════════
# §5 — Complete Spectral Profile (per matrix)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpectralProfile:
    """Complete spectral analysis of a single matrix."""
    layer: int
    head: Optional[int]
    matrix_type: str  # "W_Q", "W_K", "W_V", "W_O", "W_QK", "activation_cov"
    shape: tuple
    gamma: float

    # Three-regime <r>
    r_lower: Optional[float] = None
    r_bulk: Optional[float] = None
    r_upper: Optional[float] = None
    n_lower: int = 0
    n_bulk: int = 0
    n_upper: int = 0

    # MP diagnostics (Jha-Reagen)
    mp_gap: float = 0.0
    outlier_count: int = 0
    outlier_energy_fraction: float = 0.0
    soft_rank: float = 0.0
    stable_rank: float = 0.0

    # Heavy-tail exponent (Liu et al.)
    alpha: Optional[float] = None
    alpha_r2: Optional[float] = None

    # MP edges
    lambda_max: float = 0.0
    lambda_plus: float = 0.0
    lambda_minus: float = 0.0

    # Domain conditioning (if activation)
    domain: Optional[str] = None  # "manufacturing" | "random" | "code" | None (weights)

    # Timing
    compute_seconds: float = 0.0


def analyze_matrix(matrix, layer: int, head: Optional[int],
                   matrix_type: str, domain: Optional[str] = None,
                   device: str = "cuda") -> SpectralProfile:
    """
    Run full spectral analysis on a single matrix.

    For weight matrices: compute W @ W.T eigenvalues (Gram matrix).
    For activation covariance: already symmetric, eigendecompose directly.

    GPU-native path: matmul in native dtype (bf16), promote to float32
    for eigvalsh (LAPACK requirement), pull only eigenvalues to CPU.
    """
    import torch

    t0 = time.time()

    # Accept both numpy arrays and torch tensors
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    # Move to GPU if not already there
    if matrix.device.type != device and device == "cuda":
        matrix = matrix.to(device)

    m, n = matrix.shape
    gamma = min(m, n) / max(m, n)

    # Compute eigenvalues of the Gram matrix — GPU-native
    if matrix_type == "activation_cov":
        # Already a covariance matrix — eigendecompose directly
        gram = matrix.float()  # eigvalsh needs float32 minimum
    elif m >= n:
        # W^T W (n×n, faster when m > n) — matmul in native dtype (bf16)
        gram = (matrix.T @ matrix).float()
    else:
        # W W^T (m×m, faster when n > m)
        gram = (matrix @ matrix.T).float()

    # GPU eigvalsh — LAPACK on CUDA, no CPU roundtrip
    eigenvalues = torch.linalg.eigvalsh(gram)

    # Only NOW pull to CPU (tiny vector, not the full matrix)
    eigenvalues = torch.abs(eigenvalues).cpu().numpy()
    eigenvalues = np.sort(eigenvalues)

    # Three-regime spacing
    spacing = three_regime_spacing(eigenvalues, gamma)

    # Jha-Reagen MP diagnostics
    mp = mp_diagnostics(eigenvalues, gamma)

    # Heavy-tail exponent
    ht = heavy_tail_exponent(eigenvalues)

    elapsed = time.time() - t0

    return SpectralProfile(
        layer=layer, head=head, matrix_type=matrix_type,
        shape=(m, n), gamma=gamma,
        r_lower=spacing["r_lower"], r_bulk=spacing["r_bulk"],
        r_upper=spacing["r_upper"],
        n_lower=spacing["n_lower"], n_bulk=spacing["n_bulk"],
        n_upper=spacing["n_upper"],
        mp_gap=mp["mp_gap"], outlier_count=mp["outlier_count"],
        outlier_energy_fraction=mp["outlier_energy_fraction"],
        soft_rank=mp["soft_rank"], stable_rank=mp["stable_rank"],
        alpha=ht["alpha"], alpha_r2=ht["r_squared"],
        lambda_max=mp["lambda_max"],
        lambda_plus=mp["lambda_plus"], lambda_minus=mp["lambda_minus"],
        domain=domain, compute_seconds=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════
# §6 — Model Weight Extraction (Llama/Qwen/Mistral)
# ═══════════════════════════════════════════════════════════════════

def extract_weight_profiles(model_name: str, output_dir: str,
                            device: str = "cuda", dtype_str: str = "float16",
                            hw: HardwareProfile = None):
    """
    Load a pretrained model, extract weight matrices per layer,
    run full spectral analysis on each.

    Analyzes per layer:
      - W_Q, W_K, W_V, W_O (individual projection matrices)
      - W_Q @ W_K^T per head (attention Gram matrix — Jha & Reagen)

    Saves results as JSONL for downstream analysis.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    # Use hardware profile if available
    if hw is None:
        hw = HardwareProfile()
    effective_dtype = dtype_str or hw.model_dtype

    print(f"Loading {model_name} on {device} ({effective_dtype})...")
    t0 = time.time()

    dtype = getattr(torch, effective_dtype)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = config.hidden_size // num_heads
    kv_group_size = num_heads // num_kv_heads

    print(f"Architecture: {num_layers} layers, {num_heads} Q heads, {num_kv_heads} KV heads, dim={config.hidden_size}, head_dim={head_dim}, GQA group={kv_group_size}")

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "weight_profiles.jsonl")
    profiles = []

    with open(results_path, 'w') as out_f:
        for layer_idx in range(num_layers):
            layer_t0 = time.time()

            # Get attention weight matrices — STAY ON GPU in native dtype
            # Llama/Qwen naming: model.layers[i].self_attn.{q,k,v,o}_proj.weight
            try:
                attn = model.model.layers[layer_idx].self_attn
                w_q = attn.q_proj.weight.detach()  # (num_heads*head_dim, hidden) on GPU, bf16
                w_k = attn.k_proj.weight.detach()  # (num_kv_heads*head_dim, hidden) on GPU, bf16
                w_v = attn.v_proj.weight.detach()  # (num_kv_heads*head_dim, hidden) on GPU, bf16
                w_o = attn.o_proj.weight.detach()  # (hidden, num_heads*head_dim) on GPU, bf16
            except AttributeError:
                print(f"  Layer {layer_idx}: attribute error, skipping")
                continue

            # Analyze individual weight matrices — GPU-native eigvalsh
            for mat, name in [(w_q, "W_Q"), (w_k, "W_K"), (w_v, "W_V"), (w_o, "W_O")]:
                profile = analyze_matrix(mat, layer_idx, None, name)
                profiles.append(profile)
                out_f.write(json.dumps(asdict(profile)) + "\n")

            # Analyze W_Q @ W_K^T per head (Jha-Reagen: where pathologies concentrate)
            # GQA: Q has num_heads, K has num_kv_heads. Each KV head serves kv_group_size Q heads.
            q_heads = w_q.reshape(num_heads, head_dim, -1)
            k_heads = w_k.reshape(num_kv_heads, head_dim, -1)

            for h in range(min(num_heads, hw.sample_heads_per_layer)):  # Sample heads per layer (hw-configured)
                kv_h = h // kv_group_size  # Map Q head to its KV group
                qk = q_heads[h] @ k_heads[kv_h].T  # head_dim × head_dim, bf16 matmul on GPU
                profile = analyze_matrix(qk, layer_idx, h, "W_QK")
                profiles.append(profile)
                out_f.write(json.dumps(asdict(profile)) + "\n")

            layer_elapsed = time.time() - layer_t0
            if (layer_idx + 1) % 10 == 0:
                print(f"  Layer {layer_idx+1}/{num_layers} ({layer_elapsed:.1f}s/layer)")
            out_f.flush()

    total_time = time.time() - t0
    print(f"\nWeight analysis complete: {len(profiles)} profiles in {total_time:.1f}s")
    print(f"Results: {results_path}")

    return profiles


# ═══════════════════════════════════════════════════════════════════
# §7 — Activation Capture & Domain-Conditioned Analysis
# ═══════════════════════════════════════════════════════════════════

# Domain-specific input batches
DOMAIN_INPUTS = {
    "manufacturing": [
        "What is the recommended feed rate for CNC milling 6061-T6 aluminum at 0.5mm depth of cut using a 4-flute carbide end mill?",
        "Explain the difference between G43 and G44 tool length compensation in Fanuc controls.",
        "What shielding gas mixture is best for GMAW welding of 304 stainless steel in the flat position?",
        "Describe the heat treatment process for 4140 steel to achieve 40 HRC hardness.",
        "How do you calculate the bend allowance for 16-gauge sheet metal with a 90-degree bend?",
        "What are the common causes of tool chatter in turning operations and how do you diagnose them?",
        "Explain Custom Macro B variable types and the difference between local and global variables.",
        "What is the Taylor tool life equation and how do you apply it to optimize cutting speed?",
        "Describe the injection molding process parameters that affect part warpage.",
        "What NDT method is most appropriate for detecting subsurface cracks in forged steel components?",
    ],
    "random": [
        "The weather in Paris during spring is typically mild with occasional rain showers.",
        "She walked through the garden, admiring the colorful flowers blooming along the path.",
        "The committee met on Tuesday to discuss the annual budget allocation for community programs.",
        "He ordered a coffee and sat by the window, watching people pass by on the street below.",
        "The documentary explored the migration patterns of monarch butterflies across North America.",
        "They decided to paint the living room a warm shade of terracotta for the renovation.",
        "The library hosted a reading group every Thursday evening for classic literature enthusiasts.",
        "She packed her suitcase carefully, making sure to include warm layers for the mountain trip.",
        "The local farmers market opens every Saturday morning with fresh produce and handmade goods.",
        "He spent the afternoon organizing his bookshelf by genre and then alphabetically by author.",
    ],
    "code": [
        "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])",
        "SELECT p.name, COUNT(o.id) AS order_count FROM products p LEFT JOIN orders o ON p.id = o.product_id GROUP BY p.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC;",
        "class Node: def __init__(self, val=0, left=None, right=None): self.val = val; self.left = left; self.right = right",
        "import torch; model = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))",
        "async function fetchData(url) { const response = await fetch(url); if (!response.ok) throw new Error(response.status); return response.json(); }",
        "fn fibonacci(n: u64) -> u64 { match n { 0 => 0, 1 => 1, _ => fibonacci(n-1) + fibonacci(n-2) } }",
        "CREATE TABLE embeddings (id SERIAL PRIMARY KEY, vector FLOAT8[] NOT NULL, metadata JSONB, created_at TIMESTAMPTZ DEFAULT NOW());",
        "docker run --gpus all -v /data:/workspace -p 8080:8080 ghcr.io/shdwdev/rag-pipeline:latest python3 run.py",
        "kubectl apply -f deployment.yaml && kubectl rollout status deployment/rag-api --timeout=120s",
        "git rebase -i HEAD~5 && git push --force-with-lease origin feature/spectral-observatory",
    ],
}


def capture_activation_profiles(model_name: str, output_dir: str,
                                device: str = "cuda", dtype_str: str = "float16",
                                max_layers: int = 80, hw: HardwareProfile = None):
    """
    Feed domain-specific inputs through the model, capture per-layer
    activation covariance matrices, compute spectral profiles.

    This is the novel contribution: domain-conditioned spectral analysis.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hw is None:
        hw = HardwareProfile()
    effective_dtype = dtype_str or hw.model_dtype

    dtype = getattr(torch, effective_dtype)
    print(f"Loading {model_name} for activation capture...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, output_hidden_states=True,
    )
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "activation_profiles.jsonl")
    profiles = []

    with open(results_path, 'w') as out_f:
        for domain_name, texts in DOMAIN_INPUTS.items():
            print(f"\n  Domain: {domain_name} ({len(texts)} inputs)")
            domain_t0 = time.time()

            # Tokenize batch
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

            for layer_idx, h in enumerate(hidden_states):
                if layer_idx >= max_layers:
                    break

                # h: (batch, seq_len, hidden_dim) — stays on GPU
                b, s, d = h.shape
                flat = h.reshape(b * s, d)  # still bf16 on GPU

                # Compute activation covariance matrix — GPU-native
                # Center the activations (bf16 matmul, promote for eigvalsh later)
                flat_centered = flat - flat.mean(dim=0, keepdim=True)
                # Covariance: (d × d) — bf16 matmul on GPU, fast
                cov = (flat_centered.T @ flat_centered) / (flat_centered.shape[0] - 1)

                # Full spectral analysis
                profile = analyze_matrix(
                    cov, layer_idx, None, "activation_cov", domain=domain_name
                )
                profiles.append(profile)
                out_f.write(json.dumps(asdict(profile)) + "\n")

            domain_elapsed = time.time() - domain_t0
            print(f"    {len(hidden_states)} layers analyzed in {domain_elapsed:.1f}s")
            out_f.flush()

            # Free GPU memory between domains
            del outputs, hidden_states
            torch.cuda.empty_cache()

    print(f"\nActivation analysis complete: {len(profiles)} profiles")
    print(f"Results: {results_path}")
    return profiles


# ═══════════════════════════════════════════════════════════════════
# §8 — Summary & Visualization Data
# ═══════════════════════════════════════════════════════════════════

def generate_summary(output_dir: str):
    """
    Load all profiles and generate summary statistics + visualization-ready data.

    Key plots to generate downstream:
      1. <r> vs layer depth (bulk, upper, lower — three lines)
      2. α vs layer depth (Liu et al. replication)
      3. <r> vs α scatter (are they anticorrelated?)
      4. MP-Gap vs layer depth
      5. Outlier energy fraction vs layer depth
      6. Domain-conditioned <r> trajectories (manufacturing vs random vs code)
    """
    weight_path = os.path.join(output_dir, "weight_profiles.jsonl")
    activation_path = os.path.join(output_dir, "activation_profiles.jsonl")

    summary = {
        "weight_profiles": [],
        "activation_profiles": [],
    }

    # Load weight profiles
    if os.path.exists(weight_path):
        with open(weight_path) as f:
            summary["weight_profiles"] = [json.loads(l) for l in f]

    # Load activation profiles
    if os.path.exists(activation_path):
        with open(activation_path) as f:
            summary["activation_profiles"] = [json.loads(l) for l in f]

    # Per-layer aggregates for weight matrices
    weight_by_layer = {}
    for p in summary["weight_profiles"]:
        layer = p["layer"]
        mtype = p["matrix_type"]
        if layer not in weight_by_layer:
            weight_by_layer[layer] = {}
        if mtype not in weight_by_layer[layer]:
            weight_by_layer[layer][mtype] = []
        weight_by_layer[layer][mtype].append(p)

    # Per-layer per-domain aggregates for activations
    act_by_layer_domain = {}
    for p in summary["activation_profiles"]:
        key = (p["layer"], p["domain"])
        act_by_layer_domain[key] = p

    # Generate trajectory data
    trajectories = {
        "layers": [],
        "r_bulk_WQ": [], "r_bulk_WK": [], "r_bulk_WQK": [],
        "alpha_WQ": [], "alpha_WK": [],
        "mp_gap_WQK": [], "outlier_energy_WQK": [],
        "r_bulk_act_manufacturing": [], "r_bulk_act_random": [], "r_bulk_act_code": [],
        "alpha_act_manufacturing": [], "alpha_act_random": [], "alpha_act_code": [],
    }

    max_layer = max((p["layer"] for p in summary["weight_profiles"]), default=0)
    for layer in range(max_layer + 1):
        trajectories["layers"].append(layer)

        # Weight matrix averages
        for mtype, key in [("W_Q", "WQ"), ("W_K", "WK"), ("W_QK", "WQK")]:
            profs = weight_by_layer.get(layer, {}).get(mtype, [])
            if profs:
                r_vals = [p["r_bulk"] for p in profs if p["r_bulk"] is not None]
                trajectories[f"r_bulk_{key}"].append(np.mean(r_vals) if r_vals else None)
                if mtype != "W_QK":
                    a_vals = [p["alpha"] for p in profs if p["alpha"] is not None]
                    trajectories[f"alpha_{key}"].append(np.mean(a_vals) if a_vals else None)
                if mtype == "W_QK":
                    mg_vals = [p["mp_gap"] for p in profs]
                    oe_vals = [p["outlier_energy_fraction"] for p in profs]
                    trajectories[f"mp_gap_{key}"].append(np.mean(mg_vals) if mg_vals else None)
                    trajectories[f"outlier_energy_{key}"].append(np.mean(oe_vals) if oe_vals else None)
            else:
                trajectories[f"r_bulk_{key}"].append(None)
                if mtype != "W_QK":
                    trajectories[f"alpha_{key}"].append(None)
                if mtype == "W_QK":
                    trajectories[f"mp_gap_{key}"].append(None)
                    trajectories[f"outlier_energy_{key}"].append(None)

        # Activation trajectories per domain
        for domain in ["manufacturing", "random", "code"]:
            ap = act_by_layer_domain.get((layer, domain))
            trajectories[f"r_bulk_act_{domain}"].append(ap["r_bulk"] if ap else None)
            trajectories[f"alpha_act_{domain}"].append(ap["alpha"] if ap else None)

    # Save summary
    summary_path = os.path.join(output_dir, "spectral_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "trajectories": trajectories,
            "stats": {
                "total_weight_profiles": len(summary["weight_profiles"]),
                "total_activation_profiles": len(summary["activation_profiles"]),
                "num_layers": max_layer + 1,
            }
        }, f, indent=2, default=str)

    print(f"Summary written: {summary_path}")
    return trajectories


# ═══════════════════════════════════════════════════════════════════
# §9 — CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Spectral Neural Map — RMT Analysis of Transformers")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="./spectral_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--weights-only", action="store_true",
                        help="Only analyze weight matrices (skip activations)")
    parser.add_argument("--activations-only", action="store_true",
                        help="Only analyze activations (skip weights)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only generate summary from existing results")

    args = parser.parse_args()

    if args.summary_only:
        generate_summary(args.output)
        return

    # §0: Hardware onboarding — auto-detect and configure
    hw = HardwareProfile()
    print(hw.report())

    # Save hardware profile alongside results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "hardware_profile.json"), 'w') as f:
        json.dump(hw.to_dict(), f, indent=2, default=str)

    # Override dtype from CLI if provided, else use hardware-detected optimal
    effective_dtype = args.dtype or hw.model_dtype

    if not args.activations_only:
        print("=" * 60)
        print("PHASE 1: Weight Matrix Spectral Analysis")
        print("=" * 60)
        extract_weight_profiles(args.model, args.output, args.device, effective_dtype, hw=hw)

    if not args.weights_only:
        print("\n" + "=" * 60)
        print("PHASE 2: Domain-Conditioned Activation Analysis")
        print("=" * 60)
        capture_activation_profiles(args.model, args.output, args.device, effective_dtype, hw=hw)

    print("\n" + "=" * 60)
    print("PHASE 3: Summary Generation")
    print("=" * 60)
    generate_summary(args.output)


if __name__ == "__main__":
    main()
