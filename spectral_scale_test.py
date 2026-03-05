#!/usr/bin/env python3
"""
Scale Test: 7B/8B models — is α separation architectural or capacity-driven?

If Qwen-7B shows α divergence and Llama-8B doesn't → architectural
If neither shows it → capacity effect (needs 72B+)
If both show it → universal at all scales

Uses same 4-domain × 30-input protocol as expanded analysis.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Import the analysis function from expanded script
sys.path.insert(0, os.path.expanduser("~"))
from spectral_expanded import DOMAIN_INPUTS, analyze_activation


def run_model(model_name, output_dir, label):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{label}_profiles.jsonl")

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({label})...")
    print(f"{'='*60}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    model.eval()
    load_time = time.time() - t0
    n_layers = model.config.num_hidden_layers
    print(f"Loaded in {load_time:.1f}s — {n_layers} layers")

    all_profiles = []
    with open(results_path, 'w') as out_f:
        for domain_name, texts in DOMAIN_INPUTS.items():
            print(f"  Domain: {domain_name} ({len(texts)} inputs)")
            domain_t0 = time.time()

            for input_idx, text in enumerate(texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512, padding=False).to("cuda")

                with torch.no_grad():
                    outputs = model(**inputs)

                hidden_states = outputs.hidden_states

                for layer_idx, h in enumerate(hidden_states):
                    if layer_idx >= n_layers:
                        break

                    act = h.squeeze(0)
                    profile = analyze_activation(act, layer_idx)
                    if profile is None:
                        continue

                    profile["domain"] = domain_name
                    profile["input_idx"] = int(input_idx)
                    profile["model"] = label
                    all_profiles.append(profile)
                    out_f.write(json.dumps(profile) + "\n")

                del outputs, hidden_states
                torch.cuda.empty_cache()

                if (input_idx + 1) % 10 == 0:
                    elapsed = time.time() - domain_t0
                    print(f"    [{input_idx+1}/{len(texts)}] {elapsed:.1f}s")

            domain_elapsed = time.time() - domain_t0
            print(f"    Complete: {domain_elapsed:.1f}s")
            out_f.flush()

    print(f"\nTotal profiles: {len(all_profiles)}")

    # Quick stats
    from scipy import stats as sp_stats
    domains = {}
    for p in all_profiles:
        d = p["domain"]
        if d not in domains:
            domains[d] = []
        domains[d].append(p)

    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")

    for d in sorted(domains.keys()):
        alphas = np.array([p["alpha"] for p in domains[d]])
        print(f"  {d:20s}: N={len(alphas):5d}, α={alphas.mean():.4f} ± {alphas.std():.4f}")

    # Key comparison: MFG vs Random
    mfg = np.array([p["alpha"] for p in domains["manufacturing"]])
    rand = np.array([p["alpha"] for p in domains["random"]])
    mat = np.array([p["alpha"] for p in domains["materials_science"]])

    t_mr, p_mr = sp_stats.ttest_ind(mfg, rand, equal_var=False)
    t_mm, p_mm = sp_stats.ttest_ind(mfg, mat, equal_var=False)

    d_mr = (mfg.mean() - rand.mean()) / np.sqrt((mfg.std()**2 + rand.std()**2) / 2)
    d_mm = (mfg.mean() - mat.mean()) / np.sqrt((mfg.std()**2 + mat.std()**2) / 2)

    print(f"\n  MFG vs Random:    Δα={mfg.mean()-rand.mean():+.4f}, d={d_mr:+.3f}, p={p_mr:.2e}")
    print(f"  MFG vs MatSci:    Δα={mfg.mean()-mat.mean():+.4f}, d={d_mm:+.3f}, p={p_mm:.2e}")

    sig_mr = "YES" if p_mr < 0.001 else "MARGINAL" if p_mr < 0.05 else "NO"
    sig_mm = "YES" if p_mm < 0.001 else "MARGINAL" if p_mm < 0.05 else "NO"
    print(f"  Significant (MFG vs Random): {sig_mr}")
    print(f"  Significant (MFG vs MatSci): {sig_mm}")

    # Cleanup model from GPU
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"\n  GPU freed.")

    return all_profiles


def main():
    output_dir = os.path.expanduser("~/spectral_scale_test")
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("Qwen/Qwen2.5-7B", "qwen_7b"),
        ("mistralai/Mistral-7B-v0.3", "mistral_7b"),
        # Llama <70B all gated (403). Mistral-7B is the alt architecture:
        # sliding window attention, different head config, same parameter class.
    ]

    all_results = {}
    for model_name, label in models:
        try:
            profiles = run_model(model_name, output_dir, label)
            all_results[label] = profiles
        except Exception as e:
            print(f"ERROR on {label}: {e}")
            import traceback; traceback.print_exc()

    # Comparative summary
    print(f"\n{'='*60}")
    print(f"COMPARATIVE SUMMARY — Scale Test")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'Δα(MFG-Rand)':>14s} {'d(MFG-Rand)':>14s} {'Δα(MFG-Mat)':>14s} {'d(MFG-Mat)':>14s}")

    for label, profiles in all_results.items():
        domains = {}
        for p in profiles:
            d = p["domain"]
            if d not in domains:
                domains[d] = []
            domains[d].append(p)

        mfg = np.array([p["alpha"] for p in domains["manufacturing"]])
        rand = np.array([p["alpha"] for p in domains["random"]])
        mat = np.array([p["alpha"] for p in domains["materials_science"]])

        d_mr = (mfg.mean() - rand.mean()) / np.sqrt((mfg.std()**2 + rand.std()**2) / 2)
        d_mm = (mfg.mean() - mat.mean()) / np.sqrt((mfg.std()**2 + mat.std()**2) / 2)

        print(f"  {label:18s} {mfg.mean()-rand.mean():+14.4f} {d_mr:+14.3f} {mfg.mean()-mat.mean():+14.4f} {d_mm:+14.3f}")

    # Save combined summary
    summary = {
        "models": {label: len(profiles) for label, profiles in all_results.items()},
        "test": "scale_comparison",
        "hypothesis": "architectural_vs_capacity",
    }
    with open(os.path.join(output_dir, "scale_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
