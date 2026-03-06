#!/usr/bin/env python3
"""
Generate null-control corpora for spectral analysis.

Takes the curated corpus and produces:
  1. intra_shuffle — sentences scrambled within each abstract (breaks structure, keeps vocab)
  2. cross_shuffle — category labels randomly reassigned (same texts, wrong bins)
  3. random_baseline — pure random vectors (no text, generated at embed time)

Usage:
  python3 generate_controls.py --source ~/datasets/arxiv_spectral_corpus_10k.json \
                                --output-dir ~/datasets/controls/
"""

import argparse
import json
import random
import re
from pathlib import Path
from copy import deepcopy


def scramble_sentences(text: str) -> str:
    """Split text into sentences, shuffle order."""
    # Simple sentence split on ., !, ? followed by space+capital or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        # If only one sentence, shuffle words instead
        words = text.split()
        random.shuffle(words)
        return " ".join(words)
    random.shuffle(sentences)
    return " ".join(sentences)


def main():
    parser = argparse.ArgumentParser(description="Generate null controls")
    parser.add_argument("--source", required=True, help="Curated corpus JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.source) as f:
        data = json.load(f)

    papers = data.get("papers") or data.get("documents", [])
    papers_key = "papers" if "papers" in data else "documents"
    metadata = data.get("metadata", {})

    print(f"Source: {len(papers)} papers")

    # ── Control 1: Intra-category sentence shuffle ──────────────
    print("Generating intra-category shuffle...")
    intra = deepcopy(data)
    for paper in intra[papers_key]:
        paper["abstract"] = scramble_sentences(paper["abstract"])
        paper["title"] = scramble_sentences(paper["title"])
    intra["metadata"]["control"] = "intra_shuffle"
    intra["metadata"]["description"] = "Sentences scrambled within each abstract. Preserves vocabulary, breaks semantic structure."

    intra_path = out_dir / "control_intra_shuffle.json"
    with open(intra_path, "w") as f:
        json.dump(intra, f)
    print(f"  → {intra_path} ({intra_path.stat().st_size / 1e6:.1f} MB)")

    # ── Control 2: Cross-category label shuffle ─────────────────
    print("Generating cross-category shuffle...")
    cross = deepcopy(data)
    categories = [p["primary_category"] for p in cross[papers_key]]
    regimes = [p.get("target_regime", "") for p in cross[papers_key]]
    notes = [p.get("target_note", "") for p in cross[papers_key]]

    # Shuffle labels independently of texts
    combined = list(zip(categories, regimes, notes))
    random.shuffle(combined)
    for i, paper in enumerate(cross[papers_key]):
        paper["primary_category"] = combined[i][0]
        paper["target_regime"] = combined[i][1]
        paper["target_note"] = combined[i][2]
    cross["metadata"]["control"] = "cross_shuffle"
    cross["metadata"]["description"] = "Category labels randomly reassigned. Same texts, wrong bins."

    cross_path = out_dir / "control_cross_shuffle.json"
    with open(cross_path, "w") as f:
        json.dump(cross, f)
    print(f"  → {cross_path} ({cross_path.stat().st_size / 1e6:.1f} MB)")

    # ── Control 3: Word-level shuffle (nuclear option) ──────────
    print("Generating word shuffle (bag-of-words control)...")
    bow = deepcopy(data)
    for paper in bow[papers_key]:
        words = paper["abstract"].split()
        random.shuffle(words)
        paper["abstract"] = " ".join(words)
        tw = paper["title"].split()
        random.shuffle(tw)
        paper["title"] = " ".join(tw)
    bow["metadata"]["control"] = "word_shuffle"
    bow["metadata"]["description"] = "All words shuffled. Preserves word frequency, destroys ALL structure."

    bow_path = out_dir / "control_word_shuffle.json"
    with open(bow_path, "w") as f:
        json.dump(bow, f)
    print(f"  → {bow_path} ({bow_path.stat().st_size / 1e6:.1f} MB)")

    # ── Embed-format versions (JSONL for quick loading) ─────────
    print("\nGenerating embed JSONL files...")
    for name, corpus_data in [("intra_shuffle", intra), ("cross_shuffle", cross), ("word_shuffle", bow)]:
        jsonl_path = out_dir / f"control_{name}.embed.jsonl"
        with open(jsonl_path, "w") as f:
            for paper in corpus_data[papers_key]:
                f.write(json.dumps({
                    "id": paper["id"],
                    "text": paper["title"] + " " + paper["abstract"],
                    "category": paper["primary_category"],
                    "regime": paper.get("target_regime", ""),
                    "control": name,
                }) + "\n")
        print(f"  → {jsonl_path} ({jsonl_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDone. Controls in {out_dir}/")
    print(f"  Total files: 6 (3 full JSON + 3 embed JSONL)")
    print(f"\n  Upload all of {out_dir}/ to burst instance alongside the primary corpus.")


if __name__ == "__main__":
    main()
