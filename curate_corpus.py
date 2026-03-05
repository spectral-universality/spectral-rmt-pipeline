#!/usr/bin/env python3
"""
arXiv Corpus Curator for Spectral Taxonomy Analysis
=====================================================
Curates a balanced, multi-domain corpus from the Kaggle arXiv dump
for RMT spectral analysis (⟨r⟩, α, soft rank diagnostics).

Usage:
  1. Download arxiv dump: kaggle datasets download -d Cornell-University/arxiv
  2. Unzip: unzip arxiv.zip  (produces arxiv-metadata-oai-snapshot.json)
  3. Run: python3 curate_corpus.py --source arxiv-metadata-oai-snapshot.json

Output: arxiv_spectral_corpus_10k.json — ready for burst compute

Categories selected for spectral diversity:
  - Predicted Poisson (self-contained, axiomatic)
  - Predicted GUE (entangled, cross-referential)
  - Predicted GOE (correlated, bounded)
  - Edge cases (meta-structure, ephemera targets)
  - Controls (rigid, procedural)
  - Domain-spanning (interdisciplinary, out-of-field)

Author: Shadow
Date: 2026-03-04
"""

import json
import sys
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Target Categories
# ═══════════════════════════════════════════════════════════════

CATEGORIES = {
    # --- Predicted Poisson (self-contained, axiomatic) ---
    'math.LO': {
        'label': 'Logic',
        'prediction': 'Poisson',
        'rationale': 'Formal systems, self-contained proofs',
        'target_n': 400,
    },
    'math.CO': {
        'label': 'Combinatorics',
        'prediction': 'Poisson',
        'rationale': 'Discrete, independent results',
        'target_n': 400,
    },
    'quant-ph': {
        'label': 'Quantum Physics',
        'prediction': 'Poisson',
        'rationale': 'Confirmed Poisson in original 747 run (0.473)',
        'target_n': 400,
    },
    'math.NA': {
        'label': 'Numerical Analysis',
        'prediction': 'Poisson',
        'rationale': 'Algorithmic recipes, self-contained methods',
        'target_n': 400,
    },

    # --- Predicted GUE (entangled, cross-referential) ---
    'hep-th': {
        'label': 'High-Energy Theory',
        'prediction': 'GUE',
        'rationale': 'Confirmed GUE in original run (0.627). String theory, AdS/CFT.',
        'target_n': 400,
    },
    'cond-mat.str-el': {
        'label': 'Strongly Correlated Electrons',
        'prediction': 'GUE',
        'rationale': 'Dense theoretical web, heavy cross-referencing',
        'target_n': 400,
    },
    'gr-qc': {
        'label': 'General Relativity / Quantum Cosmology',
        'prediction': 'GUE',
        'rationale': 'Cross-references QFT, cosmology, differential geometry',
        'target_n': 400,
    },
    'math.AG': {
        'label': 'Algebraic Geometry',
        'prediction': 'GUE',
        'rationale': 'Notoriously interconnected, deep cross-references',
        'target_n': 400,
    },

    # --- Predicted GOE (correlated but bounded) ---
    'cs.LG': {
        'label': 'Machine Learning',
        'prediction': 'GOE',
        'rationale': 'Confirmed GOE in original run (0.532)',
        'target_n': 400,
    },
    'stat.ML': {
        'label': 'Statistical ML',
        'prediction': 'GOE',
        'rationale': 'Confirmed GOE in original run',
        'target_n': 400,
    },
    'cond-mat.stat-mech': {
        'label': 'Statistical Mechanics',
        'prediction': 'GOE',
        'rationale': 'Confirmed GOE in original run (0.548)',
        'target_n': 400,
    },
    'physics.bio-ph': {
        'label': 'Biological Physics',
        'prediction': 'GOE',
        'rationale': 'Confirmed GOE in original run (0.533)',
        'target_n': 400,
    },

    # --- Edge cases / ephemera targets ---
    'cs.CL': {
        'label': 'Computational Linguistics',
        'prediction': 'unknown',
        'rationale': 'Papers ABOUT language, embedded AS language. Meta-structure.',
        'target_n': 400,
    },
    'math.SP': {
        'label': 'Spectral Theory',
        'prediction': 'GUE?',
        'rationale': 'Papers about eigenvalues, analyzed by eigenvalues. Original: 0.570.',
        'target_n': 400,  # may have fewer available
    },
    'cs.AI': {
        'label': 'Artificial Intelligence',
        'prediction': 'unknown',
        'rationale': 'Sprawling, references neuroscience to philosophy. May fragment.',
        'target_n': 400,
    },
    'nlin.AO': {
        'label': 'Adaptation & Self-Organizing Systems',
        'prediction': 'unknown',
        'rationale': 'Small community, could go any direction',
        'target_n': 400,  # may have fewer available
    },
    'physics.hist-ph': {
        'label': 'History of Physics',
        'prediction': 'unknown',
        'rationale': 'Narratives about science, not science itself',
        'target_n': 400,  # may have fewer available
    },
    'math.NT': {
        'label': 'Number Theory',
        'prediction': 'GOE',
        'rationale': 'Original: 0.523. May split into deep/shallow subcommunities.',
        'target_n': 400,
    },

    # --- Controls ---
    'cs.DS': {
        'label': 'Data Structures & Algorithms',
        'prediction': 'Poisson',
        'rationale': 'Rigid, procedural, independent results',
        'target_n': 400,
    },
    'math.PR': {
        'label': 'Probability',
        'prediction': 'Poisson',
        'rationale': 'Foundational, relatively self-contained',
        'target_n': 400,
    },
    'astro-ph.CO': {
        'label': 'Cosmology (Observational)',
        'prediction': 'GOE',
        'rationale': 'Data-driven, different from theoretical physics',
        'target_n': 400,
    },
    'cs.CR': {
        'label': 'Cryptography',
        'prediction': 'Poisson',
        'rationale': 'Tight community, formal results',
        'target_n': 400,
    },

    # --- Domain-spanning ---
    'physics.med-ph': {
        'label': 'Medical Physics',
        'prediction': 'unknown',
        'rationale': 'Applied, interdisciplinary',
        'target_n': 400,  # may have fewer
    },
    'q-bio.MN': {
        'label': 'Molecular Networks',
        'prediction': 'unknown',
        'rationale': 'Connects to protein RMT work',
        'target_n': 400,  # may have fewer
    },
    'econ.TH': {
        'label': 'Economic Theory',
        'prediction': 'unknown',
        'rationale': 'Completely different field, tests domain-agnosticism',
        'target_n': 400,  # may have fewer
    },
    'stat.TH': {
        'label': 'Statistical Theory',
        'prediction': 'Poisson',
        'rationale': 'Formal results about statistics itself',
        'target_n': 400,
    },
}


# ═══════════════════════════════════════════════════════════════
# Corpus Curation
# ═══════════════════════════════════════════════════════════════

def get_primary_category(paper):
    """Extract primary category from arXiv metadata."""
    # Kaggle dump has 'categories' as space-separated string
    # and sometimes 'primary_category' field
    if 'primary_category' in paper and paper['primary_category']:
        return paper['primary_category']
    cats = paper.get('categories', '').split()
    return cats[0] if cats else None


def has_valid_abstract(paper):
    """Check if paper has a usable abstract."""
    abstract = paper.get('abstract', '').strip()
    if not abstract:
        return False
    if len(abstract) < 100:  # too short to be real
        return False
    if abstract.lower().startswith('this paper has been withdrawn'):
        return False
    return True


def curate(source_path, output_path, seed=42):
    """Main curation pipeline."""
    rng = random.Random(seed)
    target_cats = set(CATEGORIES.keys())

    # --- Pass 1: Collect candidates ---
    print(f'Scanning {source_path} ...')
    candidates = defaultdict(list)
    total_scanned = 0
    total_matched = 0

    with open(source_path) as f:
        for line in f:
            total_scanned += 1
            if total_scanned % 500000 == 0:
                print(f'  Scanned {total_scanned:,} papers, {total_matched:,} matched ...')

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            primary_cat = get_primary_category(paper)
            if primary_cat not in target_cats:
                continue

            if not has_valid_abstract(paper):
                continue

            # Keep only what we need
            candidates[primary_cat].append({
                'id': paper.get('id', ''),
                'title': paper.get('title', '').replace('\n', ' ').strip(),
                'abstract': paper.get('abstract', '').replace('\n', ' ').strip(),
                'primary_category': primary_cat,
                'categories': paper.get('categories', ''),
                'update_date': paper.get('update_date', ''),
            })
            total_matched += 1

    print(f'\nScan complete: {total_scanned:,} papers, {total_matched:,} in target categories')

    # --- Pass 2: Sample ---
    corpus = []
    stats = {}

    print(f'\n{"Category":<25} {"Available":>10} {"Target":>8} {"Sampled":>8}  {"Prediction":<10}')
    print('-' * 70)

    for cat in sorted(CATEGORIES.keys()):
        info = CATEGORIES[cat]
        available = candidates[cat]
        target_n = info['target_n']

        if len(available) <= target_n:
            sampled = available
        else:
            sampled = rng.sample(available, target_n)

        # Add text field for pipeline compatibility
        for doc in sampled:
            doc['text'] = doc['title'] + '. ' + doc['abstract']

        corpus.extend(sampled)
        stats[cat] = {
            'label': info['label'],
            'available': len(available),
            'sampled': len(sampled),
            'prediction': info['prediction'],
        }

        print(f'{cat:<25} {len(available):>10,} {target_n:>8} {len(sampled):>8}  {info["prediction"]:<10}')

    # Shuffle corpus
    rng.shuffle(corpus)

    # --- Save ---
    output = {
        'metadata': {
            'source': str(source_path),
            'n_categories': len(CATEGORIES),
            'n_documents': len(corpus),
            'seed': seed,
            'curation_date': '2026-03-04',
            'purpose': 'RMT spectral taxonomy analysis — 10K corpus',
            'category_stats': stats,
            'category_definitions': {k: v['rationale'] for k, v in CATEGORIES.items()},
        },
        'documents': corpus,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=1)

    print(f'\n{"="*70}')
    print(f'Corpus saved: {output_path}')
    print(f'Total documents: {len(corpus):,}')
    print(f'Categories: {len(stats)}')

    # Warnings for small categories
    for cat, s in stats.items():
        if s['sampled'] < 100:
            print(f'  ⚠ {cat} ({s["label"]}): only {s["sampled"]} papers — may be underpowered')

    print(f'\nCorpus size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB')

    return stats


# ═══════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curate arXiv corpus for spectral analysis')
    parser.add_argument('--source', required=True,
                        help='Path to arxiv-metadata-oai-snapshot.json (Kaggle dump)')
    parser.add_argument('--output', default='arxiv_spectral_corpus_10k.json',
                        help='Output corpus path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling')
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f'ERROR: Source not found: {args.source}')
        print(f'\nDownload from Kaggle:')
        print(f'  pip install kaggle')
        print(f'  kaggle datasets download -d Cornell-University/arxiv')
        print(f'  unzip arxiv.zip')
        sys.exit(1)

    curate(args.source, args.output, args.seed)
