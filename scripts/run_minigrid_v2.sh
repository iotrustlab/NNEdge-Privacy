#!/usr/bin/env bash
set -e

# Mini-Grid v2 for imbalance and semantics fixes
# All runs: 7-class, fusion=attention, seed=1, sampler=class_uniform, warmup_epochs=5, select_by=macro_f1

BASE="source .venv/bin/activate && python -m games.STM.experiments --grid mini_v2 --classes 7 --seeds 1 --fusion attention --sampler class_uniform --warmup_epochs 5 --select_by macro_f1 --loss balanced_softmax --label_smoothing 0.0"

echo "🚀 Starting Mini-Grid v2 for imbalance and semantics fixes..."
echo "📊 All runs: 7-class, fusion=attention, seed=1, sampler=class_uniform, warmup_epochs=5, select_by=macro_f1"
echo "=" * 80

# 1) Imbalance ablation
echo "🔬 Running Imbalance Ablation Experiments..."
echo "   Run 1: base featureset with balanced_softmax"
$BASE --featureset base --adjacency identity --hierarchical false

echo "   Run 2: micro+cross featureset with balanced_softmax"
$BASE --featureset micro+cross --adjacency identity --hierarchical false

echo "   Run 3: micro+cross+semantics with balanced_softmax and α-gate"
$BASE --featureset micro+cross+semantics --adjacency fixed --hierarchical false --gate_alpha_init 0.1

echo "   Run 4: micro+cross+semantics+film with balanced_softmax and α-gate"
$BASE --featureset micro+cross+semantics+film --adjacency fixed --hierarchical false --gate_alpha_init 0.1

# 2) Graph sanity
echo "🔗 Running Graph Sanity Experiments..."
echo "   Run 5: identity adjacency"
$BASE --featureset micro+cross+semantics --adjacency identity --hierarchical false --gate_alpha_init 0.1

echo "   Run 6: fixed adjacency"
$BASE --featureset micro+cross+semantics --adjacency fixed --hierarchical false --gate_alpha_init 0.1

echo "   Run 7: learned adjacency"
$BASE --featureset micro+cross+semantics --adjacency learned --hierarchical false --gate_alpha_init 0.1

echo "   Run 8: hierarchical head"
$BASE --featureset micro+cross+semantics --adjacency fixed --hierarchical true --hier_lambda 0.3 --gate_alpha_init 0.1

echo "✅ Mini-Grid v2 completed!"
echo "📁 Results saved to: results/experiments_fixcheck_v2/"
echo "📊 Check results.csv and REPORT_FIXCHECK.md for analysis"
