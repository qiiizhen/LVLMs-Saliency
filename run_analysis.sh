#!/bin/bash
#SBATCH --job-name=saliency_analysis
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --account=rc_classwork17_pi
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

cd /work/users/z/h/zhenqi

PY=/nas/longleaf/home/zhenqi/.conda/envs/lvlm-saliency/bin/python

echo "=== analyze_from_png.py ==="
$PY LVLMs-Saliency/analyze_from_png.py

echo ""
echo "=== analyze_ablation.py ==="
$PY LVLMs-Saliency/analyze_ablation.py

echo ""
echo "=== analyze_position_trend.py ==="
$PY LVLMs-Saliency/analyze_position_trend.py

echo ""
echo "=== analyze_case_study.py ==="
$PY LVLMs-Saliency/analyze_case_study.py

echo ""
echo "=== analyze_layer.py ==="
$PY LVLMs-Saliency/analyze_layer.py
