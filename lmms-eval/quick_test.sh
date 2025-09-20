#!/bin/bash

# 快速参数敏感性测试
# 测试几个极端配置看是否有明显差异

cd /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval

echo "=== 快速参数敏感性测试 ==="
echo "目的：验证参数变化是否会影响性能"
echo ""

# 基础配置 (已知成功)
echo "[1/5] 测试基础配置 (已知成功)..."
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=8,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=0.5,score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=3.0,alea_threshold=3.0,epi_center=0.5,alea_center=0.5,exploit_bonus=2.0,explore_penalty=-0.2,balance_factor=0.5,uncertain_penalty=-0.5 \
  --use_msd --use_talon --output_path results/quick_test_baseline.jsonl --limit 20 | grep -E "(avg_accept_len|relaxed_overall)"

echo ""

# 高score_a配置
echo "[2/5] 测试高score_a配置 (score_a=2.0)..."
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=8,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=2.0,score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=3.0,alea_threshold=3.0,epi_center=0.5,alea_center=0.5,exploit_bonus=2.0,explore_penalty=-0.2,balance_factor=0.5,uncertain_penalty=-0.5 \
  --use_msd --use_talon --output_path results/quick_test_high_a.jsonl --limit 20 | grep -E "(avg_accept_len|relaxed_overall)"

echo ""

# 极端奖励配置
echo "[3/5] 测试极端奖励配置 (exploit_bonus=5.0, explore_penalty=-1.0)..."
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=8,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=0.5,score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=3.0,alea_threshold=3.0,epi_center=0.5,alea_center=0.5,exploit_bonus=5.0,explore_penalty=-1.0,balance_factor=0.5,uncertain_penalty=-0.5 \
  --use_msd --use_talon --output_path results/quick_test_extreme.jsonl --limit 20 | grep -E "(avg_accept_len|relaxed_overall)"

echo ""

# 极端阈值配置
echo "[4/5] 测试极端阈值配置 (thresholds=8.0)..."
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=8,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=0.5,score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=8.0,alea_threshold=8.0,epi_center=0.5,alea_center=0.5,exploit_bonus=2.0,explore_penalty=-0.2,balance_factor=0.5,uncertain_penalty=-0.5 \
  --use_msd --use_talon --output_path results/quick_test_thresh.jsonl --limit 20 | grep -E "(avg_accept_len|relaxed_overall)"

echo ""

# 关闭不确定性 (baseline)
echo "[5/5] 测试关闭不确定性配置 (baseline)..."
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=false \
  --use_msd --use_talon --output_path results/quick_test_baseline_off.jsonl --limit 20 | grep -E "(avg_accept_len|relaxed_overall)"

echo ""
echo "=== 快速测试完成 ==="
echo "如果5个配置的结果差异很小，说明当前参数范围不敏感，需要更大的变化范围"
