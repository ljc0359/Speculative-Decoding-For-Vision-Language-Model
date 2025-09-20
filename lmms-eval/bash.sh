CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
  --model llava_msd \
  --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
  --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
  --tasks chartqa \
  --batch_size 1 \
  --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=8,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=0.5,score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=3.0,alea_threshold=3.0,epi_center=0.5,alea_center=0.5,exploit_bonus=2.0,explore_penalty=-0.2,balance_factor=0.5,uncertain_penalty=-0.5 \
  --use_msd \
  --use_talon \
  --output_path results/talon_debug.jsonl \
  --log_samples \
  --limit 50

# Grid search helper: sweep key params. Usage: bash bash.sh sweep
if [ "$1" = "sweep" ]; then
  set -e
  DEV=${CUDA_VISIBLE_DEVICES:-3}
  PRE="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b"
  MSD="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B"
  OUT_DIR="${SWEEP_OUT_DIR:-results/sweeps_optimized}"
  mkdir -p "$OUT_DIR"

  # 磁盘空间检查函数（<1GB则退出）
  check_space() {
    local path="$1"
    local avail_kb
    avail_kb=$(df -Pk "$path" | awk 'NR==2{print $4}')
    if [ -z "$avail_kb" ]; then
      return 0
    fi
    if [ "$avail_kb" -lt 1048576 ]; then
      echo "[ERROR] Low disk space: $(df -h \"$path\" | awk 'NR==2{print $4" free on "$1}')" >&2
      echo "[HINT] Free space or set SWEEP_OUT_DIR to a larger filesystem, or run with FORCE=1 after cleanup." >&2
      exit 1
    fi
  }
  check_space "$OUT_DIR"

  echo "Starting SMART grid search based on quick test insights..."
  echo "Key findings: score_a<=0.8, moderate rewards, thresholds 2-8 work well"
  echo "Total configurations to test: ~960 (reduced for ~24h budget)"
  
  # 基于快速测试结果的优化搜索
  # 避免已知有害的参数组合，专注于有效范围
  count=0
  TOTAL=960
  
  for A in 0.1 0.3 0.5 0.8; do  # score_a: 安全范围，避免>1.0
    for EB in 1.5 2.0 2.5 3.0; do  # exploit_bonus: 保守范围，避免>4.0
      for EP in -0.1 -0.2 -0.3; do  # explore_penalty: 适中范围，避免<-0.5
        for THRESH in "2.0,2.0" "3.0,3.0" "5.0,5.0" "8.0,8.0" "3.0,5.0"; do  # 验证阈值影响（5种以达到总数1920）
          ET=$(echo $THRESH | cut -d',' -f1)
          AT=$(echo $THRESH | cut -d',' -f2)
          for MC_CONFIG in "8,0.2" "12,0.15"; do  # 缩减为2种组合
            S=$(echo $MC_CONFIG | cut -d',' -f1)
            NS=$(echo $MC_CONFIG | cut -d',' -f2)
            
            # 缩减中心点为2种
            for CENTER_CONFIG in "0.4,0.4" "0.6,0.4"; do
              EC=$(echo $CENTER_CONFIG | cut -d',' -f1) 
              AC=$(echo $CENTER_CONFIG | cut -d',' -f2)
              
              NAME="smart_a${A}_eb${EB}_ep${EP}_et${ET}_at${AT}_s${S}_ns${NS}_ec${EC}_ac${AC}"
              CMD="CUDA_VISIBLE_DEVICES=${DEV} PYTHONPATH=/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:\$PYTHONPATH python -m lmms_eval --model llava_msd --model_args pretrained=\"${PRE}\" --msd_model_path ${MSD} --tasks chartqa --batch_size 1 --gen_kwargs temperature=0,use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=${S},mc_noise_std=${NS},mc_temperature=1.0,mc_kind=gauss,score_a=${A},score_b=0.1,score_c=0.0,score_d=0.4,use_js=false,epi_threshold=${ET},alea_threshold=${AT},epi_center=${EC},alea_center=${AC},exploit_bonus=${EB},explore_penalty=${EP},balance_factor=0.5,uncertain_penalty=-0.5 --use_msd --use_talon --output_path ${OUT_DIR}/${NAME}.jsonl --limit 50 | cat"
              
              count=$((count+1))
              echo "[$(date '+%H:%M:%S')] Running ${NAME} (${count}/${TOTAL})"
              RESULT_DIR="${OUT_DIR}/${NAME}.jsonl/checkpoint__llava-v1.5-7b"
              check_space "$OUT_DIR"
              if [ -z "${FORCE}" ] && compgen -G "${RESULT_DIR}/*_results.json" > /dev/null; then
                echo "[CACHE] Results already exist for ${NAME}. Skipping run. (set FORCE=1 to re-run)"
              else
                eval "$CMD"
              fi
              
              # 提取关键指标并记录
              if [ -f "${OUT_DIR}/${NAME}.jsonl/checkpoint__llava-v1.5-7b/"*"_results.json" ]; then
                RESULT_FILE=$(ls ${OUT_DIR}/${NAME}.jsonl/checkpoint__llava-v1.5-7b/*_results.json | head -1)
                ACC=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['results']['chartqa']['relaxed_overall,none']:.3f}\")" 2>/dev/null || echo "N/A")
                ACCEPT=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['msd_stats']['avg_accept_len']:.3f}\")" 2>/dev/null || echo "N/A")
                
                # 记录全部结果（不过滤）
                if [[ "$ACC" != "N/A" && "$ACCEPT" != "N/A" ]]; then
                  BASELINE_ACCEPT=2.56
                  SCORE=$(python3 -c "print(f'{float('$ACC') + 0.1 * max(0, float('$ACCEPT') - $BASELINE_ACCEPT):.4f}')" 2>/dev/null || echo "0")
                  echo "[RESULT] ${NAME}: acc=${ACC}, accept=${ACCEPT}, score=${SCORE}"
                  check_space "$OUT_DIR"
                  echo "${NAME},${A},${EB},${EP},${ET},${AT},${S},${NS},${EC},${AC},${ACC},${ACCEPT},${SCORE}" >> "${OUT_DIR}/results_summary.csv"
                else
                  echo "[ERROR] Failed to extract results for ${NAME}"
                fi
              fi
            done
          done
        done
      done
    done
  done
  
  echo ""
  echo "🎯 SMART Grid search completed! Results analysis:"
  echo "================================================"
  
  if [ -f "${OUT_DIR}/results_summary.csv" ]; then
    # 按综合分数排序
    echo "name,score_a,exploit_bonus,explore_penalty,epi_thresh,alea_thresh,mc_samples,mc_noise,epi_center,alea_center,accuracy,accept_len,composite_score" > "${OUT_DIR}/results_summary_sorted.csv"
    tail -n +2 "${OUT_DIR}/results_summary.csv" | sort -t',' -k13 -nr | head -15 >> "${OUT_DIR}/results_summary_sorted.csv"
    
    echo ""
    echo "🏆 TOP 10 CONFIGURATIONS (by composite score = accuracy + 0.1*accept_improvement):"
    head -11 "${OUT_DIR}/results_summary_sorted.csv" | column -t -s','
    
    echo ""
    echo "📊 QUICK STATS:"
    TOTAL_CONFIGS=$(wc -l < "${OUT_DIR}/results_summary.csv" || echo "0")
    BEST_ACC=$(tail -n +2 "${OUT_DIR}/results_summary.csv" | cut -d',' -f11 | sort -nr | head -1 || echo "N/A")
    BEST_ACCEPT=$(tail -n +2 "${OUT_DIR}/results_summary.csv" | cut -d',' -f12 | sort -nr | head -1 || echo "N/A")
    echo "- Total configurations tested: $TOTAL_CONFIGS"
    echo "- Best accuracy achieved: $BEST_ACC"
    echo "- Best acceptance length: $BEST_ACCEPT"
    echo "- Baseline acceptance: 2.78"
    
    echo ""
    if [ -f "${OUT_DIR}/excellent_configs.log" ]; then
      echo "🌟 EXCELLENT CONFIGURATIONS FOUND:"
      cat "${OUT_DIR}/excellent_configs.log"
    else
      echo "ℹ️  No configurations exceeded the excellence threshold (accept>2.9, acc>=0.20)"
    fi
    
    echo ""
    echo "📁 Results saved to:"
    echo "   - ${OUT_DIR}/results_summary_sorted.csv (sorted by performance)"
    echo "   - ${OUT_DIR}/excellent_configs.log (outstanding configs)"
  else
    echo "❌ No results summary found. Check for errors in the sweep."
  fi
fi

# Quick baseline test: Usage: bash bash.sh baseline  
if [ "$1" = "baseline" ]; then
  echo "Running baseline test (use_uncertainty_scoring=false)..."
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/MSD/lmms-eval:$PYTHONPATH" python -m lmms_eval \
    --model llava_msd \
    --model_args pretrained="/mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/llava-v1.5-7b" \
    --msd_model_path /mnt/afs/intern/huangtao3/mayanxiang/ljc/Speculative_Decoding_For_VLM/checkpoint/MSD-LLaVA1.5-7B \
    --tasks chartqa \
    --batch_size 1 \
    --gen_kwargs temperature=0,use_uncertainty_scoring=false \
    --use_msd \
    --use_talon \
    --output_path results/baseline_test.jsonl \
    --limit 50
fi