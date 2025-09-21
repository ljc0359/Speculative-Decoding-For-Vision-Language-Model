#!/bin/bash

# 🚀 多阶段推测解码优化脚本
# 支持单独运行各阶段或自动化流水线模式

set -e

# 📁 结果目录设置
RESULTS_DIR="results/staged_search"
mkdir -p "$RESULTS_DIR"

# 🎯 运行单个配置的函数
run_config() {
  local name="$1"
  local params="$2"
  local stage_prefix="$3"
  
  echo "🔄 Running ${stage_prefix}_${name}..."
  
  # 构建完整的配置名称
  local full_name="${stage_prefix}_${name}"
  
  # 运行评估 - 修复模型参数
  python -m lmms_eval \
    --model llava_msd \
    --model_args pretrained="/root/Speculative_decoding/checkpoint/llava-v1.5-7b" \
    --msd_model_path /root/Speculative_decoding/checkpoint/MSD-LLaVA1.5-7B \
    --tasks chartqa \
    --batch_size 1 \
    --gen_kwargs temperature=0,${params} \
    --use_msd \
    --use_talon \
    --log_samples \
    --log_samples_suffix ${full_name} \
    --output_path ${RESULTS_DIR}/${full_name}.json \
    --limit 200
  
  # 解析结果
  local result_file="${RESULTS_DIR}/${full_name}.json"
  if [ -f "$result_file" ]; then
    local accuracy=$(python test_parse.py "$result_file" accuracy 2>/dev/null || echo "0")
    local accept_len=$(python test_parse.py "$result_file" accept_len 2>/dev/null || echo "0")
    local composite_score=0  # 暂时设为0，后续可以计算
    
    echo "${full_name},${accuracy},${accept_len},${composite_score},${params}" >> "${RESULTS_DIR}/${stage_prefix}_results.csv"
    echo "✅ ${full_name}: accuracy=${accuracy}, accept_len=${accept_len}"
  else
    echo "❌ 结果文件未找到: $result_file"
  fi
}

# 🔍 解析最佳结果函数
parse_best_results() {
  local stage="$1"
  local results_file="${RESULTS_DIR}/${stage}_results.csv"
  
  if [ ! -f "$results_file" ]; then
    echo "⚠️  警告: ${stage} 结果文件不存在: $results_file"
    return 1
  fi
  
  echo "📊 解析 ${stage^^} 最佳结果..."
  
  # 按accept_len排序并获取最佳结果
  local best_line=$(sort -t',' -k3 -nr "$results_file" | head -1)
  
  if [ -z "$best_line" ]; then
    echo "❌ 无法找到 ${stage} 的最佳结果"
    return 1
  fi
  
  # 解析最佳结果
  local best_name=$(echo "$best_line" | cut -d',' -f1)
  local best_accuracy=$(echo "$best_line" | cut -d',' -f2)
  local best_accept_len=$(echo "$best_line" | cut -d',' -f3)
  local best_params=$(echo "$best_line" | cut -d',' -f5-)
  
  echo "🏆 ${stage^^} 最佳配置: $best_name"
  echo "   📈 Accuracy: $best_accuracy"
  echo "   🎯 Accept Length: $best_accept_len"
  
  # 根据阶段设置相应的环境变量
  case "$stage" in
    "stage1")
      # 从配置名称中提取参数 (格式: s1_a1.0_eb3.0_ep-2.0)
      export BEST_SCORE_A=$(echo "$best_name" | grep -o 'a[0-9.]*' | sed 's/a//')
      export BEST_EB=$(echo "$best_name" | grep -o 'eb[0-9.]*' | sed 's/eb//')
      export BEST_EP=$(echo "$best_name" | grep -o 'ep-[0-9.]*' | sed 's/ep//')
      
      echo "   ✅ 已设置 Stage1 最佳参数:"
      echo "      BEST_SCORE_A=${BEST_SCORE_A}"
      echo "      BEST_EB=${BEST_EB}"
      echo "      BEST_EP=${BEST_EP}"
      ;;
      
    "stage2")
      # 从参数字符串中提取阈值和中心点参数
      export BEST_ET=$(echo "$best_params" | grep -o 'epi_threshold=[^,]*' | cut -d'=' -f2)
      export BEST_AT=$(echo "$best_params" | grep -o 'alea_threshold=[^,]*' | cut -d'=' -f2)
      export BEST_EC=$(echo "$best_params" | grep -o 'epi_center=[^,]*' | cut -d'=' -f2)
      export BEST_AC=$(echo "$best_params" | grep -o 'alea_center=[^,]*' | cut -d'=' -f2)
      export BEST_UP=$(echo "$best_params" | grep -o 'uncertain_penalty=[^,]*' | cut -d'=' -f2)
      
      echo "   ✅ 已设置 Stage2 最佳参数:"
      echo "      BEST_ET=${BEST_ET}"
      echo "      BEST_AT=${BEST_AT}"
      echo "      BEST_EC=${BEST_EC}"
      echo "      BEST_AC=${BEST_AC}"
      echo "      BEST_UP=${BEST_UP}"
      ;;
  esac
  
  return 0
}

# 📈 阶段结果分析函数
analyze_stage_results() {
  local stage="$1"
  local out_dir="$2"
  
  echo ""
  echo "🎯 ${stage^^} COMPLETED! 结果分析:"
  echo "=================================="
  
  if [ -f "${out_dir}/${stage}_results.csv" ]; then
    echo "name,accuracy,accept_len,composite_score,params" > "${out_dir}/${stage}_sorted.csv"
    tail -n +1 "${out_dir}/${stage}_results.csv" | sort -t',' -k3 -nr | head -10 >> "${out_dir}/${stage}_sorted.csv"
    
    echo ""
    echo "🏆 TOP 10 CONFIGURATIONS:"
    echo "Rank | Name | Accuracy | Accept_Len | Score"
    echo "-----|------|----------|------------|-------"
    
    local rank=1
    tail -n +2 "${out_dir}/${stage}_sorted.csv" | while IFS=',' read -r name accuracy accept_len score params; do
      printf "%4d | %-20s | %8s | %10s | %5s\n" "$rank" "$name" "$accuracy" "$accept_len" "$score"
      rank=$((rank+1))
    done
    
    echo ""
    echo "📁 详细结果已保存到:"
    echo "   📄 ${out_dir}/${stage}_results.csv (完整结果)"
    echo "   📊 ${out_dir}/${stage}_sorted.csv (排序后前10名)"
    
    # 解析最佳结果用于下一阶段
    parse_best_results "$stage"
  else
    echo "❌ 未找到结果文件: ${out_dir}/${stage}_results.csv"
  fi
}

# 🎮 主要执行逻辑
STAGE=${1:-"help"}

case $STAGE in
  "help")
    echo "🚀 多阶段推测解码优化脚本"
    echo ""
    echo "用法:"
    echo "  bash bash.sh <stage>     # 运行指定阶段"
    echo "  bash bash.sh auto        # 自动运行完整流水线"
    echo "  bash bash.sh baseline    # 运行基线测试"
    echo ""
    echo "可用阶段:"
    echo "  stage1    - 基础参数优化 (score_a, exploit_bonus, explore_penalty)"
    echo "  stage2    - 双重不确定性阈值优化"
    echo "  stage3    - 精细化参数调优"
    echo "  auto      - 自动依次运行 stage1 -> stage2 -> stage3"
    echo "  baseline  - 基线性能测试"
    echo ""
    ;;
    
  "auto")
    echo "🤖 启动自动化三阶段优化流水线"
    echo "======================================="
    
    STAGES=("stage1" "stage2" "stage3")
    
    echo "将依次运行: Stage1 -> Stage2 -> Stage3"
    echo "预计总时间: 8-12小时"
    echo ""
    
    for stage in "${STAGES[@]}"; do
      echo ""
      echo "🚀 开始执行 ${stage^^}"
      echo "======================================="
      
      # 运行当前阶段
      bash "$0" "$stage"
      
      # 检查阶段是否成功完成
      if [ $? -eq 0 ]; then
        echo "✅ ${stage^^} 执行成功!"
        
        # 分析结果并为下一阶段准备参数
        analyze_stage_results "$stage" "$RESULTS_DIR"
      else
        echo "❌ ${stage^^} 执行失败，停止流水线"
        exit 1
      fi
      
      echo ""
      echo "⏱️  等待5秒后继续下一阶段..."
      sleep 5
    done
    
    echo ""
    echo "🎉 自动化流水线完成!"
    echo "======================================="
    echo "📊 最终结果总结:"
    
    # 显示所有阶段的最佳结果
    for stage in "${STAGES[@]}"; do
      if [ -f "${RESULTS_DIR}/${stage}_sorted.csv" ]; then
        echo ""
        echo "🏆 ${stage^^} 最佳结果:"
        head -2 "${RESULTS_DIR}/${stage}_sorted.csv" | tail -1
      fi
    done
    ;;
    
  "baseline")
    echo "📊 运行基线测试..."
    echo "使用默认参数进行性能基准测试"
    echo ""
    
    # 基线配置 - 不使用不确定性评分
    baseline_params="use_uncertainty_scoring=false"
    
    run_config "baseline" "$baseline_params" "baseline"
    
    echo ""
    echo "✅ 基线测试完成!"
    echo "结果已保存到 ${RESULTS_DIR}/baseline_baseline.json"
    ;;
    
  "stage1")
    echo "🎯 STAGE 1: 基础参数优化"
    echo "目标: 寻找最优的score_a, exploit_bonus, explore_penalty组合"
    echo "策略: 网格搜索关键参数空间"
    echo "配置数量: 48个 (预计2-3小时)"
    echo ""
    
    count=0
    total=48
    
    # 清空之前的结果
    > "${RESULTS_DIR}/stage1_results.csv"
    
    for A in 1.0 2.0 4.0 8.0; do  # 🔥 大幅增加score_a
      for EB in 3.0 6.0 10.0; do  # 🔥 强化exploit奖励
        for EP in -0.5 -1.0 -1.5 -2.0; do  # 🔥 加强explore惩罚
          count=$((count+1))
          name="a${A}_eb${EB}_ep${EP}"
          params="use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=12,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=${A},score_b=0.05,score_c=0.0,score_d=0.3,use_js=false,epi_threshold=2.0,alea_threshold=1.5,epi_center=0.4,alea_center=0.4,exploit_bonus=${EB},explore_penalty=${EP},balance_factor=0.5,uncertain_penalty=-0.5"
          
          echo "Progress: ${count}/${total}"
          run_config "$name" "$params" "s1"
        done
      done
    done
    
    # 分析Stage1结果
    analyze_stage_results "stage1" "$RESULTS_DIR"
    ;;
    
  "stage2")
    echo "🎯 STAGE 2: 优化双重不确定性阈值"
    echo "目标: 基于Stage1最佳score_a，优化阈值和中心点"
    echo "策略: 固定较好的score_a，系统测试阈值组合"
    echo "配置数量: 60个 (预计3-4小时)"
    echo ""
    
    # 硬编码Stage1最佳参数 (s1_a1.0_eb3.0_ep-2.0)
    BEST_SCORE_A=1.0
    BEST_EB=3.0
    BEST_EP=-2.0
    
    echo "使用Stage1最佳参数: score_a=${BEST_SCORE_A}, exploit_bonus=${BEST_EB}, explore_penalty=${BEST_EP}"
    echo ""
    
    count=0
    total=60
    
    # 清空之前的结果
    > "${RESULTS_DIR}/stage2_results.csv"
    
    for THRESH_CONFIG in "1.0,1.0" "1.5,1.0" "2.0,1.5" "2.5,2.0" "3.0,2.5"; do
      ET=$(echo $THRESH_CONFIG | cut -d',' -f1)
      AT=$(echo $THRESH_CONFIG | cut -d',' -f2)
      for CENTER_CONFIG in "0.3,0.3" "0.4,0.3" "0.4,0.4" "0.5,0.4"; do
        EC=$(echo $CENTER_CONFIG | cut -d',' -f1)
        AC=$(echo $CENTER_CONFIG | cut -d',' -f2)
        for UP in -0.3 -0.5 -0.8; do
          count=$((count+1))
          name="et${ET}_at${AT}_ec${EC}_ac${AC}_up${UP}"
          params="use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=12,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=${BEST_SCORE_A},score_b=0.05,score_c=0.0,score_d=0.3,use_js=false,epi_threshold=${ET},alea_threshold=${AT},epi_center=${EC},alea_center=${AC},exploit_bonus=${BEST_EB},explore_penalty=${BEST_EP},balance_factor=0.5,uncertain_penalty=${UP}"
          
          echo "Progress: ${count}/${total}"
          run_config "$name" "$params" "s2"
        done
      done
    done
    
    # 分析Stage2结果
    analyze_stage_results "stage2" "$RESULTS_DIR"
    ;;
    
  "stage3")
    echo "🎯 STAGE 3: 精细化参数调优"
    echo "目标: 基于前两阶段最佳参数，精细调优所有参数"
    echo "策略: 在最佳配置周围进行局部搜索"
    echo "配置数量: 36个 (预计2-3小时)"
    echo ""
    
    # 使用从Stage2解析的最佳参数，如果没有则使用默认值
    BEST_SCORE_A=${BEST_SCORE_A:-1.0}
    BEST_EB=${BEST_EB:-3.0}
    BEST_EP=${BEST_EP:--2.0}
    BEST_ET=${BEST_ET:-2.0}
    BEST_AT=${BEST_AT:-1.5}
    BEST_EC=${BEST_EC:-0.4}
    BEST_AC=${BEST_AC:-0.4}
    BEST_UP=${BEST_UP:--0.5}
    
    echo "使用前两阶段最佳参数进行精细调优..."
    echo "Base: score_a=${BEST_SCORE_A}, exploit_bonus=${BEST_EB}, explore_penalty=${BEST_EP}"
    echo "      epi_threshold=${BEST_ET}, alea_threshold=${BEST_AT}, uncertain_penalty=${BEST_UP}"
    echo ""
    
    count=0
    total=36
    
    # 清空之前的结果
    > "${RESULTS_DIR}/stage3_results.csv"
    
    # 在最佳参数周围进行精细搜索
    for SCORE_B in 0.03 0.05 0.07; do
      for SCORE_C in 0.0 0.1 0.2; do
        for SCORE_D in 0.2 0.3 0.4; do
          for BF in 0.3 0.5 0.7; do
            count=$((count+1))
            name="sb${SCORE_B}_sc${SCORE_C}_sd${SCORE_D}_bf${BF}"
            params="use_uncertainty_scoring=true,use_mc_alea_epi=true,uncertainty_stride=1,mc_samples=12,mc_noise_std=0.2,mc_temperature=1.0,mc_kind=gauss,score_a=${BEST_SCORE_A},score_b=${SCORE_B},score_c=${SCORE_C},score_d=${SCORE_D},use_js=false,epi_threshold=${BEST_ET},alea_threshold=${BEST_AT},epi_center=${BEST_EC},alea_center=${BEST_AC},exploit_bonus=${BEST_EB},explore_penalty=${BEST_EP},balance_factor=${BF},uncertain_penalty=${BEST_UP}"
            
            echo "Progress: ${count}/${total}"
            run_config "$name" "$params" "s3"
          done
        done
      done
    done
    
    # 分析Stage3结果
    analyze_stage_results "stage3" "$RESULTS_DIR"
    ;;
    
  *)
    echo "❌ 未知阶段: $STAGE"
    echo "使用 'bash bash.sh help' 查看可用选项"
    exit 1
    ;;
esac

echo ""
echo "🎉 执行完成!"