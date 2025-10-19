# ==== Inject HuggingFace token (DO NOT COMMIT SECRETS) ====
export HUGGINGFACE_HUB_TOKEN="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"
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"

# # Run evaluation
python -m lmms_eval \
  --model llava_msd_calibrated \
  --model_args pretrained="/root/Speculative_decoding/checkpoint/llava-v1.5-7b" \
  --msd_model_path /root/Speculative_decoding/checkpoint/MSD-LLaVA1.5-7B \
  --tasks mathverse_testmini_vision_intensive \
  --batch_size 1 \
  --gen_kwargs temperature=0 \
  --use_msd \
  --log_samples \
  --output_path /root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/lmms-eval/results/mathverse_testmini_vision_intensive.json \
  --limit 1000