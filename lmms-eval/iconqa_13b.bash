# ==== Inject HuggingFace token (DO NOT COMMIT SECRETS) ====
export HUGGINGFACE_HUB_TOKEN="YOUR_TOKEN"
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"


# # Run evaluation
# python -m lmms_eval \
#   --model llava_msd_calibrated \
#   --model_args pretrained="/root/Speculative_decoding/checkpoint/llava-v1.5-7b" \
#   --msd_model_path /root/Speculative_decoding/checkpoint/MSD-LLaVA1.5-7B \
#   --tasks iconqa \
#   --batch_size 1 \
#   --gen_kwargs temperature=0 \
#   --use_msd \
#   --log_samples \
#   --output_path /root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/lmms-eval/results/iconqa.json \
#   --limit 1000

# Run evaluation
python -m lmms_eval \
  --model llava_msd_calibrated \
  --model_args pretrained="/root/Speculative_decoding/checkpoint/llava-v1.5-13b" \
  --msd_model_path /root/Speculative_decoding/checkpoint/MSD-LLaVA1.5-13B \
  --tasks ai2d \
  --batch_size 1 \
  --gen_kwargs temperature=1 \
  --use_msd \
  --log_samples \
  --output_path /root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/lmms-eval/results/ai2d.json \
  --bottom 1000

  # python -m lmms_eval \
  #   --model llava_msd_calibrated \
  #   --model_args pretrained="/root/Speculative_decoding/checkpoint/llava-v1.5-7b" \
  #   --msd_model_path /root/Speculative_decoding/checkpoint/MSD-LLaVA1.5-7B \
  #   --tasks chartqa \
  #   --batch_size 1 \
  #   --gen_kwargs temperature=0 \
  #   --use_msd \
  #   --log_samples \
  #   --output_path /root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/lmms-eval/results/chartqa.json \
  #   --bottom 1500