import torch

torch.backends.cuda.matmul.allow_tf32 = True

import copy
import os
from tqdm import tqdm
from datetime import timedelta
import json

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from packaging import version
import warnings

from eagle.model.ea_model import EaModel
from eagle.model.utils import temp_cache
try:
    from eagle.model.calibration_logger import get_calibration_logger, CALIBRATION_LOGGING_ENABLED
    print("CALIBRATION_LOGGING_ENABLED True")
except ImportError as e:
    print(f"CALIBRATION_LOGGING_ENABLED False: {e}")
    CALIBRATION_LOGGING_ENABLED = False
    calibration_logger = None

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_msd_calibrated")
class Llava_MSD_Calibrated(lmms):
    """
    Llava Model with MSD Calibration
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        msd_model: str = None,
        use_msd: bool = False,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="auto",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        **kwargs,
    ) -> None:

        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self._tokenizer, self._model, self._image_processor, self._max_length = EaModel.from_pretrained(
                base_model_path=pretrained,
                ea_model_path=msd_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                total_token=-1
        )

        self._config = self._model.base_model.config
        self.model.eval()
        self.model.base_model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.output_len = 0
        self.output_num = 0
        self.use_msd = use_msd
        # acceptance stats across entire evaluation run
        self.total_accept_len = 0
        self.total_accept_steps = 0

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        
        # print(requests)
        calibrator_mode = "isotonic" ## isotonic, monotonic
        # 数据分割配置
        total_samples = len(requests)
        train_ratio = 0  # 40% 用于训练
        val_ratio = 0    # 10% 用于验证
        test_ratio = 1   # 50% 用于测试
        
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        # 动态创建结果保存路径
        # 从requests中获取任务名称
        task_name = requests[0].task_name if requests else "unknown"
        # FIX: 从第一个 request 的 arguments 元组中解出 gen_kwargs 并读取 temperature
        try:
            req0 = requests[0]
            if hasattr(req0, "arguments") and isinstance(req0.arguments, tuple) and len(req0.arguments) >= 2 and isinstance(req0.arguments[1], dict):
                temperature = req0.arguments[1].get("temperature", 0)
            else:
                temperature = 0
        except Exception:
            temperature = 0

        train_count = train_end
        val_count = val_end - train_end
        test_count = total_samples - val_end
        
        # 构建动态路径（新增校准器策略标识）
        result_folder_name = f"{task_name}_calib_{calibrator_mode}_train_{train_count}_val_{val_count}_test_{test_count}_total_{total_samples}_temperature_{temperature}"
        base_calibration_path = f"/root/Speculative_decoding/calibration_data/{result_folder_name}"
        cross_attention_path = base_calibration_path
    
        calibration_logger = get_calibration_logger(base_calibration_path)

        # 确保目录存在
        os.makedirs(base_calibration_path, exist_ok=True)
        os.makedirs(cross_attention_path, exist_ok=True)
        
        print(f"Results will be saved to: {base_calibration_path}")

        # 跳过训练样本的标志（当已存在校准器时启用）
        skip_to_test = False

        # 检查是否已存在训练好的校准器
        calibrator_dir = os.path.join(base_calibration_path, "calibrators")
        isotonic_path = os.path.join(calibrator_dir, "grouped_isotonic_calibrator.pkl")
        monotonic_path = os.path.join(calibrator_dir, "monotonic_network_calibrator.pkl")

        calibrator_trained = False
        trained_calibrators = {}
        
        print(f"Isotonic path: {isotonic_path}")
        print(f"Monotonic path: {monotonic_path}")
        print(os.path.exists(isotonic_path))
        print(os.path.exists(monotonic_path))
        
        if os.path.exists(isotonic_path) or os.path.exists(monotonic_path):
            print("\n" + "="*50)
            print("Found existing calibrators! Loading pre-trained calibrators...")
            print(f"  - Isotonic: {isotonic_path}")
            print("="*50)
            
            try:
                trained_calibrators = self._load_existing_calibrators(isotonic_path, monotonic_path)
                if trained_calibrators:
                    calibrator_trained = True
                    skip_to_test = True  # 已有校准器，直接跳到测试集
                    # 将训练好的校准器传递给模型（按选择注入）
                    if calibrator_mode in trained_calibrators:
                        self.model.calibrator = trained_calibrators[calibrator_mode]
                        print(f"Pre-trained {calibrator_mode} calibrator loaded into model for inference.")
                    elif 'isotonic' in trained_calibrators:
                        # 回退到 isotonic
                        self.model.calibrator = trained_calibrators['isotonic']
                        print("Requested calibrator not found, fallback to isotonic for inference.")
                    
                    print("Skipping training samples - proceeding directly to TEST split!")
                    print(f"Will process test samples only: {total_samples - val_end} samples")
                    print("="*50 + "\n")
                else:
                    print("Failed to load existing calibrators, will proceed with training.")
            except Exception as e:
                print(f"Error loading existing calibrators: {e}")
                print("Will proceed with training new calibrators.")

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        chunks_list = list(chunks)
        
        for chunk_idx, chunk in enumerate(chunks_list):
            # 确定当前chunk属于哪个数据集

            # if skip_to_test == True and chunk_idx <= train_end:
            #     if chunk_idx % 100 == 0: ## log 
            #         print(f"[msdgenerate] skipping train sample {chunk_idx}")
            #     continue

            current_sample_idx = chunk_idx  # 假设batch_size=1
            
            if current_sample_idx < train_end and not skip_to_test:
                data_phase = "train"
                train_calibrator_flag = True
                use_calibrator = False  # 训练阶段不使用校准器
            elif current_sample_idx < val_end:
                data_phase = "val"
                train_calibrator_flag = False
                use_calibrator = calibrator_trained  # 验证阶段使用已训练的校准器
            else:
                data_phase = "test"
                train_calibrator_flag = False
                use_calibrator = calibrator_trained  # 测试阶段使用已训练的校准器

            # 在训练阶段结束后训练校准器（仅当未预训练，且处理到 train_end 时才触发）
            if current_sample_idx == train_end and not calibrator_trained:
                print("\n" + "="*50)
                print("Training phase completed. Starting calibrator training...")
                print("="*50)
                
                # 保存训练阶段的校准数据
                if CALIBRATION_LOGGING_ENABLED and calibration_logger is not None:
                    calibration_logger.save_data("training_calibration_data")
                    print("Training calibration data saved.")
                
                # 训练校准器：调用 EAGLE 的 calibrator_training
                try:
                    # import sys, os
                    # eagle_path = "/root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/EAGLE"
                    # if eagle_path not in sys.path:
                    #     sys.path.append(eagle_path)
                    from eagle.model.calibrators import calibrator_training

                    calibrator_dir = os.path.join(base_calibration_path, "calibrators")
                    os.makedirs(calibrator_dir, exist_ok=True)
                    json_path = os.path.join(base_calibration_path, "training_calibration_data.json")

                    print(f"Starting calibrator_training with json: {json_path}")
                    iso_cal, mono_cal = calibrator_training(calibrator_dir, json_path)
                    trained_calibrators = {'isotonic': iso_cal, 'monotonic': mono_cal}
                    calibrator_trained = True
                    print("Calibrator training completed and models saved.")
                except Exception as e:
                    print(f"Error training calibrators via EAGLE.calibrator_training: {e}")
                    trained_calibrators = {}
                    calibrator_trained = False
                
                # 将训练好的校准器传递给模型（按选择注入）
                if trained_calibrators:
                    if calibrator_mode in trained_calibrators:
                        self.model.calibrator = trained_calibrators[calibrator_mode]
                        print(f"Calibrator ({calibrator_mode}) loaded into model for inference.")
                    elif 'isotonic' in trained_calibrators:
                        self.model.calibrator = trained_calibrators['isotonic']
                        print("Requested calibrator not available, fallback to isotonic.")
                
                print("Calibrator training completed!")
                print("="*50 + "\n")
            
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks_list)} - Phase: {data_phase} - Use Calibrator: {use_calibrator}")
            
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                # if "Are the giraffes" in question:
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                conv.system=""
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 512
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
                
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            try:
                inputs_embeds, attention_mask = self.model.base_model.get_inputs_embeds(input_ids,image_tensor,gen_kwargs["image_sizes"],attention_masks)
                if self.use_msd:
                    # reset per-chunk acceptance counters
                    setattr(self.model, "acclen", 0)
                    setattr(self.model, "accnum", 0)
                    
                    print(f"actual temperature {gen_kwargs['temperature']}")
                    # 传递inputs_embeds给msdgenerate，这样可以正确计算图像token位置
                    output_ids = self.model.msdgenerate(
                        input_ids, 
                        inputs_embeds=inputs_embeds, 
                        temperature=gen_kwargs["temperature"], 
                        max_new_tokens=512,
                        enable_attention_logging=True,
                        enable_candidate_calibration=True,
                        image_tensor=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        attention_masks_for_padding=attention_masks,
                        train_calibrator=train_calibrator_flag,
                        use_calibrator=use_calibrator,
                        # calibrator=trained_calibrators.get(calibrator_mode, None) if use_calibrator else None
                        calibrator=None
                    ) 
                    
                    print(f"actual temperature {gen_kwargs['temperature']}")
                    # accumulate overall stats
                    self.total_accept_len += int(getattr(self.model, "acclen", 0))
                    self.total_accept_steps += int(getattr(self.model, "accnum", 0))

                    print(f"Average Acceptance Rate: {self.total_accept_len/self.total_accept_steps}")
                else:
                    output_ids = self.model.naivegenerate(input_ids, inputs_embeds=inputs_embeds, temperature=gen_kwargs["temperature"], max_new_tokens=512)
                cont = output_ids[:,input_ids.shape[1]:]

                self.output_len += cont[0].shape[0]
                self.output_num += 1
                temp_cache.total_out_num += cont[0].shape[0]
                temp_cache.total_in_num += input_ids.shape[1]
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        
        res = re_ords.get_original(res)

        pbar.close()
        
        if CALIBRATION_LOGGING_ENABLED and calibration_logger is not None:
            # calibration_logger.save_data("test_calibration_data")
            
            # 计算ECE并绘制图表 - 使用测试数据
            figure_save_path = os.path.join(base_calibration_path, "test_ece_bin_20.png")
            os.makedirs(os.path.dirname(figure_save_path), exist_ok=True)
            
            stats = calibration_logger.get_calibration_stats(
                num_bins=20, 
                save_figure=True, 
                figure_path=figure_save_path
            )
            
            if stats:
                eval_logger.info(f"Test phase calibration analysis completed:")
                eval_logger.info(f"  - ECE: {stats.get('ece', 'N/A'):.4f}")
                eval_logger.info(f"  - Overall acceptance rate: {stats.get('overall_acceptance_rate', 'N/A'):.4f}")
                eval_logger.info(f"  - Total samples: {stats.get('total_samples', 'N/A')}")
                eval_logger.info(f"  - Figure saved to: {figure_save_path}")
    
                # 将总体接受率与样本数写入 base_calibration_path 下的 "acceptance rate" 文件
                try:
                    acceptance_file = os.path.join(base_calibration_path, "acceptance rate")
                    acceptance_payload = {
                        "average_acceptance_rate": float(stats.get("overall_acceptance_rate", 0.0)),
                        "total_samples": int(stats.get("total_samples", 0)),
                        "self.total_accept_len": self.total_accept_len,
                        "self.total_accept_steps": self.total_accept_steps,
                        "average_acc_per_step": self.total_accept_len / self.total_accept_steps
                    }

                    with open(acceptance_file, "w", encoding="utf-8") as f:
                        json.dump(acceptance_payload, f, ensure_ascii=False, indent=2)
                    eval_logger.info(f"  - Acceptance rate summary written to: {acceptance_file}")
                except Exception as write_err:
                    eval_logger.warning(f"Failed to write acceptance rate summary: {write_err}")
            
            # 生成跨模态注意力分析图表
            calibration_logger.plot_cross_modal_attention_comprehensive_analysis(
                save_path=cross_attention_path, 
                confidence_binning="both"
            )
            
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA_eagle")

    def _load_existing_calibrators(self, isotonic_path, monotonic_path):
        """加载已存在的校准器"""
        import sys
        import pickle

        # 添加EAGLE模块路径
        eagle_path = "/root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/EAGLE"
        if eagle_path not in sys.path:
            sys.path.append(eagle_path)

        # 导入校准器类与嵌套的单调网络模型类
        from eagle.model.calibrators import (
            GroupedIsotonicCalibrator,
            MonotonicNetworkCalibrator,
            MonotonicMLP
        )

        # 自定义Unpickler以兼容历史pickle中的命名空间
        class _CalibratorUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # 同时兼容两种嵌套命名
                affine_mono = getattr(MonotonicNetworkCalibrator, "_AffineMono", None) or getattr(MonotonicMLP, "_AffineMono", None)
                mapping = {
                    "GroupedIsotonicCalibrator": GroupedIsotonicCalibrator,
                    "MonotonicNetworkCalibrator": MonotonicNetworkCalibrator,
                    "MonotonicMLP": MonotonicMLP,
                    "_AffineMono": affine_mono,
                    "MonotonicNetworkCalibrator._AffineMono": affine_mono,
                    "MonotonicMLP._AffineMono": affine_mono,
                }
                # 名称优先匹配（无论原始module为何）
                if name in mapping and mapping[name] is not None:
                    return mapping[name]
                # 老命名空间也允许直接返回（双保险）
                if name in mapping and module in (
                    "lmms_eval.__main__",
                    "__main__",
                    "eagle.model.calibrators",
                ):
                    return mapping[name]
                return super().find_class(module, name)

        trained_calibrators = {}

        # 加载分组等渗校准器
        try:
            print(f"Loading isotonic calibrator from: {isotonic_path}")
            with open(isotonic_path, "rb") as f:
                isotonic_cal = _CalibratorUnpickler(f).load()
            if not isinstance(isotonic_cal, GroupedIsotonicCalibrator):
                isotonic_cal = GroupedIsotonicCalibrator.load(isotonic_path)
            trained_calibrators["isotonic"] = isotonic_cal
            print("✓ Isotonic calibrator loaded successfully")
        except Exception as e:
            print(f"Error loading isotonic calibrator: {e}")

        # 加载单调网络校准器
        try:
            print(f"Loading monotonic calibrator from: {monotonic_path}")
            # 加载单调网络校准器（增强容错）
            try:
                print(f"Loading monotonic calibrator from: {monotonic_path}")
                with open(monotonic_path, "rb") as f:
                    monotonic_cal = _CalibratorUnpickler(f).load()
                if not isinstance(monotonic_cal, MonotonicNetworkCalibrator):
                    monotonic_cal = MonotonicNetworkCalibrator.load(monotonic_path)
                trained_calibrators["monotonic"] = monotonic_cal
                print("✓ Monotonic calibrator loaded successfully")
            except Exception as e:
                print(f"Error loading monotonic calibrator with custom Unpickler: {e}")
                # 回退：使用类方法的兼容加载
                try:
                    monotonic_cal = MonotonicNetworkCalibrator.load(monotonic_path)
                    trained_calibrators["monotonic"] = monotonic_cal
                    print("✓ Monotonic calibrator loaded successfully via classmethod fallback")
                except Exception as e2:
                    print(f"Error loading monotonic calibrator via classmethod fallback: {e2}")

        except Exception as e:
            print("Error loading monotonic calibrator")
        
        if trained_calibrators:
            print(f"Successfully loaded {len(trained_calibrators)} calibrators")
        return trained_calibrators

