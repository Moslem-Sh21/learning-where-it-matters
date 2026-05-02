import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL training script for Bottleneck Concept Module (Linear Probing).")

    # ==================================================================
    # 1. Model & Architecture
    # ==================================================================
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--resolution", type=int, default=1024, help="SDXL native resolution (1024x1024).")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="The directory where the downloaded models and datasets will be stored.")

    # ==================================================================
    # 2. Data & Loading
    # ==================================================================
    parser.add_argument("--train_data_dir", type=str, default="datasets_SDXL_female", help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, default="exps_female_sdxl",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard/WandB log directory.")

    parser.add_argument(
        "--select",
        type=str,
        default="random",
        choices=["random", "top"],
        help="Strategy to select concept/prompt from labels.json (random or top)."
    )

    parser.add_argument("--center_crop", action="store_true",
                        help="Whether to center crop images before resizing to resolution.")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="For debugging purposes or quicker training, truncate the number of training examples.")

    # ==================================================================
    # 3. Training Hyperparameters
    # ==================================================================
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. (Note: train_sdxl.py currently forces 'no' (FP32) for concept stability)."
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="The integration to report the results and logs to.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # ==================================================================
    # 4. Optimizer & Scheduler
    # ==================================================================
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # ==================================================================
    # 5. Validation / Logging
    # ==================================================================
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--log_every_epochs', type=int, default=1, help="Log images/checkpoints every N epochs.")

    # ==================================================================
    # 6. Inference / Testing / Legacy Arguments
    #    (These are kept for compatibility if you use a separate test.py)
    # ==================================================================
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--concept', nargs='+')
    parser.add_argument('--num_test_samples', type=int, default=2)
    parser.add_argument('--negative_prompt', default=None, type=str, help="negative prompts for SDXL")
    parser.add_argument('--scheduler', default='euler_a', type=str,
                        choices=['pndm', 'ddim', 'ddpm', 'euler_a', 'dpm_solver'])
    parser.add_argument('--num_inference_steps', default=30, type=int)

    # Legacy / Unused in current train_sdxl.py but potentially useful for extension
    parser.add_argument('--fp16', action='store_true',
                        help="Use float16 precision (Legacy flag, use --mixed_precision instead)")
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true',
                        help="Enable xformers for memory efficiency")

    args = parser.parse_args()

    # Basic Validation
    if args.resolution < 1024:
        print("Warning: SDXL works best with resolution >= 1024. Consider using 1024x1024.")

    return args


if __name__ == '__main__':
    args = parse_args()
    print("Configuration loaded.")
