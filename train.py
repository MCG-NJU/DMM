import argparse
import math
import os
import os.path as osp
import json
import random
import time
import functools
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from packaging import version
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from modeling.dmm_unet import UNet2DConditionModel, DMMUNet2DConditionModel
from data.mock import load_data
from utils.log_utils import get_logger
from utils.train_utils import (
    encode_prompt,
    import_model_class_from_model_name_or_path,
)


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--project_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    parser.add_argument("--seed", type=int, default=2048, help="A seed for reproducible training.")

    # Optimizer
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10000000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "multisteps"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    # dataset
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this",
    )
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Possibility for unconditional")
    parser.add_argument("--use_generated_dataset", action="store_true", default=False, help="if use generated dataset")

    # Logging
    parser.add_argument("--log_interval", type=int, default=1, help="Number of steps for print log.")
    parser.add_argument("--save_interval", type=int, default=2500, help="Number of steps for save model.")
    parser.add_argument("--eval_interval", type=int, default=-1, help="Number of steps for evaluation.")

    # pretrained
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained Stable Diffusion model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained VAE model",
    )

    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )

    # teacher candidates
    parser.add_argument("--models_json_file", type=str, help="Json file containing all pretrained models")

    # feature distillation
    parser.add_argument("--use_feat_loss", action="store_true", default=False)
    parser.add_argument("--feat_loss_type", type=str, choices=["l2", "cos"])
    parser.add_argument("--feat_loss_weight", type=float)

    # resume
    parser.add_argument("--resume_from", type=str, help="The checkpoint path for resuming", default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.project_dir,
    )
    logger.info(f"[Train args]: {args}")

    accelerator.init_trackers("logs")
    logger.info(f"Logging dir: {args.project_dir}")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load Models
    logger.info(f"Start to load model...")

    # Select teacher
    with open(args.models_json_file, 'r') as f:
        models_info: List[Dict] = json.load(f)
    num_models = len(models_info)
    logger.info(f"All {num_models} models' information:")
    logger.info(json.dumps(models_info, indent=2))

    assert accelerator.num_processes % num_models == 0
    teacher_index = accelerator.process_index % num_models
    teacher_model_info = models_info[teacher_index]
    teacher_model_name = teacher_model_info["model_name"]
    teacher_model_path = teacher_model_info["model_path"]
    logger.info(f"Rank {accelerator.process_index} use teacher: {teacher_model_info}", main_process_only=False)

    # set feature distillation layers
    distill_feat_layers = dict(
        down_block_samples=[0, 1, 2, 3],
        mid_block_samples=[0],
        up_block_samples=[0, 1, 2, 3],
    )
    logger.info(f"Distill feature layers: {distill_feat_layers}")

    logger.info(f"Load noise scheduler from: {args.pretrained_model_path}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")

    logger.info(f"Load tokenizer from: {args.pretrained_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer", use_fast=False)

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_path, subfolder="text_encoder")
    logger.info(f"Load text encoder {text_encoder_cls} from: {args.pretrained_model_path}")
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")

    teacher_text_encoder_cls = import_model_class_from_model_name_or_path(teacher_model_path, subfolder="text_encoder")
    logger.info(f"Load teacher text encoder {teacher_text_encoder_cls} from: {teacher_model_path}")
    teacher_text_encoder = teacher_text_encoder_cls.from_pretrained(teacher_model_path, subfolder="text_encoder")

    logger.info(f"Load vae from: {args.vae_model_path}")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path)

    logger.info(f"Load unet config from: {args.pretrained_model_path}")
    unet_base = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    unet = DMMUNet2DConditionModel.from_config(unet_base.config, addition_embed_type="model_merge", num_models=num_models)

    logger.info(f"Load unet state dict from: {args.pretrained_model_path}")
    msg = unet.load_state_dict(unet_base.state_dict(), strict=False)
    logger.info(f"{msg}")  # without model_embedding, add_embedding

    logger.info(f"Load teacher unet from: {teacher_model_path}")
    teacher_unet = UNet2DConditionModel.from_pretrained(teacher_model_path, subfolder="unet")

    # Freeze vae, text_encoder and teacher_unet
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    teacher_text_encoder.requires_grad_(False)
    teacher_text_encoder.eval()
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()
    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32.")
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}")

    # Handle mixed precision and device placement
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Moving vae, text_encoder to device and cast to weight_dtype")
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher_text_encoder.to(accelerator.device, dtype=weight_dtype)

    logger.info("Moving target_unet, teacher_unet to device and optionally cast to weight_dtype")
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:  # optionally cast to weight_dtype
        teacher_unet.to(dtype=weight_dtype)

    # Enable optimizations
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        teacher_unet.enable_xformers_memory_efficient_attention()
        logger.info("Eable xformers")
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer creation
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Here, we compute not just the text embeddings but also the additional embeddings
    def compute_embeddings(prompt_batch,
                           proportion_empty_prompts,
                           text_encoder,
                           teacher_text_encoder,
                           tokenizer,
                           model_id,
                           is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        prompt_embeds_teacher = encode_prompt(prompt_batch, teacher_text_encoder, tokenizer, proportion_empty_prompts,
                                              is_train)
        model_ids = torch.LongTensor([model_id] * len(prompt_batch)).to(accelerator.device)
        out_dict = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_teacher": prompt_embeds_teacher,
            "model_ids": model_ids
        }
        return out_dict

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        teacher_text_encoder=teacher_text_encoder,
        tokenizer=tokenizer,
    )

    logger.info(f"Loading dataset and dataloader")
    train_dataset, train_dataloader = load_data(args, accelerator)

    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Dataloader size: {len(train_dataloader)}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    ## accelerator steps is global steps: step * num_processes
    logger.info("Load lr_scheduler")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("Accelerator preparing unet, dataloader, optimizer and lr_scheduler")
    unet, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(unet, train_dataloader, optimizer, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    global_step = 0
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    NUM_BATCHES = len(train_dataloader)

    logger.info("***** Running training *****")
    logger.info(f"  Num batches per epoch = {NUM_BATCHES}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Mixed precision = {accelerator.mixed_precision}")

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        accelerator.load_state(args.resume_from)
        global_step = int(os.path.basename(args.resume_from).split("-")[1])
        logger.info(f"Resume global_step to {global_step}")

    logger.info("Train!")
    for epoch in range(args.num_train_epochs):
        unet.train()
        for n_iter, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True)  # (b, 3, 512, 512)
                pixel_values = pixel_values.to(dtype=weight_dtype)
                prompt_batch: List[str] = batch["captions"]
                print(prompt_batch)
                encoded_text = compute_embeddings_fn(prompt_batch, model_id=teacher_index)

                # encode pixel values with batch size of at most 32
                latents = []
                for i in range(0, pixel_values.shape[0], 32):
                    latents.append(vae.encode(pixel_values[i:i + 32]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Add noise to the latents according to the noise magnitude at each timestep
                timesteps = torch.randint(0, 1000, (bsz, ), device=latents.device).long()
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")
                prompt_embeds_teacher = encoded_text.pop("prompt_embeds_teacher")

                # Predict the noise residual and compute loss
                pred_dict = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=encoded_text,
                )
                noise_pred = pred_dict.sample  # (b, c, h, w)

                with torch.no_grad():
                    with torch.autocast("cuda"):
                        pred_dict_teacher = teacher_unet(
                            noisy_model_input,
                            timesteps,
                            encoder_hidden_states=prompt_embeds_teacher,
                            added_cond_kwargs=None,
                        )
                        noise_pred_teacher = pred_dict_teacher.sample

                loss = 0.

                noise_loss = F.mse_loss(noise_pred.float(), noise_pred_teacher.float(), reduction="mean")

                loss = loss + noise_loss

                if args.use_feat_loss:
                    total_feat_loss = 0.
                    num_feat_blocks = 0
                    feat_loss_dict: Dict[str, Dict[int, torch.Tensor]] = {}
                    for block_name, indexes in distill_feat_layers.items():
                        feat_loss_dict[block_name] = {}
                        for index in indexes:
                            pred = pred_dict[block_name][index]
                            target = pred_dict_teacher[block_name][index].detach()
                            if args.feat_loss_type == "l2":
                                feat_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                            elif args.feat_loss_type == "cos":
                                cosine_similarity = F.cosine_similarity(pred.float(), target.float(), dim=1)  # (b, h, w)
                                feat_loss = (1 - cosine_similarity.mean())
                            else:
                                raise NotImplementedError
                            feat_loss_dict[block_name][index] = feat_loss.detach().item()
                            total_feat_loss = total_feat_loss + feat_loss
                            num_feat_blocks += 1
                    loss = loss + args.feat_loss_weight * total_feat_loss / num_feat_blocks

                # backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1

                # log && tensorboard
                if (global_step % args.log_interval == 0):
                    if accelerator.is_main_process:
                        accelerator.log({"train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                        accelerator.log({"train/loss": loss.detach().item(),
                                         "train/loss_noise": noise_loss.detach().item()},
                                        step=global_step)
                        if args.use_feat_loss:
                            accelerator.log({"train/loss_feat": total_feat_loss.detach().item()}, step=global_step)

                    state_dict = {
                        "global_step": f"{global_step}",
                        "epoch": '[{}/{}]'.format(int(global_step / num_update_steps_per_epoch), args.num_train_epochs),
                        "iter": '[{}/{}]'.format(global_step % num_update_steps_per_epoch, num_update_steps_per_epoch),
                        "substeps": '[{}/{}]'.format(n_iter % (args.gradient_accumulation_steps * num_update_steps_per_epoch),
                                         args.gradient_accumulation_steps * num_update_steps_per_epoch),
                        "loss": '{:.6f}'.format(loss.detach().item()),
                        "noise_loss": '{:.6f}'.format(noise_loss.detach().item()),
                    }
                    if args.use_feat_loss:
                        state_dict.update({
                            "loss_feat": '{:.6f}'.format(total_feat_loss.detach().item()),
                        })
                        logger.debug(f"Feature Loss: {json.dumps(feat_loss_dict, indent=2)}")
                    logger.info(state_dict)

                # save checkpoint
                if (global_step % args.save_interval == 0):
                    save_path = os.path.join(accelerator.project_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if global_step >= args.max_train_steps:
                    break

    accelerator.end_training()


if __name__ == "__main__":
    main()
