import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import subprocess
import sys

import torch
import torch.nn.functional as Funet
from torch.utils.data import Dataset
import torch.nn.functional as F

from datasets import load_dataset
from cleanfid import fid

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from Conv import *
from torch import autocast
# from DeepFlix.generative.diffusion.evaluation.evaluate import evaluate_datasets
import wandb
import numpy as np
import xformers

from oct_diffusion.eval.evaluate import evaluate_using_filepaths
from oct_diffusion.model.DiffusionModel import SDM


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
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
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--save_n_steps",
        type=int,
        default=1,
        help=("Save the model every n global_steps"),
    )
    
    
    parser.add_argument(
        "--save_starting_step",
        type=int,
        default=1,
        help=("The step from which it starts saving intermediary checkpoints"),
    )
    
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=1000000,
        help=("The step at which the text_encoder is no longer trained"),
    )


    parser.add_argument(
        "--image_captions_filename",
        action="store_true",
        help="Get captions from filename",
    )    
    
    
    parser.add_argument(
        "--dump_only_text_encoder",
        action="store_true",
        default=False,        
        help="Dump only text-encoder",
    )

    parser.add_argument(
        "--train_only_unet",
        action="store_true",
        default=False,        
        help="Train only the unet",
    )
    
    parser.add_argument(
        "--train_only_text_encoder",
        action="store_true",
        default=False,        
        help="Train only the text-encoder",
    )        
    
    parser.add_argument(
        "--Resumetr",
        type=str,
        default="False",        
        help="Resume training info",
    )    
    
    parser.add_argument(
        "--Style",
        action="store_true",
        default=False,        
        help="Further reduce overfitting",
    )        
    
    parser.add_argument(
        "--Session_dir",
        type=str,
        default="",     
        help="Current session directory",
    )    

    parser.add_argument(
        "--external_captions",
        action="store_true",
        default=False,        
        help="Use captions stored in a txt file",
    )    
    
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="",
        help="The folder where captions files are stored",
    )        

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help="Offset Noise",
    )  
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=5000,
        help="Evaluate the model every n steps. Includes a generation process of images",
    )
    parser.add_argument(
        "--generate_n_images_eval",
        type=int,
        default=100,
        help="Number of images to be generated for each eval run",
    )


    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        args,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.image_captions_filename = None

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if args.image_captions_filename:
            self.image_captions_filename = True
        
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index, args=parse_args()):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        instance_prompt = self.instance_prompt
        
        if self.image_captions_filename:
            filename = Path(path).stem
            
            pt=''.join([i for i in filename if not i.isdigit()])
            pt=pt.replace("_"," ")
            pt=pt.replace("(","")
            pt=pt.replace(")","")
            pt=pt.replace("-","")
            pt=pt.replace("conceptimagedb","")  
            
            if args.external_captions:
              cptpth=os.path.join(args.captions_dir, filename+'.txt')
              if os.path.exists(cptpth):
                with open(cptpth, "r") as f:
                   instance_prompt=pt+' '+f.read()
              else:
                instance_prompt=pt
            else:
                if args.Style:
                  instance_prompt = ""
                else:
                  instance_prompt = pt
            #sys.stdout.write(" [0;32m" +instance_prompt+" [0m")
            #sys.stdout.flush()


        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    i=args.save_starting_step
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )
    # if accelerator.is_main_process:
    #     run = wandb.init(project="foundation_model")
    #     session_name = args.Session_dir.split("/")[-1]
    #     run.name = f"db-{session_name}"
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                with torch.autocast("cuda"):                
                    images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    if args.train_only_unet or args.dump_only_text_encoder:
      if os.path.exists(str(args.output_dir+"/text_encoder_trained")):
        text_encoder = CLIPTextModel.from_pretrained(args.output_dir, subfolder="text_encoder_trained")
      else:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    else:
      text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    if args.enable_xformers_memory_efficient_attention:
        vae.enable_xformers_memory_efficient_attention()
        unet.enable_xformers_memory_efficient_attention()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    from datasets import load_dataset
    # load dataset 
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    dataset_name_mapping = {
        "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
        
    if not os.path.exists("./train_images"):
        os.mkdir("./train_images")
    from PIL import Image
    train_image_dataset = load_dataset(f"{args.dataset_name}", split="train")

    import glob

    files = glob.glob("./train_images/*")
    if len(files) < len(train_image_dataset):
        for i in tqdm(range(len(train_image_dataset))):
            img = Image.fromarray(np.array(train_image_dataset[i]["image"]))
            img.save(f"./train_images/{train_image_dataset[i][args.caption_column]}-({i}).png")
    else:
        print("Found dataset")

    train_dataset = DreamBoothDataset(
        instance_data_root="./train_images/",
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        args=args,
    )

    # val_dataset = DreamBoothDataset(
    #     instance_data_root=args.instance_data_dir_val,
    #     instance_prompt=args.instance_prompt,
    #     class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    #     class_prompt=args.class_prompt,
    #     tokenizer=tokenizer,
    #     size=args.resolution,
    #     center_crop=args.center_crop,
    #     args=args,
    # )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def bar(prg):
       br='|'+'█' * prg + ' ' * (25-prg)+'|'
       return br
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    print(f"{args.resolution}")
    from PIL import Image

    # create folder val_images if not exists
    if not os.path.exists("./val_images"):
        os.mkdir("./val_images")
    files = glob.glob("./train_images/*")
    if len(files) < len(train_image_dataset):
        val_image_dataset = load_dataset(f"{args.dataset_name}", split="val")
        for i in tqdm(range(len(val_image_dataset))):
            img = Image.fromarray(np.array(val_image_dataset[i]["image"]))
            img.save(f"./val_images/{i}.png")
    else:
        print("Found val dataset")

    try:
        fid.make_custom_stats("val", fdir="./val_images/")
    except:
        print("Stats already exist")


    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    loss_avg = AverageMeter()
    curr_step_evaluated = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        losses = list()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                if args.offset_noise:
                  noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
                else:
                  noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    losses.append(loss.item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_avg.update(loss.detach_(), bsz)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            fll=round((global_step*100)/args.max_train_steps)
            fll=round(fll/4)
            pr=bar(fll)
            
            logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            progress_bar.set_description_str("Progress:")
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if args.train_text_encoder and global_step == args.stop_text_encoder_training and global_step >= 5:
              if accelerator.is_main_process:
                print(" [0;32m" +" Freezing the text_encoder ..."+" [0m")                
                frz_dir=args.output_dir + "/text_encoder_frozen"
                if os.path.exists(frz_dir):
                  subprocess.call('rm -r '+ frz_dir, shell=True)
                os.mkdir(frz_dir)
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                )
                pipeline.text_encoder.save_pretrained(frz_dir)
                         
            if args.save_n_steps >= 200:
               if global_step < args.max_train_steps and global_step+1==i:
                  inst=os.path.basename(args.Session_dir)
                  inst =inst + "_step_" + str(global_step+1)
                  print(" [1;32mSAVING CHECKPOINT...")
                  if accelerator.is_main_process:
                     pipeline = StableDiffusionPipeline.from_pretrained(
                           args.pretrained_model_name_or_path,
                           unet=accelerator.unwrap_model(unet),
                           text_encoder=accelerator.unwrap_model(text_encoder),
                     )               
                     chkpth=args.Session_dir+"/"+inst+".ckpt"
                     if args.mixed_precision=="fp16":
                        save_stable_diffusion_checkpoint(unet.config.cross_attention_dim == 1024, chkpth, pipeline.text_encoder, pipeline.unet, None, 0, 0, torch.float16, pipeline.vae)
                     else:
                        save_stable_diffusion_checkpoint(unet.config.cross_attention_dim == 1024, chkpth, pipeline.text_encoder, pipeline.unet, None, 0, 0, None, pipeline.vae)
                     print("Done, resuming training ...[0m")   
                     i=i+args.save_n_steps

            if global_step % 10 == 0:
                if accelerator.is_main_process:
                    logger.info("logging loss")
                    losses_arr = np.array(losses)
                    mean_loss = np.mean(losses_arr)
                    # wandb.log({"loss": loss_avg.avg.item()}, step=global_step)
                    losses = list()
                
            # EVAL only UNet
            if args.train_only_unet and curr_step_evaluated != global_step:
                if global_step % args.eval_every_n_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                        weight_dtype = torch.float32
                        if args.mixed_precision == "fp16":
                            weight_dtype = torch.float16
                        elif args.mixed_precision == "bf16":
                            weight_dtype = torch.bfloat16
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                        )
                        if args.enable_xformers_memory_efficient_attention:
                            pipeline.enable_xformers_memory_efficient_attention()
                        # pipeline.to(accelerator.device) 
                        #                     # moving all components to cpu to free up gpu memory
                        vae.to("cpu")
                        unet.to("cpu")
                        text_encoder.to("cpu")

                        torch.cuda.empty_cache()

                        sdm = SDM(
                            pipeline=StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            vae=vae,
                            unet=accelerator.unwrap_model(unet),
                            torch_dtype=weight_dtype,
                            safety_checker=None,
                            ), 
                            use_xformers=True
                            )
                        sdm.sample_dataset_distributed(
                            dataset_size=100,
                            num_inference_steps=25,
                            guidance_scale=4,
                            num_images_per_prompt=1,
                            save_path="./synth_data/",
                            class_prompts=["NORMAL", "CNV", "DRUSEN", "DME"],
                            height=args.resolution,
                            width=args.resolution,
                        )

                        del sdm
                        torch.cuda.empty_cache()  
                        if accelerator.is_main_process:
                            # EVAL with a imagenet pretrained inception network
                            # create new folder named ALL
                            import shutil
                            os.makedirs("./synth_data/ALL", exist_ok=True)

                            for c in ["NORMAL", "CNV", "DRUSEN", "DME"]:
                                for f in os.listdir(f"./synth_data/{c}"):
                                    shutil.copy(f"./synth_data/{c}/{f}", f"./synth_data/ALL/{f}")

                            score = fid.compute_fid("./synth_data/ALL/", dataset_name="val",
                                    mode="clean", dataset_split="custom")
                            
                            wandb.log({"FID": score}, step=global_step)
                        vae.to(accelerator.device)
                        unet.to(accelerator.device)
                        text_encoder.to(accelerator.device)
                        curr_step_evaluated = global_step  

           
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
      if args.dump_only_text_encoder:
         txt_dir=args.output_dir + "/text_encoder_trained"
         if args.train_only_text_encoder:            
             pipeline = StableDiffusionPipeline.from_pretrained(
                 args.pretrained_model_name_or_path,
                 text_encoder=accelerator.unwrap_model(text_encoder),
             )
             pipeline.save_pretrained(args.output_dir)     
         else:
             if not os.path.exists(txt_dir):
               os.mkdir(txt_dir)
             pipeline = StableDiffusionPipeline.from_pretrained(
                 args.pretrained_model_name_or_path,
                 unet=accelerator.unwrap_model(unet),
                 text_encoder=accelerator.unwrap_model(text_encoder),
             )
             pipeline.text_encoder.save_pretrained(txt_dir)

      elif args.train_only_unet:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)

        txt_dir=args.output_dir + "/text_encoder_trained"
        if os.path.exists(txt_dir):
           subprocess.call('rm -r '+txt_dir, shell=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()
