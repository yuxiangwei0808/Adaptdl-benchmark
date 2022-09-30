import argparse
from email.policy import default
import json
import logging
import math
import os
import random
import time
from pathlib import Path

from tensorboardX import SummaryWriter
import datasets
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import evaluate
import transformers
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import adaptdl
import adaptdl.torch
import adaptdl.env
import adaptdl.collective
from adaptdl._signal import get_exit_flag
from adaptdl.torch._metrics import get_progress, report_train_metrics, report_valid_metrics

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='qnli',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        default=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
        default='bert-base-uncased',
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./output', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main(task_name=None):
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    if task_name is not None:
        args.task_name = task_name
        
    tb_writer = SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp"))
    # tb_writer = SummaryWriter('./tmp')

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if task_name is None:
        adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Handle the repository creation
    if adaptdl.env.replica_rank() == 0:
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
    # accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.task_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset("glue", args.task_name)
    # else:
    #     # Loading the dataset from local csv or json file.
    #     data_files = {}
    #     if args.train_file is not None:
    #         data_files["train"] = args.train_file
    #     if args.validation_file is not None:
    #         data_files["validation"] = args.validation_file
    #     extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    #     raw_datasets = load_dataset(extension, data_files=data_files)
    # raw_datasets.save_to_disk(f'./data/{args.task_name}/')
    raw_datasets = load_from_disk(f'/benchmark/glue/{args.task_name}')
    # raw_datasets = load_from_disk(f'/home/lcwyx/demo_adaptdl/job_submit/adaptdl/benchmark/models/bert/data/{args.task_name}')

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir if args.cache_dir else None,)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        cache_dir=args.cache_dir,
    ).to(torch.cuda.current_device())

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    if adaptdl.env.replica_rank() == 0:
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    t_total = len(train_dataloader) * args.num_train_epochs

    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataloader = adaptdl.torch.AdaptiveDataLoader(train_dataset, batch_size=args.per_device_train_batch_size, drop_last=True)
    train_dataloader.autoscale_batch_size(384, local_bsz_bounds=(4, 64), gradient_accumulation=True)
    eval_dataloader = adaptdl.torch.AdaptiveDataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if args.with_tracking:
    #     experiment_config = vars(args)
    #     # TensorBoard cannot log Enums, need the raw value
    #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #     accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            pass
            # accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            # accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # model.zero_grad()
    if adaptdl.collective.allreduce(get_exit_flag()):
        exit(143)

    for epoch in adaptdl.torch.remaining_epochs_until(args.num_train_epochs):
        accum = adaptdl.torch.Accumulator()
        # if args.with_tracking:
        #     total_loss = 0
        begin_train = time.time()
        for step, batch in enumerate(train_dataloader):
            batch_size = len(batch['input_ids'][0])
            model.train()
            # We need to skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == starting_epoch:
            #     if resume_step is not None and step < resume_step:
            #         completed_steps += 1
            #         continue
            for key in batch:
                tensor = None
                if isinstance(batch[key], list):
                    for l in batch[key]:
                        if tensor is None:
                            tensor = l.view(batch_size, 1)
                        else:
                            tensor = torch.concat((tensor, l.view(batch_size, 1)), dim=1)
                    batch[key] = tensor.to(torch.cuda.current_device())
                else:
                    batch[key] = batch[key].to(torch.cuda.current_device())

            outputs = model(**batch)
            loss = outputs.loss

            accum["loss_sum"] += loss.item()
            accum["loss_cnt"] += batch['input_ids'].shape[0]

            loss.backward()

            if train_dataloader._elastic.is_sync_step():
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # We keep track of the loss at each epoch
            # if args.with_tracking:
            #     total_loss += loss.detach().float()
            # loss = loss / args.gradient_accumulation_steps
            # if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            #     optimizer.step()
            #     lr_scheduler.step()
            #     optimizer.zero_grad()
            #     progress_bar.update(1)
            completed_steps += 1

            optimizer.step()
            lr_scheduler.step()

            current_step = get_progress()
            # if current_step < args.warmup_steps:
            #     factor = float(current_step) / float(max(1, args.warmup_steps))
            # else:
            #     factor = max(0.0, float(t_total - current_step) / float(max(1, t_total - args.warmup_steps)))
            # for group in optimizer.param_groups:
            #     group["lr"] = args.learning_rate * factor

            # model.zero_grad()

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if args.output_dir is not None:
            #             output_dir = os.path.join(args.output_dir, output_dir)
            #         accelerator.save_state(output_dir)

            train_dataloader.to_tensorboard(tb_writer, current_step, "AdaptDL/Data")
            model.to_tensorboard(tb_writer, current_step, "AdaptDL/Model")
            tb_writer.add_scalar("loss", loss.item(), current_step)

            print(current_step, loss.item())
            # if completed_steps >= args.max_train_steps:
            #     break
        use_time = time.time() - begin_train
        with accum.synchronized():
            try:
                accum["loss_avg"] = accum["loss_sum"] / accum["loss_cnt"]
            except KeyError:
                accum["loss_sum"] = 0
                accum["loss_cnt"] = 0
                accum["loss_avg"] = 0
            tb_writer.add_scalar("Loss/Train", accum["loss_avg"], epoch)
            report_train_metrics(epoch, accum["loss_avg"], per_epoch_time=use_time, samples=accum["loss_cnt"], task_name=args.task_name)
            print("Train:", accum)

        ### evaluation
        logger.info('Begin Evaluatation')
        model.eval()
        samples_seen = 0
        begin_valid = time.time()
        accum_valid = adaptdl.torch.Accumulator()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch_size = len(batch['input_ids'][0])

                for key in batch:
                    tensor = None
                    if isinstance(batch[key], list):
                        for l in batch[key]:
                            if tensor is None:
                                tensor = l.view(batch_size, 1)
                            else:
                                tensor = torch.concat((tensor, l.view(batch_size, 1)), dim=1)
                        batch[key] = tensor.to(torch.cuda.current_device())
                    else:
                        batch[key] = batch[key].to(torch.cuda.current_device())

                outputs = model(**batch)
                valid_loss = outputs.loss

                accum_valid["loss_sum"] += valid_loss.item()
                accum_valid["loss_cnt"] += batch['input_ids'].shape[0]
                
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            # predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            # if accelerator.num_processes > 1:
            #     if step == len(eval_dataloader) - 1:
            #         predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
            #         references = references[: len(eval_dataloader.dataset) - samples_seen]
            #     else:
            #         samples_seen += references.shape[0]
            # metric.add_batch(
            #     predictions=predictions,
            #     references=references,
            # )
            metric.add_batch(predictions=predictions, references=batch['labels'])
        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            logger.info(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    # "train_loss": loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                }
            )
        use_time = time.time() - begin_valid
        with accum_valid.synchronized():
            try:
                accum_valid["loss_avg"] = accum_valid["loss_sum"] / accum_valid["loss_cnt"]
            except KeyError:
                accum_valid["loss_sum"] = 0
                accum_valid["loss_cnt"] = 0
                accum_valid["loss_avg"] = 0
            tb_writer.add_scalar("Loss/Valid", accum_valid["loss_avg"], epoch)
            tb_writer.add_scalar('eval_acc', eval_metric["accuracy"], epoch)
            report_valid_metrics(epoch, accum_valid["loss_avg"], accuracy=eval_metric["accuracy"], per_epoch_time=use_time, samples=accum_valid["loss_cnt"], task_name=args.task_name)

        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
        #         repo.push_to_hub(
        #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
        #         )

        # if args.checkpointing_steps == "epoch":
        #     output_dir = f"epoch_{epoch}"
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)
        #     accelerator.save_state(output_dir)

    # if args.with_tracking:
    #     accelerator.end_training()

    if args.output_dir is not None:
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(
        #     args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        # )
        if adaptdl.env.replica_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    # if args.task_name == "mnli":
    #     # Final evaluation on mismatched validation set
    #     eval_dataset = processed_datasets["validation_mismatched"]
    #     eval_dataloader = DataLoader(
    #         eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    #     )
    #     eval_dataloader = accelerator.prepare(eval_dataloader)
    #
    #     model.eval()
    #     for step, batch in enumerate(eval_dataloader):
    #         outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1)
    #         metric.add_batch(
    #             predictions=accelerator.gather(predictions),
    #             references=accelerator.gather(batch["labels"]),
    #         )
    #
    #     eval_metric = metric.compute()
    #     logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
    tb_writer.export_scalars_to_json(os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp") + f'{args.task_name}_tensorboard.json'))
    tb_writer.close()


if __name__ == "__main__":
    time.sleep(300)
    s = time.time()
    main()
    with open(adaptdl.env.checkpoint_path() + "/overall_results.txt", "a") as f:
        f.write(f'total consume time: {time.time() - s}')