import logging
import os
import random
import time
import numpy as np

import torch
import hyadis
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from hyadis.elastic.runner import ElasticRunner
from hyadis.elastic import ReturnObjects
from hyadis.utils import get_job_id, get_logger
from datasets import load_dataset, load_from_disk

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


logger = logging.getLogger(__name__)

@hyadis.elastic.initialization(True)
def init_all_workers(args, batch_size=None, learning_rate=None, **kwargs):
    if args.seed is not None:
        set_seed(args.seed)

    raw_datasets = load_from_disk(os.path.join(args.data_dir, args.task_name))
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
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
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
    criterion = ...

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

    return ReturnObjects(
        model=model,
        data=train_dataloader,
        optim=optimizer,
        criterion=criterion,
    )


@hyadis.elastic.train_step
def step(data, engine, **kwargs):
    engine.train()
    inputs, targets = data
    inputs, targets = inputs.to(torch.cuda.current_device()), targets.to(
        torch.cuda.current_device()
    )
    engine.zero_grad()
    outputs = engine(inputs)
    loss = engine.criterion(outputs, targets)
    engine.backward(loss)
    engine.step()
    return ReturnObjects(loss=loss)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(epoch: int, runner: ElasticRunner):
    runner.set_epoch(epoch)

    with EpochProgressBar(runner, description=f"Epoch {epoch}") as p:
        epoch_end = False
        while not epoch_end:
            epoch_end = step()
            p.update(sample_count=runner.global_batch_size)


class LoggingRunner:
    def __init__(self, *args, **kwargs):
        self.job_start = time.time()
        self.logger = get_logger()

        self.logger.info(f"job id {get_job_id()} start")
        self.runner = hyadis.elastic.init(*args, **kwargs)
        self.logger.info(f"job id {get_job_id()} ready")

    def __enter__(self):
        return self.runner

    def __exit__(self, type, value, traceback):
        self.logger.info(
            f"{get_job_id()} complete: used time = {time.time() - self.job_start:.3f} s"
        )
        self.runner.shutdown()


class EpochProgressBar:
    def __init__(self, runner, description):
        self.progress_bar = tqdm(total=len(runner.data), desc=description)
        self.num_samples = 0
        self.runner = runner

    def __enter__(self):
        self.start_time = time.time()
        return self

    def update(self, sample_count: int, progress: int = 1):
        self.num_samples += sample_count
        self.progress_bar.update(progress)

    def __exit__(self, type, value, traceback):
        self.progress_bar.close()
        self.end_time = time.time()
        get_logger().info(
            f"Loss = {self.runner.reduced_loss.item():.3f} | "
            + f"LR = {self.runner.learning_rate:.3f} | "
            + f"Throughput = {self.num_samples/(self.end_time - self.start_time):.3f}"
        )


def main(args):
    hyadis.init(address="auto")

    with LoggingRunner(
        args.num_workers,
        use_gpu=True,
        autoscale=False,
        global_batch_size=args.batch_size * args.num_workers,
        learning_rate=args.learning_rate,
    ) as runner:
        init_all_workers(args)
        for epoch in range(args.epochs):
            train_epoch(epoch, runner)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int, help="batch size")
    parser.add_argument("--epochs", required=True, type=int, help="number of epochs")
    parser.add_argument(
        "--learning_rate", required=True, type=float, help="learning rate"
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="qnli",
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
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
        "--data_dir",
        default="",
        type=str,
        help="Where do you want to load the data from disk.",
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
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    main(parser.parse_args())
