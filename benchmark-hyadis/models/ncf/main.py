import os
import time

import numpy
import torch
import hyadis
import argparse
import data_utils

from tqdm import tqdm
import torch.optim as optim
from hyadis.elastic.runner import ElasticRunner
from hyadis.elastic.parallel import (
    ElasticDistributedDataLoader,
)
from hyadis.elastic import ReturnObjects
from hyadis.utils import get_job_id, get_logger

from model import NCF
import config


@hyadis.elastic.initialization(True)
def init_all_workers(args, batch_size=None, learning_rate=None, **kwargs):
    if config.model == "NeuMF-pre":
        assert os.path.exists(config.GMF_model_path), "lack of GMF model"
        assert os.path.exists(config.MLP_model_path), "lack of MLP model"
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None
    train_data, _, user_num, item_num, train_mat = data_utils.load_all()

    model = NCF(
        user_num,
        item_num,
        args.factor_num,
        args.num_layers,
        args.dropout,
        config.model,
        GMF_model,
        MLP_model,
    )
    model.cuda(torch.cuda.current_device())

    train_loader = ElasticDistributedDataLoader(
        data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)

    return ReturnObjects(
        model=model, data=train_loader, optim=optimizer, criterion=criterion
    )


@hyadis.elastic.train_step
def step(data, engine, **kwargs):
    engine.train()
    user, item, label = data
    user = user.cuda(torch.cuda.current_device())
    item = item.cuda(torch.cuda.current_device())
    label = label.float().cuda(torch.cuda.current_device())
    engine.zero_grad()

    prediction = engine(user, item)
    loss = engine.criterion(prediction, label)
    engine.step()
    return ReturnObjects(loss=loss)


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


def train_epoch(epoch: int, runner: ElasticRunner):
    runner.set_epoch(epoch)

    with EpochProgressBar(runner, description=f"Epoch {epoch}") as p:
        epoch_end = False
        runner.data.dataset.ng_sample()
        while not epoch_end:
            epoch_end = step()
            p.update(sample_count=runner.global_batch_size)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int, help="batch size")
    parser.add_argument("--epochs", required=True, type=int, help="number of epochs")
    parser.add_argument(
        "--learning_rate", required=True, type=float, help="learning rate"
    )

    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--factor_num",
        type=int,
        default=32,
        help="predictive factors numbers in the model",
    )

    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of layers in MLP model"
    )
    parser.add_argument(
        "--num_ng", type=int, default=4, help="sample negative items for training"
    )
    parser.add_argument(
        "--test_num_ng",
        type=int,
        default=99,
        help="sample part of negative items for testing",
    )

    main(parser.parse_args())
