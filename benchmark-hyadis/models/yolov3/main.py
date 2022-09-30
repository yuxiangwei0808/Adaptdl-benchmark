import argparse
import time

import hyadis
import torch
import torch.optim as optim
from hyadis.elastic import ReturnObjects
from hyadis.elastic.runner import ElasticRunner
from hyadis.utils import get_job_id, get_logger
from tqdm import tqdm

import config.yolov3_config_voc as cfg
import utils.datasets as data
from model.loss.yolo_loss import YoloV3Loss
from model.yolov3 import Yolov3


@hyadis.elastic.initialization(True)
def init_all_workers(args, batch_size=None, learning_rate=None, **kwargs):
    model = Yolov3().cuda(torch.cuda.current_device())

    train_loader = torch.utils.data.DataLoader(
        data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]),
        batch_size=cfg.TRAIN["BATCH_SIZE"],
        num_workers=cfg.TRAIN["NUMBER_WORKERS"],
        drop_last=True,
        shuffle=True,
    )

    criterion = YoloV3Loss(
        anchors=cfg.MODEL["ANCHORS"],
        strides=cfg.MODEL["STRIDES"],
        iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN["LR_INIT"],
        momentum=cfg.TRAIN["MOMENTUM"],
        weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
    )

    return ReturnObjects(
        model=model, data=train_loader, optim=optimizer, criterion=criterion, foo=None
    )


@hyadis.elastic.train_step
def step(data, engine, **kwargs):
    get_logger().info(f"{kwargs}")
    engine.train()
    imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = (
        d.cuda(torch.cuda.current_device()) for d in data
    )

    engine.zero_grad()
    p, p_d = engine(imgs)
    loss, loss_giou, loss_conf, loss_cls = engine.criterion(
        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    engine.backward(loss)
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

    parser.add_argument(
        "--weight_path", required=True, type=str, help="weight file path"
    )

    main(parser.parse_args())
