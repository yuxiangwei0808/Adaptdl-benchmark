import random
import time
import warnings

import torch
import hyadis
import argparse

from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from hyadis.elastic.runner import ElasticRunner
from hyadis.elastic import ReturnObjects
from hyadis.utils import get_job_id, get_logger
import torchvision.models as models

from datasets import load_from_disk

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@hyadis.elastic.initialization(True)
def init_all_workers(args, batch_size=None, learning_rate=None, **kwargs):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    model = models.__dict__[args.arch]()
    model.cuda(torch.cuda.current_device())

    train_loader = torch.utils.data.DataLoader(
        CustomDataset(
            load_from_disk(args.data)["train"],
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    return ReturnObjects(
        model=model, data=train_loader, optim=optimizer, criterion=criterion
    )


@hyadis.elastic.train_step
def step(data, engine, **kwargs):
    engine.train()
    images, targets = data
    images, targets = images.to(torch.cuda.current_device()), targets.to(
        torch.cuda.current_device()
    )
    engine.zero_grad()
    outputs = engine(images)
    loss = engine.criterion(outputs, targets)
    engine.backward(loss)
    engine.step()
    return ReturnObjects(loss=loss)


def train_epoch(epoch: int, runner: ElasticRunner):
    runner.set_epoch(epoch)

    with EpochProgressBar(runner, description=f"Epoch {epoch}") as p:
        epoch_end = False
        while not epoch_end:
            epoch_end = step()
            p.update(sample_count=runner.global_batch_size)
        # adjust_learning_rate()
        # Use torch.optim.lr_scheduler.StepLR

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, set, transform):
        self.transform = transform
        self.extra_transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.set = set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        image = self.transform(self.set[idx]["image"].convert("RGB"))
        # if image.shape[0] == 1:
        #     image = self.extra_transform(image)
        #     print('image shape after enlarge:', image.shape)
        image = self.normalize(image)
        label = self.set[idx]["label"]
        return image, label


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int, help="batch size")
    parser.add_argument("--epochs", required=True, type=int, help="number of epochs")
    parser.add_argument(
        "--learning_rate", required=True, type=float, help="learning rate"
    )

    parser.add_argument(
        "data",
        metavar="DIR",
        default="/workspace/test/imagenet-1k",
        help="path to dataset",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--autoscale-bsz",
        dest="autoscale_bsz",
        default=False,
        action="store_true",
        help="autoscale batchsize",
    )

    main(parser.parse_args())
