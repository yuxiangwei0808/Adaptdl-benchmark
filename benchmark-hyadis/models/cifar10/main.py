"""Train CIFAR10 with PyTorch."""
import argparse
import time

import hyadis
import torch
import torchvision
import torchvision.transforms as transforms
from hyadis.elastic import ReturnObjects
from hyadis.elastic.parallel import (
    ElasticDistributedDataLoader,
)
from hyadis.elastic.runner import ElasticRunner
from hyadis.utils import get_job_id, get_logger
from tqdm import tqdm

def get_model_from_string(model_string):
    """Candidates:
    net = VGG('VGG19')
    net = ResNet18()
    net = PreActResNet18()
    net = GoogLeNet()
    net = DenseNet121()
    net = ResNeXt29_2x64d()
    net = MobileNet()
    net = MobileNetV2()
    net = DPN92()
    net = ShuffleNetG2()
    net = SENet18()
    net = ShuffleNetV2(1)
    """
    from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    return eval(model_string)()


@hyadis.elastic.initialization(True)
def init_all_workers(args, batch_size=None, learning_rate=None, **kwargs):
    model = get_model_from_string(args.model)
    model.to(torch.cuda.current_device())

    train_loader = ElasticDistributedDataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    return ReturnObjects(
        model=model, data=train_loader, optim=optimizer, criterion=criterion
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


class LoggingRunner:
    def __init__(self, *args, **kwargs):
        self.job_start = time.time()
        print(f"job id {get_job_id()} start")
        self.runner = hyadis.elastic.init(*args, **kwargs)
        print(f"job id {get_job_id()} ready")

    def __enter__(self):
        return self.runner

    def __exit__(self, type, value, traceback):
        print(
            f"{get_job_id()} complete: used time = {time.time() - self.job_start:.3f} s"
        )
        self.runner.shutdown()


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
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="training data path, same at all workers.",
    )
    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--learning_rate", default=0.08, type=float, help="learning rate"
    )
    parser.add_argument("--epochs", default=90, type=int, help="number of epochs")
    parser.add_argument(
        "--model", default="ResNet18", type=str, help="model name in model.py"
    )

    main(parser.parse_args())
