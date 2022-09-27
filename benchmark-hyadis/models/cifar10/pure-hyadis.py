import argparse
import time
import hyadis
import ray
import torch
from hyadis.elastic.parallel import ElasticDistributedDataLoader, ElasticDistributedDataParallel, ElasticOptimizer
from hyadis.utils import get_job_id, get_logger
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm
from hyadis.utils import get_logger
from hyadis.elastic import ReturnObjects


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int)
    return parser.parse_args()


@hyadis.elastic.initialization(True)
def test_init(path, batch_size=None, learning_rate=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize((224, 224))
    ])

    training_data = datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transform,
    )

    model = resnet18()
    model.to(torch.cuda.current_device())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = ElasticDistributedDataLoader(dataset=training_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=False)

    return ReturnObjects(model=model, data=dataloader, optim=optimizer, criterion=criterion)


@hyadis.elastic.train_step
def test_train(data, engine, **kwargs):
    engine.train()
    x, y = data
    x, y = x.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())
    engine.zero_grad()
    o = engine(x)
    loss = engine.criterion(o, y)
    engine.backward(loss)
    engine.step()
    return ReturnObjects(loss=loss)


def train(epoch, runner):
    logger = get_logger()
    runner.set_epoch(epoch)
    epoch_end = False
    num_samples = 0
    start_time = time.time()
    progress = tqdm(total=len(runner.data))
    while not epoch_end:
        epoch_end = test_train()
        if not epoch_end:
            num_samples += runner.global_batch_size
            progress.update(1)
    progress.close()

    end_time = time.time()
    logger.info(f"[Epoch {epoch}] Loss = {runner.reduced_loss.item():.3f} | " + f"LR = {runner.learning_rate:.3f} | " +
                f"Throughput = {num_samples/(end_time - start_time):.3f}")


def main():
    args = get_args()
    job_start = time.time()
    ray.init()
    print(f"{get_job_id()} (size={args.num_workers}) start")
    runner = hyadis.elastic.init(args.num_workers,
                                 use_gpu=True,
                                 autoscale=False,
                                 global_batch_size=args.batch_size * args.num_workers,
                                 learning_rate=0.01)
    print(f"{get_job_id()} (size={runner.size()}) ready")

    test_init(args.data_path)

    train(0, runner)
    runner.resize(4)
    for epoch in range(1, 6):
        train(epoch, runner)

    job_end = time.time()
    print(f"{get_job_id()} (size={runner.size()}) complete: used time = {job_end - job_start:.3f}")

    runner.shutdown()


if __name__ == "__main__":
    main()
