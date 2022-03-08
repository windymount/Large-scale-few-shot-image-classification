from sched import scheduler

from sympy import arg
import dataloader
import torchvision
import torch
from time import time
from torch.utils.tensorboard import SummaryWriter
from utils import running_average
import argparse


def main(device):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0)
    args = parser.parse_args()
    # Load data and pretrained model
    CIFAR100 = dataloader.few_shot_CIFAR100()
    res50_model = torchvision.models.resnet50(pretrained=True)
    res50_model.fc = torch.nn.Identity()
    res50_model.to(device)
    res50_model.eval()
    # linear probe
    linear_probe = torch.nn.Linear(2048, 5, device=device)
    model = torch.nn.Sequential(res50_model, linear_probe)
    # Training
    support, query = CIFAR100.sample_episode(5, 50, 150)
    train_loader = torch.utils.data.DataLoader(support, 
        batch_size=1, 
        shuffle=False,
        num_workers=1)
    test_loader = torch.utils.data.DataLoader(query, 
        batch_size=8, 
        shuffle=False,
        num_workers=4)
    crit = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(res50_model.fc.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=0.0001)
    writer = SummaryWriter()
    for epoch in range(25):
        train_epoch(res50_model, linear_probe, epoch, crit, train_loader, test_loader, optimizer,
                    writer, device)
        


def train_epoch(backbone, probe, epoch, crit, train_loader, test_loader, optimizer, writer, device, scheduler=None):
    ep_start = time()
    total_loss = running_average()
    probe.train()
    for batch_idx, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            output = backbone(img)
        output = probe(output)
        loss = crit(output, label)
        optimizer.zero_grad()
        loss.backward()
        total_loss.update(loss.item(), img.shape[0])
        optimizer.step()
        """"""
        # if batch_idx > 5:break

    print("Epoch {} total loss {} time {}".format(epoch, total_loss.value, time()-ep_start))
    val_loss, val_acc = validate(torch.nn.Sequential(backbone, probe), test_loader, device)
    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_acc", val_acc, epoch)
    writer.add_scalar("train_loss", total_loss.value, epoch)

    if scheduler:
        scheduler.step(val_acc)
    return val_acc


def validate(model, test_loader, device=torch.device("cuda"), verbal=True):
    acc = running_average()
    avg_loss = running_average()
    crit = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            pred = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[1]
            correct = pred.eq(label)
            loss = crit(output, label)
            avg_loss.update(loss, img.shape[0])
            acc.update(correct.float().sum(), img.shape[0])
        if verbal:
            print("Test avg loss:{} acc: {}".format(avg_loss.value, acc.value))
        return avg_loss.value, acc.value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)