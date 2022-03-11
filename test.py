from baseline import baseline_advanced
import dataloader
import torchvision
import torch
import torch.multiprocessing as mp
from time import time
from torch.utils.tensorboard import SummaryWriter
from params import N_EPOCHS, N_EXPERIMENTS, N_QUERY, N_SUPPORT
from utils import running_average
import argparse
import numpy as np


def main(args):
    # Load data and pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    CIFAR100 = dataloader.few_shot_CIFAR100()
    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone_output_feature = 1000
    if not args.use_fc:
        backbone_output_feature = 2048
        res50_model.fc = torch.nn.Identity()
    res50_model.to(device)
    result = []
    cls_range = list(range(args.n_cls_start, args.n_cls_end+1, 5))
    for num_cls in cls_range:
        avg_acc = running_average()
        for exp_id in range(args.n_exp):
            # Training
            support, query = CIFAR100.sample_episode(num_cls, N_SUPPORT, N_QUERY)
            train_loader = torch.utils.data.DataLoader(support, 
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1)
            test_loader = torch.utils.data.DataLoader(query, 
                batch_size=8, 
                shuffle=False,
                num_workers=4)
            support_loader = torch.utils.data.DataLoader(support, 
                batch_size=1,
                shuffle=False,
                num_workers=1)
            if args.use_adv_baseline:
                # Calculate init weights
                res50_model.eval()
                construct_weight = {}
                for img, label in support_loader:
                    img, label = img.to(device), label.to(device)
                    with torch.no_grad():
                        output = res50_model(img)
                    if label.item() not in construct_weight:
                        construct_weight[label.item()] = running_average()
                    construct_weight[label.item()].update(output)
                init_weight = torch.zeros((num_cls, backbone_output_feature))
                for i in construct_weight:
                    init_weight[i, :] = construct_weight[i].value
                linear_probe = baseline_advanced(num_cls, init_weight)
                linear_probe.to(device)
            else:
                linear_probe = torch.nn.Linear(backbone_output_feature, num_cls, device=device)
            model = torch.nn.Sequential(res50_model, linear_probe)
            crit = torch.nn.CrossEntropyLoss().to(device)
            if args.finetune_backbone:
                optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                             weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(linear_probe.parameters(), args.lr,
                                        weight_decay=args.weight_decay)
            writer = SummaryWriter()
            for epoch in range(N_EPOCHS):
                train_epoch(res50_model, linear_probe, epoch, crit, train_loader, test_loader, optimizer,
                            writer, device, verbal=args.verbose, finetune_backbone=args.finetune_backbone)
            _, acc = validate(torch.nn.Sequential(res50_model, linear_probe), test_loader, device)
            avg_acc.update(acc.item())
        result.append(avg_acc.value)
    save_result = np.array(result)
    save_cls_num = np.array(cls_range)
    np.save(args.output, np.vstack((save_cls_num, save_result)))
    return save_result

def train_epoch(backbone, probe, epoch, crit, train_loader, test_loader, optimizer, writer, device, scheduler=None, 
                verbal=False, finetune_backbone=False):
    ep_start = time()
    total_loss = running_average()
    probe.train()
    if finetune_backbone:
        backbone.train()
    else:
        backbone.eval()
    for batch_idx, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        if finetune_backbone:
            output = backbone(img)
        else:
            with torch.no_grad():
                output = backbone(img)
        output = probe(output)
        loss = crit(output, label)
        optimizer.zero_grad()
        loss.backward()
        total_loss.update(loss.item(), img.shape[0])
        optimizer.step()
    if verbal:
        print("Epoch {} total loss {} time {}".format(epoch, total_loss.value, time()-ep_start))
    val_loss, val_acc = validate(torch.nn.Sequential(backbone, probe), test_loader, device, verbal)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="result.npy")
    parser.add_argument("--n-exp", type=int, default=N_EXPERIMENTS)
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--use-fc", action="store_true")
    parser.add_argument("--n-cls-start", type=int, default=5)
    parser.add_argument("--n-cls-end", type=int, default=50)
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--use-adv-baseline", action="store_true")
    args = parser.parse_args()
    main(args)