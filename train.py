import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
from tqdm import tqdm
import cv2

from utils.resnet34_pretrained import ResNet34
from utils.resnet import ResNet101
from torch.utils.tensorboard import SummaryWriter
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import PAFBDataset
from utils.loader import Loader


def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:, None, None, None]


def create_image(representation):
    B, C, H, W = representation.shape

    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals) / (robust_max_vals - robust_min_vals)
    representation = torch.clamp(255 * representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation


def main(args):
    # datasets, add augmentation to training set
    training_dataset = PAFBDataset(args.training_dataset, augmentation=True)
    validation_dataset = PAFBDataset(args.validation_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, args, device=args.device)
    validation_loader = Loader(validation_dataset, args, device=args.device)
    loaders = {"train": training_loader, "valid": validation_loader}

    # model, and put to device
    model = ResNet101(img_channel=36, num_classes=10)
    model = model.to(args.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(args.log_dir)

    iteration = 0
    min_validation_loss = 20
    sum_accuracy = 0
    sum_loss = 0

    for epoch in range(args.num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            if phase == "valid" and epoch % 5 != 0:
                continue

            for events, labels in tqdm(loaders[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    pred_labels = model(events)
                    loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

                    sum_accuracy += accuracy.item()
                    sum_loss += loss.item()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        iteration += 1

            if phase == "valid" and loss < min_validation_loss:
                min_validation_loss = loss
                state_dict = model.state_dict()

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, "./checkpoint/model_best.pth")
                print("New validation best at ", loss)

            loss = sum_loss / len(loaders[phase])
            accuracy = sum_accuracy / len(loaders[phase])

            if phase == "train":
                print(f"{phase} iteration {iteration}  Loss {loss:.4f}  Accuracy {accuracy:.4f}")
            else:
                print(f"{phase} Loss {loss:.4f}  Accuracy {accuracy:.4f}")

            writer.add_scalar(f"{phase}/accuracy", accuracy, iteration)
            writer.add_scalar(f"{phase}/loss", loss, iteration)

            representation_vizualization = create_image(events)
            writer.add_image(f"{phase}/representation", representation_vizualization, iteration)

            sum_accuracy = 0
            sum_loss = 0

        if phase == "train" and epoch % 10 == 0:
            lr_scheduler.step()

        if epoch % args.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "./checkpoint/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="", required=True)
    parser.add_argument("--training_dataset", default="", required=True)

    # logging options
    parser.add_argument("--log_dir", default="", required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    args = parser.parse_args()

    assert os.path.isdir(dirname(args.log_dir)), f"Log directory root {dirname(args.log_dir)} not found."
    assert os.path.isdir(
        args.validation_dataset), f"Validation dataset directory {args.validation_dataset} not found."
    assert os.path.isdir(args.training_dataset), f"Training dataset directory {args.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {args.num_epochs}\n"
          f"batch_size: {args.batch_size}\n"
          f"device: {args.device}\n"
          f"log_dir: {args.log_dir}\n"
          f"training_dataset: {args.training_dataset}\n"
          f"validation_dataset: {args.validation_dataset}\n"
          f"----------------------------")

    main(args)

