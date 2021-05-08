import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm
import cv2

from utils.models import Classifier
from torch.utils.tensorboard import SummaryWriter
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import PAFBDataset
from utils.loader import Loader


def show(img):
    npimg = np.array(img).transpose((1, 2, 0))
    # npimg = np.array(img)

    cv2.imshow('frame', npimg)
    cv2.waitKey(1000)


def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:, None, None, None]


def create_image(representation):
    B, C, H, W = representation.shape

    # representation = representation.view(B, 1, C // 1, H, W)

    # representation = representation.detach().cpu()

    # for frame in representation[0][0]:
    #     show(frame)

    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals) / (robust_max_vals - robust_min_vals)
    representation = torch.clamp(255 * representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation


def FLAGS():
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

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(
        flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"----------------------------")

    return flags


if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    training_dataset = PAFBDataset(flags.training_dataset, augmentation=True)
    validation_dataset = PAFBDataset(flags.validation_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)

    # model, and put to device
    model = Classifier()
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        sum_accuracy = 0
        sum_loss = 0

        model = model.train()
        for events, labels in tqdm.tqdm(training_loader):
            optimizer.zero_grad()

            pred_labels = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

            loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
            lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)

        representation_vizualization = create_image(events)
        writer.add_image("training/representation", representation_vizualization, iteration)

        if i % 5 == 0:
            print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
            sum_accuracy = 0
            sum_loss = 0
            model = model.eval()

            for events, labels in tqdm.tqdm(validation_loader):
                with torch.no_grad():
                    pred_labels = model(events)
                    loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

                sum_accuracy += accuracy
                sum_loss += loss

            validation_loss = sum_loss.item() / len(validation_loader)
            validation_accuracy = sum_accuracy.item() / len(validation_loader)

            writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
            writer.add_scalar("validation/loss", validation_loss, iteration)

            # visualize representation
            representation_vizualization = create_image(events)
            writer.add_image("validation/representation", representation_vizualization, iteration)

            print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                state_dict = model.state_dict()

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, "./checkpoint/model_best.pth")
                print("New best at ", validation_loss)

        if i % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "./checkpoint/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))
