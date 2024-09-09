# Author: Mattes Warning
import os
import torch
import argparse

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from model.Model import Model
from dataset.Dataset import MedMNIST2D
from torch.utils.data import DataLoader

def train(model, optimizer, train_loader, val_loader, args):

    criterion = nn.BCEWithLogitsLoss()

    best_eval_loss = 9999

    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()

        for imgs, canny_imgs, targets in train_loader:

            optimizer.zero_grad()

            outputs = model(imgs, canny_imgs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            avg_loss = epoch_loss / (i + 1)
            wandb.log({"train-loss": avg_loss})

            if epoch % args.eval_every == 0:
                eval_loss = evaluation_loop(model, val_loader)
                wandb.log({"eval-loss": eval_loss})

                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(model)


def evaluation_loop(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0

    with torch.no_grad():
        for imgs, canny_imgs, targets in loader:

            outputs = model(imgs, canny_imgs)
            loss = criterion(outputs, targets)

            running_loss += loss

        avg_loss = running_loss / (i + 1)

        return avg_loss


def save_checkpoint(model):
    out_dir = os.path.join(f"ckpts", f'weights.pt')
    torch.save({"model_state_dict": model.state_dict()}, out_dir)



def main(args):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = MedMNIST2D(data_path=args.data_path, split='train', transform=data_transform)
    val_dataset = MedMNIST2D(data_path=args.data_path, split='val', transform=data_transform)
    test_dataset = MedMNIST2D(data_path=args.data_path, split='test', transform=data_transform)

    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)


    optimizer = optim.adam()
    model = Model()

    train(model, optimizer, train_loader, val_loader, args)

    # TODO: test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()

    main(args)
