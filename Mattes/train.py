# Author: Mattes Warning
import os
import torch
import argparse
import sys
myDir = os.getcwd()
sys.path.append(myDir)

from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())

sys.path.append(a)

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from model.Model import Model, Model2
from dataset.Dataset import MedMNIST2D
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, train_loader, val_loader, args):

    weight = torch.tensor([4.39, 2.78, 1.3, 12.5, 1.28, 0.21, 10], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)# weight=

    best_eval_loss = 9999

    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        
        i = 0
        for imgs, canny_imgs, targets in train_loader:
            imgs = imgs.to(device)
            canny_imgs = canny_imgs.to(device)
            targets = targets.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(imgs)
            #outputs = model(imgs, canny_imgs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            i+=1
            epoch_loss += loss
            
            #if i == 5:
            #    print(outputs, targets)

        avg_loss = epoch_loss / (i + 1)
            #wandb.log({"train-loss": avg_loss})
        print("Train_loss:", avg_loss.item())

        if epoch % args.eval_every == 0:
                eval_loss = evaluation_loop(model, val_loader)
                print("eval_loss:", eval_loss.item())
                #wandb.log({"eval-loss": eval_loss})

        if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(model)


def evaluation_loop(model, loader):
    model.eval()
    weight = torch.tensor([4.34, 2.78, 1.3, 12.5, 1.28, 0.21, 10], device=device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        i = 0
        for imgs, canny_imgs, targets in loader:
            
            imgs = imgs.to(device)
            canny_imgs = canny_imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            #outputs = model(imgs, canny_imgs)
            loss = criterion(outputs, targets)
            
            class_predictions = torch.argmax(outputs, dim=1)
            true_classes = torch.argmax(targets, dim=1)
            correct_predictions = (class_predictions == true_classes)
            acc = correct_predictions.sum().item() / correct_predictions.size(0)

            running_acc += acc
            running_loss += loss
            i += 1

        avg_loss = running_loss / (i + 1)
        avg_acc = running_acc / (i+1)
        print("Eval Acc:", avg_acc)

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

    train_loader = DataLoader(train_dataset, batch_size=128)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset)

    model = Model2().to(device)
    optimizer = optim.Adam(model.parameters())
    

    train(model, optimizer, train_loader, val_loader, args)

    # TODO: test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dermamnist.npz")
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--eval_every", default=10)
    args = parser.parse_args()

    main(args)
