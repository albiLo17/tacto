from torch import nn
import torch

def train(trainLoader, model, optimizer, epoch, args, device, writer, modality):
    model.train()
    criterion = nn.CrossEntropyLoss()

    lossTrainList = []
    running_loss = 0.0

    for i, data in enumerate(trainLoader):
        x = {}
        for k in modality:
            x[k] = data[k].to(device)

        label = data["label"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        lossTrainList.append(loss.item())
        running_loss = (running_loss * i + loss.item()) / (i + 1)

    return lossTrainList, running_loss


def evaluation(testLoader, model, optimizer, epoch, args, device, writer, modality):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total, correct = 0, 0
    running_loss = 0.0

    for i, data in enumerate(testLoader):

        x = {}
        for k in modality:
            x[k] = data[k].to(device)

        label = data["label"].to(device)

        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, label)
            running_loss = (running_loss * i + loss.item()) / (i + 1)

            pred = outputs.argmax(axis=-1)

            total += label.size(0)
            correct += (pred == label).sum().item()
            print("\r Evaluation: ", correct / total, end=" ")

    acc = correct / total
    return acc, running_loss