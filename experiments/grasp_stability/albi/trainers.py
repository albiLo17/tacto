from torch import nn
import torch

def train(trainLoader, model, optimizer, device,  modality, criterion=nn.CrossEntropyLoss(), classification=True):
    model.train()
    criterion = criterion

    lossTrainList = []
    running_loss = 0.0

    for i, data in enumerate(trainLoader):
        x = {}
        label = []
        for k in modality:
            x[k] = data[k].to(device)
            label = torch.cat((label,data[k])) if len(label) > 0 else data[k]

        if classification:
            label = data["label"]
        label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        lossTrainList.append(loss.item())
        running_loss = (running_loss * i + loss.item()) / (i + 1)

    return lossTrainList, running_loss


def evaluation(testLoader, model, device, modality, criterion=nn.CrossEntropyLoss(), classification=True):
    model.eval()
    criterion = criterion


    total, correct = 0, 0
    running_loss = 0.0

    for i, data in enumerate(testLoader):

        x = {}
        label = []
        images = {'ground_truth': [], 'predictions': []}

        for k in modality:
            x[k] = data[k].to(device)
            label = torch.cat((label, data[k])) if len(label) > 0 else data[k]
        images['ground_truth'] = label

        if classification:
            label = data["label"]
        label.to(device)

        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, label)
            running_loss = (running_loss * i + loss.item()) / (i + 1)

            # Get last batch prediction
            images['predictions'] = outputs

            if classification:
                pred = outputs.argmax(axis=-1)
                total += label.size(0)
                correct += (pred == label).sum().item()
            print("\r Evaluation: ", correct / total, end=" ")

    if classification:
        acc = correct / total
        return acc, running_loss

    return images, running_loss


