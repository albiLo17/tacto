from torch import nn
import torch

from experiments.grasp_stability.albi.utils import crop_like

def train(args, trainLoader, model, optimizer, device,  modality, criterion=nn.CrossEntropyLoss(), classification=True):
    model.train()
    criterion = criterion

    lossTrainList = []
    running_loss = 0.0
    if not classification:
        reconstruction = {}
        for k in modality:
            reconstruction[k] = 0.0

        kl = 0.0

    for i, data in enumerate(trainLoader):
        x = {}
        label = []
        mod_label = {}
        for k in modality:
            x[k] = data[k].to(device)
            label = torch.cat((label, data[k])) if len(label) > 0 else data[k]
            mod_label[k] = data[k].to(device)

        if classification:
            label = data["label"].to(device)
        else:
            label = mod_label

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(x)

        # Crop the outputs of the models to match the initial images

        if not classification:
            if len(outputs) == 3:
                outputs[0] = crop_like(outputs[0], label)
            else:
                outputs = crop_like(outputs, label)


        loss = criterion(outputs, label)
        if not classification:
            if type(loss) == list:
                rec_losses, kl_loss = loss
            else:
                rec_losses = loss

            tot_rec_loss = 0
            for k in modality:
                tot_rec_loss += rec_losses[k]
                reconstruction[k] = (reconstruction[k] * i + rec_losses[k].item()) / (i + 1)

            rec_loss = tot_rec_loss


            if args.deterministic == 0:
                kl = (kl * i + args.kl * kl_loss.item())/ (i+1)
                loss = rec_loss + args.kl * kl_loss
            else:
                loss = rec_loss


        loss.backward()
        optimizer.step()

        lossTrainList.append(loss.item())
        running_loss = (running_loss * i + loss.item()) / (i + 1)

    if classification:
        return lossTrainList, running_loss
    else:
        return lossTrainList, [running_loss, reconstruction, kl]


def evaluation(args, testLoader, model, device, modality, criterion=nn.CrossEntropyLoss(), classification=True):
    model.eval()
    criterion = criterion


    total, correct = 0, 0
    running_loss = 0.0
    # print("Before batch")
    # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))

    for i, data in enumerate(testLoader):

        # print("Before loading min-batch")
        # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))

        x = {}
        label = []
        mod_label = {}
        labeled_images = {'ground_truth': {}, 'predictions': {}}

        for k in modality:
            x[k] = data[k].to(device)
            label = torch.cat((label, data[k])) if len(label) > 0 else data[k]
            mod_label[k] = data[k].to(device)
            labeled_images['ground_truth'][k] = data[k]
            labeled_images['predictions'][k] = []


        if classification:
            label = data["label"].to(device)
        else:
            label = mod_label

        # print("After loading min-batch")
        # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))

        with torch.no_grad():
            outputs = model(x)

            if not classification:
                if len(outputs) == 3:
                    outputs[0] = crop_like(outputs[0], label)
                    for k in modality:
                        labeled_images['predictions'][k] = outputs[0][k]
                else:
                    if isinstance(outputs, dict):
                        outputs = crop_like(outputs, label)
                        for k in modality:
                            labeled_images['predictions'][k] = outputs[k]
                    else:
                        outputs = crop_like(outputs, label)

            loss = criterion(outputs, label)

            # TODO: TO BE MODIFIED, compute test loss
            # if type(loss) == list:
            #     rec_losses, kl_loss = loss
            #
            #     if len(modality) == 1:
            #         rec_loss = rec_losses
            #     else:
            #         tot_rec_loss = 0
            #         for k in modality:
            #             tot_rec_loss += rec_losses[k]
            #         rec_loss = tot_rec_loss



            if classification:
                pred = outputs.argmax(axis=-1)
                total += label.size(0)
                correct += (pred == label).sum().item()
                print("\r Evaluation: ", correct / total, end=" ")


    if classification:
        acc = correct / total
        return acc, running_loss

    return labeled_images, running_loss


