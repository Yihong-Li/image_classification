import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet
import torchvision.models as models


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on {}.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    fine_tune = True
    if not fine_tune:
        net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    else:
        net = models.googlenet(pretrained=True)
        net.fc = torch.nn.Linear(net.fc.in_features, 5)
        print("Loading model pretrained on ImageNet1K_V1...")

    net.to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-6)

    epochs = 20
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()
            if fine_tune:
                y_hat = net(images.to(device))
                loss = loss_function(y_hat, labels)
            else:
                y_hat, aux_2, aux_1 = net(images.to(device))
                loss0 = loss_function(y_hat, labels)
                loss1 = loss_function(aux_1, labels)
                loss2 = loss_function(aux_2, labels)
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()
            train_acc += torch.eq(torch.argmax(y_hat, dim=1), labels).sum().item()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        scheduler.step()
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        train_accurate = train_acc / train_num
        val_accurate = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}, '
              f'train_accuracy: {train_accurate:.3f}, val_accuracy: {val_accurate:.3f}')
        print()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
