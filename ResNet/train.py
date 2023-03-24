import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34, resnet50
import torchvision.models as models


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on {}.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)

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
                                               num_workers=2)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    def get_pretrained_resnet(model_type):
        model = getattr(models, model_type)(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)
        print("Loading model pretrained on ImageNet1K_V1...")
        return model

    fine_tune = False
    model_name = 'resnet50'
    if fine_tune:
        net = get_pretrained_resnet(model_name)
        net.to(device)
    else:
        net = resnet50(num_classes=5)
        net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=2e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    epochs = 50
    best_acc = 0.0
    save_path = f'./{model_name}.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            y_hat = net(images.to(device))
            loss = loss_function(y_hat, labels.to(device))
            loss.backward()
            optimizer.step()
            train_acc += torch.eq(torch.argmax(y_hat, dim=1), labels.to(device)).sum().item()

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
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

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
