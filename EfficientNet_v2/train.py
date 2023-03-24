import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torchvision import transforms, models
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnetv2_s
from data import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    tb_writer = SummaryWriter()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=2,
                                   collate_fn=train_dataset.collate_fn)

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0,
                                 collate_fn=val_dataset.collate_fn)

    if args.fine_tune:
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, args.num_classes)
        model.to(device)
    else:
        model = efficientnetv2_s(num_classes=args.num_classes)
        model.to(device)

    # 是否冻结权重
    if args.freeze_layers and args.fine_tune:
        for name, param in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                param.requires_grad_(False)
            else:
                print("training {}".format(name))

    param = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print(f'train_acc: {train_acc}, val_acc: {val_acc}')

        # write to tensorboard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.flush()
        # save
        torch.save(model.state_dict(), "./efficientnetv2_s.pth".format(epoch))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArgParser for efficient net')
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-path', type=str,
                        default="../data_set/flower_data/flower_photos")
    parser.add_argument('--fine-tune', type=bool, default=True,
                        help='True if fine tune on Imagenet pretrained model')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weight-decay', type=int, default=1e-5)

    opt = parser.parse_args()
    main(opt)
