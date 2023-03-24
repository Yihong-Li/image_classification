import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils import data

from data import MyDataSet
from model import mobile_vit_small
from utils import read_split_data, train_one_epoch, evaluate, get_params_groups


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # tensorboard --logdir=runs
    tb_writer = SummaryWriter()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
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
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = mobile_vit_small(num_classes=args.num_classes).to(device)

    if args.weights != "" and args.fine_tune:
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers and args.fine_tune:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    param_gp = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(param_gp, lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    # OneCycleLR.step() should be called after each mini-batch instead of after each epoch
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, steps_per_epoch=len(train_loader),
    #                                    epochs=args.epochs, pct_start=0.2)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print(f'train_acc: {train_acc}, val_acc: {val_acc}')

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.flush()

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/mobile_vit_small_best.pth")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=3e-4)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="../data_set/flower_data/flower_photos")
    parser.add_argument('--model-name', default='mobile_vit_small')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--fine_tune', type=bool, default=True)
    parser.add_argument('--weights', type=str, default='./mobilevit_s.pt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
