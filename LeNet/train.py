import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        with torch.no_grad():
            outputs = net(val_image)  # [batch, 10]
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, step + 1, running_loss / 500, accuracy))
            running_loss = 0.0

    print('Finished Training')

    save_path = './LeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
