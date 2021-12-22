import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import DogAndCat
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train(model, epoch, train_loader, valid_loader):
    writer = SummaryWriter('./runs/resnet50_aug')
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    valid_loss_min = np.Inf
    for i in range(epoch):
        print('*' * 10, 'epoch {}'.format(i + 1), '*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        recard_loss = 0.0
        recard_acc = 0.0
        model.train()
        for j, data in enumerate(tqdm(train_loader)):
            data, label = data
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            recard_loss += loss.item() * label.size(0)
            _, pred = out.max(1)
            running_acc += (pred == label).sum().item()
            recard_acc += (pred == label).sum().item()
            loss.backward()
            optimizer.step()
            if j % 10 == 9:
                writer.add_scalar('training loss', recard_loss/(10*len(pred)), i * len(train_loader) + j)
                writer.add_scalar('training acc', recard_acc/(10*len(pred)), i * len(train_loader) + j)
                recard_loss = 0.0
                recard_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        model.eval()
        for j, data in enumerate(tqdm(valid_loader)):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = criterion(out, y)
            valid_loss += loss.item() * y.size(0)
            _, pred = out.max(1)
            valid_acc += (pred == y).sum().item()
        running_loss = running_loss / len(train_loader.dataset)
        running_acc = running_acc / len(train_loader.dataset) * 100
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset) * 100
        print('Finish {} epoch, loss={:.6f}, acc={:.6f}'.format(i + 1, running_loss, running_acc))
        print('Validation, loss={:.6f}, acc={:.6f}'.format(valid_loss, valid_acc))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'resnet50_aug.pth')
            valid_loss_min = valid_loss
    print('Finished Training')


def loadData(batch_size, train_transform, valid_transform):
    train_data = DogAndCat(img_dir='./dog_and_cat', transform=train_transform, mode='train')
    valid_data = DogAndCat(img_dir='./dog_and_cat', transform=valid_transform, mode='valid')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


if __name__ == '__main__':
    batch_size = 16
    lr = 0.001
    epoch = 10
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    train_loader, valid_loader = loadData(batch_size, train_transform, valid_transform)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    train(model, epoch, train_loader, valid_loader)
