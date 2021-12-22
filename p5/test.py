from dataset import DogAndCat
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt


def test(model, test_loader):
    acc = 0.0
    for data, label in tqdm(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        output = model(data)
        _, pred = output.max(1)
        acc += (pred == label).sum().item()
    acc /= len(test_loader.dataset)
    return acc*100


def graph(origin_acc, aug_acc):
    plt.figure()
    pred = [origin_acc, aug_acc]
    label = ('Before Random Flip', 'After Random Flip')
    plt.bar(label, pred)
    plt.savefig('compare.png')


if __name__ == '__main__':
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    test_data = DogAndCat(img_dir='./dog_and_cat', transform=test_transform, mode='test')
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('resnet50.pth'))
    origin_acc = test(model, test_loader)
    model.load_state_dict(torch.load('resnet50_aug.pth'))
    aug_acc = test(model, test_loader)
    graph(origin_acc, aug_acc)
