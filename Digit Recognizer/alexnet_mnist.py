import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import time

class AlexNet_MNIST(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet_MNIST, self).__init__()
        self.layer1 = nn.Sequential( # Input 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32 x 28 x 28
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 14 14
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential( 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 x 7 x 7
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*7*7
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

EPOCHS = 10
BATCH_SIZE = 64
LR = .01
MOMENTUM = .9
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform)
 
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

cnn = AlexNet_MNIST().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=MOMENTUM)

def train():
    start = time()
    for epoch in range(EPOCHS):
        sum_loss = .0
        for i, (digits, labels) in enumerate(trainloader):
            digits, labels = digits.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = cnn(digits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for (digits, labels) in testloader:
                digits, labels = digits.to(DEVICE), labels.to(DEVICE)
                outputs = cnn(digits)
                _, predicts = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicts == labels).sum()
            print('The recognition accuracy of the %d epoch is: %d%%' % (epoch + 1, (100 * correct // total)))
    end = time()
    print('Total training time of this model: %.2f minutes' % ((end - start) / 60))
    torch.save(cnn.state_dict(), './models/alexnet_mnist_model.pth')

if __name__ == 'main':
    train()