import torch
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np
from alexnet_mnist import AlexNet_MNIST

# configuration, gonna fix soon with argparse if I have the time
MODEL_PATH = 'models/alexnet_mnist_model.pth'
IMAGE_PATH = 'data/digits/1.png'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the pretrained model
model = AlexNet_MNIST().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(28),
    transforms.Grayscale(),
    transforms.ToTensor()
])

img = Image.open(IMAGE_PATH)
img = transform(img)
img = img.view(1, 1, 28, 28).to(DEVICE)

idx_to_digits = {
    0: '零',
    1: '一',
    2: '二',
    3: '三',
    4: '四',
    5: '五',
    6: '六',
    7: '七',
    8: '八',
    9: '九'
}

start = time()
with torch.no_grad():
    model.eval()
    result = model(img)

end = time()
print('Local GPU inference time in Win 10 using Pytorch model: %.2f minutes' % ((end - start) / 60))
ps = torch.exp(result)
topk, topdigits = ps.topk(10, dim=1)
for i in range(10):
    print("Prediction", '{:2d}'.format(i+1), ":", '{:1}'.format(idx_to_digits[topdigits.cpu().numpy()[0][i]]), ", Digit: ", topdigits[0][i].cpu().numpy(), " Score: ", topk.cpu().detach().numpy()[0][i])



