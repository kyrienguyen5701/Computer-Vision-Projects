import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn

# get device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# some constants
operators = {
    'laplacian': torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], device=device),
    'sobel_x': torch.tensor([[-1., 0. , 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device),
    'sobel_y': torch.tensor([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]], device=device)
}

# get the image, currently static
(height, width) = (1620, 1080)
img_path = 'images/IMG_1938.png'
input_img = cv2.imread(img_path)
input_img = cv2.resize(input_img, (width, height), interpolation=cv2.INTER_CUBIC)

# grayscale it
print('Pre-processing the picture ...')
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2GRAY) / 255.
gray_img = torch.from_numpy(gray_img).reshape((1, 1) + (height, width)).to(device)

print('The image is being converted ....')

# edge detector to detect text, using PyTorch
def edge_detector_torch(img, type):
    operator = operators[type]
    operator = operator.reshape((1, 1) + operator.shape)
    return nn.functional.conv2d(img.float(), operator, padding=1)

# learning parameters
gain = torch.ones(gray_img.shape, requires_grad=True, device=device)
offset = torch.zeros(gray_img.shape, requires_grad=True, device=device)
k1 = 1
k2 = .1

white_img = torch.ones(gray_img.shape, device=device)
output_img = None

# main part
lr = .01
epochs = 100
optimizer = torch.optim.Adam([gain, offset], lr=lr)
loss = 0
for epoch in range(epochs):
    
    # get rid of old gradients from last loop
    optimizer.zero_grad()

    # hypothesis function
    output_img = gray_img * gain + offset

    #loss components
    white_loss = k1 * torch.sum(torch.pow(output_img - white_img, 2))
    sobel_x_loss = k2 * torch.sum(torch.pow(edge_detector_torch(output_img, 'sobel_x') - edge_detector_torch(gray_img, 'sobel_x'), 2))
    sobel_y_loss = k2 * torch.sum(torch.pow(edge_detector_torch(output_img, 'sobel_y') - edge_detector_torch(gray_img, 'sobel_y'), 2))
    
    # overall loss
    loss = (white_loss + sobel_x_loss + sobel_y_loss)
    
    loss.backward()
    optimizer.step()

# show the loss value
print(loss)

# convert the output from tensor to numpy
output_img = torch.squeeze(output_img).cpu().detach().numpy() * 255
output_img = np.clip(output_img, 0, 255).astype(np.uint8)

# enhance the output
mean = int(np.mean(output_img) + 0.5)
degenerate = np.full(output_img.shape, mean, dtype=np.uint8)
output_img = cv2.addWeighted(degenerate, -1.5, output_img, 2.5, 0)

# save the image
cv2.imwrite('output/IMG_1938_enhanced.png', output_img)

# canvas for displaying images
plt.figure(figsize=(30, 20))

input_canvas = plt.subplot(1,2,1)
input_canvas.imshow(input_img, cmap='gray')
input_canvas.set_title(r'Original picture')

output_canvas = plt.subplot(1,2,2)
output_canvas.imshow(output_img, cmap='gray')
output_canvas.set_title(r'Converted picture')

plt.show()

    
