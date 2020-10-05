# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import myCV
# import time

# operators = {
#     'laplacian': np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]),
#     'sobel_x': np.array([[-1., 0. , 1.], [-2., 0., 2.], [-1., 0., 1.]]),
#     'sobel_y': np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]])
# }

# img_path = 'D:/Downloads/IMG_1937.png'
# img = mpimg.imread(img_path)
# gray_img = myCV.rgb2gray(img)
# gray_img = torch.from_numpy(gray_img.reshape((1, 1) + gray_img.shape))


# def filter(img, type):
#   operator = operators[type]
#   operator = torch.from_numpy(operator)
#   operator = operator.reshape((1, 1) + operator.shape)
#   print(operator)
#   return nn.functional.conv2d(img, operator, padding=1)[0][0]
  
# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     return device

# device = get_device()
# print(device)

# canvas = plt.figure(figsize=(30, 20))

# start = time.time()
# gaussian_img = myCV.gaussian(gray_img[0][0], 5)
# end = time.time()
# print('Gaussian elapsed time: {} s'.format(end - start))
# gaussian_canvas = plt.subplot(2, 2, 1)
# gaussian_canvas.imshow(gaussian_img, cmap=plt.get_cmap('gray'))
# gaussian_canvas.set_title(r'Gaussian')

# start = time.time()
# sobel_x_img = filter(gray_img, 'sobel_x')
# end = time.time()
# print('Sobel x elapsed time using GPU: {} s'.format(end - start))

# start = time.time()
# sobel_y_img = filter(gray_img, 'sobel_y')
# end = time.time()
# print('Sobel y elapsed time using GPU: {} s'.format(end - start))

# sobel_x_canvas = plt.subplot(2,2, 3)
# sobel_y_canvas = plt.subplot(2,2,4)
# sobel_x_canvas.imshow(sobel_x_img, cmap=plt.get_cmap('gray'))
# sobel_y_canvas.imshow(sobel_y_img, cmap=plt.get_cmap('gray'))
# sobel_x_canvas.set_title(r'Sobel x')
# sobel_y_canvas.set_title(r'Sobel y')

# start = time.time()
# laplacian_img = filter(gray_img, 'laplacian')
# end = time.time()
# print('Laplacian elapsed time using GPU: {} s'.format(end - start))

# laplacian_canvas = plt.subplot(2,2,2)
# laplacian_canvas.imshow(laplacian_img, cmap=plt.get_cmap('gray'))
# laplacian_canvas.set_title(r'Laplacian')

# plt.show()

# # canvas = plt.figure(figsize=(30, 20))
# # start = time.time()
# # sobel_y_img = filter(gray_img, 'sobel_y')
# # end = time.time()
# # print('Laplacian elapsed time using GPU: {} ms'.format(end - start))
# # sobel_y_canvas = plt.subplot(1,1,1)
# # sobel_y_canvas.imshow(sobel_y_img[0][0], cmap=plt.get_cmap('gray'))
# # sobel_y_canvas.set_title(r'Sobel y')
# # plt.show()

# # filters = torch.randn(1,1,3,3)
# # print(filters)
# # inputs = torch.randn(1,1,5,5)
# # print(nn.functional.conv2d(inputs, filters, padding=1))

