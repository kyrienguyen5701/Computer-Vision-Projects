import numpy as np
import cv2

def rgb2gray(img, conversion_mode='gimp'):

  modes = { 
    'luminance': (.2126, .7152, .0722), # luminance perception
    'gimp': (.299, .587, .114)   # linear approximation, used in photoshop
  }
       
  try:
    if conversion_mode not in modes.keys():
        raise Exception('This conversion mode is currently available')
 
  except Exception as e:
    print(e)
    return None

  return np.dot(img[..., :3], modes[conversion_mode])

def gray2bw_threshold(img, threshold = .5):
  try:
    if threshold > 1 or threshold < 0:
      raise Exception('This threshold is off the range.')
  
  except Exception as e:
    print(e)
    return None
  
  for i in range(len(img)):
    for j in range(len(img[0])):
      if img[i, j] >= threshold:
        img[i, j] = 1
      else:
        img[i, j] = 0
  return img

def flip(img, direction):
  (height, width) = img.shape[:2]
  directions = ('horizontal', 'vertical', 'h', 'v')
  try:
    if direction not in directions:
      raise Exception('Invalid direction')
  
  except Exception as e:
    print(e)
  
  if direction == 'horizontal' or direction == 'h':
    for i in range(height):
      for j in range(width // 2):
        temp = img[i, j].copy()
        img[i, j] = img[i, width - 1 - j]
        img[i, width - 1 - j] = temp

  if direction == 'vertical' or direction == 'v':
    for i in range(height // 2):
      for j in range(width):
        temp = img[i, j].copy()
        img[i, j] = img[height - 1 - i, j]
        img[height - 1 - i, j] = temp

def crop(img, left,top, width, height):
  return img[top:min([top + height, len(img)]), left:min([left + width, len(img[0])])]

# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
# TODO: rewrite this function without using warpAffine

def rotate(img, deg):
  (height, width) = img.shape[:2]
  (centerX, centerY) = (width // 2, height // 2)
  
  rad = deg * PI / 180
  cosine = np.cos(rad)
  sine = np.sin(rad)

  # 'new' bounding dimensions of the image
  (new_height, new_width) = (int(height * abs(cosine) + width * abs(sine)), int(height * abs(sine) + width * abs(cosine)))

  # rotation matrix
  rotation_matrix = np.array([[cosine, -sine,  (1 - cosine) * centerX + sine * centerY], [sine, cosine, -sine * centerX + (1 - cosine) * centerY]]) #~ rotation_matrix = cv2.getRotationMatrix2D((centerX, centerY), -deg, 1.0)
  
  # take into account of translation
  rotation_matrix[0, 2] += new_width // 2 - centerX
  rotation_matrix[1, 2] += new_height // 2 - centerY

  return cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

# https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf
# http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/
def convolution(img, kernel, average = False):
  
  kernel_row, kernel_column = kernel.shape
  try:
    if (len(img.shape) == 3):
      raise Exception('Make sure to convert your picture to grayscale first!')
    if kernel_row % 2 == 0:
      raise Exception('Invalid kernel size.')
  
  except Exception as e:
    print(e)
    return None
  
  height, width = img.shape

  result = np.zeros(img.shape)
 
  # padding the image for same convolution  
  pad_height = int((kernel_row - 1) / 2)
  pad_width = int((kernel_column - 1) / 2)
  padded_img = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
  padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img

  #convolution formula
  for r in range(height):
    for c in range(width):
      result[r, c] = np.sum(kernel * padded_img[r:r + kernel_row, c:c + kernel_column])
      if average:
        result[r, c] /= kernel_row * kernel_column

  return result

def dnorm(x, mu, sd):
    return 1 / ((2 * np.pi) ** .5 * sd) * np.e ** (((x - mu) / sd) ** 2 / 2)

def gaussian_kernel(size, sigma):
  row = np.linspace(-(size // 2), size // 2, size)
  for i in range(size):
      row[i] = dnorm(row[i], 0, sigma)
  kernel = np.outer(row.T, row.T)
  kernel *= 1.0 / kernel.max()
  return kernel

def gaussian(img, ksize, average=True):
  kernel = gaussian_kernel(ksize, sigma=np.sqrt(ksize))
  return convolution(img, kernel, average=average)

def sobel(img, direction = 'h', average=True):
  operator = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])
  directions = ('horizontal', 'vertical', 'h', 'v')
  try:
    if direction not in directions:
      raise Exception('Invalid direction')
  
  except Exception as e:
    print(e)

  if direction == 'v' or direction == 'vertical':
    operator = np.rot90(operator)
  
  return convolution(img, operator, average=average)

def laplacian(img, average=True):
  operator = np.array([[1,1,1], [1, -8, 1], [1, 1, 1]])
  return convolution(img, operator, average=average)