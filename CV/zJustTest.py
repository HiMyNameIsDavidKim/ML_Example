from skimage.io import imread
from torchvision import transforms as T


img = imread('./data/messi.jpeg')
img = T.ToTensor()(img)

print(img.shape)
