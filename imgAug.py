import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
# ia.seed()

image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
images = [image, image, image, image]
rotate = iaa.Affine(rotate=(-25, 25))

images_aug = rotate(images=images)
ia.imshow(np.hstack(images_aug))