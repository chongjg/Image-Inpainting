import cv2
import numpy as np
from PIL import Image

IMAGE_FILE = './examples/places2/images/5.png'
MASK_FILE = './examples/places2/masks/5.png'

img = np.array(Image.open(IMAGE_FILE).resize((512, 512)))
mask = np.array(np.array(Image.open(MASK_FILE).resize((512, 512)).convert('L')) > 0, dtype=np.uint8)
mask = cv2.merge([mask, mask, mask])
img[mask == 1] = 255

cv2.imwrite(IMAGE_FILE, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
cv2.imwrite(MASK_FILE, cv2.cvtColor(np.asarray(mask * 255), cv2.COLOR_RGB2BGR))
