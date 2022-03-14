import cv2
import numpy as np
from PIL import Image

from libs.pconv_model import PConvUnet

def inpaint_pconv(h, w):
    IMAGE_FILE = './tmp/input.png'
    MASK_FILE = './tmp/mask.png'
    OUTPUT_FILE = './tmp/output.png'

    model = PConvUnet(vgg_weights=None, inference_only=True)
    model.load('checkpoints/pconv_imagenet.26-1.07.h5', train_bn=False)

    img = np.array(Image.open(IMAGE_FILE).resize((512, 512)))
    mask = np.array(np.array(Image.open(MASK_FILE).resize((512, 512)).convert('L')) > 0, dtype=np.uint8)
    mask = cv2.merge([mask, mask, mask])
    img[mask == 1] = 255

    dst = np.squeeze(model.predict([np.expand_dims(img / 255, 0), np.expand_dims(1 - mask, 0)]), 0)
    dst = np.array(dst * 255 * mask, dtype=np.uint8) + img * (1 - mask)
    dst = cv2.resize(dst, (h, w))

    cv2.imwrite(OUTPUT_FILE, cv2.cvtColor(np.asarray(dst), cv2.COLOR_RGB2BGR))