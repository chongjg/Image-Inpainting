from skimage.io import imread, imsave

from inpainter import Inpainter

def inpaint_criminisi():
    image = imread('tmp/input.png')
    mask = imread('tmp/mask.png', as_gray=True)

    output_image = Inpainter(
        image,
        mask,
        patch_size=9,
        plot_progress=False
    ).inpaint()
    imsave('tmp/output.png', output_image)
