import numpy as np
import cv2
import os


def get_channel(img, mode = 'gray', ch_num = None, to_rgb = False):
    """
    get channels for image, mainly for RGB image

    params:
        img: numpy array, the image
        mode: which mode the image will be converted
        ch_num: the position of the channel to be returned
        to_rgb: whether to convert image to RGB(for image from files)

    return:
        numpy array,
        if ch_num　is not specified and mode is not grayscale, all channels will be returned
        else a specific channel is return
    """

    im = np.copy(img)
    if to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if mode == 'gray':
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    elif mode == 'hsv':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    elif mode == 'hls':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    elif mode == 'yuv':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)

    if ch_num is None:
        return im

    return im[:,:,ch_num]



def channels_display(test_imgfs, plt, mode = 'hsv', n_show = None):
    """
    display channels of a specific image mode

    params:
        test_imgfs: array-like (str), the path for the desired images
        plt: matplotlib.pyplot object
        mode: str, the images mode
    """
    num_ch = 1 if mode is 'gray' else 3

    imgfs = test_imgfs[:n_show] if n_show else test_imgfs
    nb = len(imgfs)
    fig, axes = plt.subplots(nb, num_ch, figsize = (num_ch * 5, nb * 3))
    fig.tight_layout()
    for i, imgf in enumerate(imgfs):
        img = cv2.imread(imgf)
        chs = get_channel(img, mode = mode, to_rgb = True)
        if num_ch is 1:
            axes[i].imshow(chs, cmap = 'gray')
            axes[i].axis('off')
        else:
            for j in range(num_ch):
                axes[i][j].imshow(chs[:,:,j], cmap = 'gray')
                axes[i][j].axis('off')
    plt.show()



def color_threshold(ch, thres = (0, 255)):
    """
    for video pipeline usage, mainly for RGB　

    params:
        ch: numpy array, the channel's value
        thres: tuple of int, the thresholds


    return: numpy array of uint8, the mask
    """

    mask = np.zeros_like(ch)
    mask[(ch >= thres[0]) & (ch <= thres[1])] = 1
    return np.uint8(mask)




def gradient_threshold(ch_im, type = 'magnitude', kernelx = 3, kernely = 3, thres = (0, 255)):
    mask = np.zeros_like(ch_im)
    sobel = None

    if type == 'x':
        sobel = abs_grad(ch_im, 'x', kernelx)

    elif type == 'y':
        sobel = abs_grad(ch_im, 'y', kernely)

    elif type == 'magnitude':
        sobel = mag_grad(ch_im, kernelx, kernely)

    elif type == 'directional':
        sobel = dir_grad(ch_im, kernelx, kernely)
    assert sobel is not None,  "invalid gradient type!"
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    mask[(scaled_sobel >= thres[0]) & (scaled_sobel <= thres[1])] = 1
    return mask


def abs_grad(ch_im, orient, knl):
    x, y = int(orient is 'x'), int(orient is 'y')
    sobel = cv2.Sobel(ch_im, cv2.CV_64F, x, y, ksize = knl)
    return np.absolute(sobel)


def mag_grad(ch_im, knlx, knly):
    sobelx = abs_grad(ch_im, 'x', knlx)
    sobely = abs_grad(ch_im, 'y', knly)
    return np.sqrt(sobelx**2 + sobely**2)

def dir_grad(ch_im, knlx, knly):
    sobelx = abs_grad(ch_im, 'x', knlx)
    sobely = abs_grad(ch_im, 'y', knly)
    return np.arctan2(sobely, sobelx)
