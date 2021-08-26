import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_bounding_box(input_image):
    threshold = 127
    if isinstance(input_image, str):
        img = plt.imread(input_image)
    else:
        img = input_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # note that h goes with x and w with y
    x, y, w, h = cv2.boundingRect(contours[1])
    return (x, y, w, h), thresh

def get_mask(base_folder, save_folder, input_image):
    img = plt.imread(base_folder + input_image)
    #img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (50, 50))
    img = np.pad(img, ((10, 10), (10, 10), (0, 0)), 'constant', constant_values=(255, 255))
    (x, y, w, h), thresh = get_bounding_box(img)
    # img = img[x:x + h, y:y + w, :]
    img = img[:, :, ::-1]
    alpha_channel = 255 * np.zeros(img[:, :, 0].shape, dtype=img.dtype)
    alpha_channel[np.nonzero(thresh == 0)] = 255
    img = cv2.merge((img, alpha_channel))
    print(img.shape)
    cv2.imwrite(save_folder + input_image.split('.')[0]+'_mask.png', img)

base_folder= '/home/carmelo/Projects/git/NFTGenerator/'
save_folder = base_folder + 'Masks/'
input_image = 'cowboy_hat.jpg'
get_mask(base_folder, save_folder, input_image)