import cv2
import numpy as np
from preprocessing import *

if __name__ == '__main__':
    img_template = load_image('../Images/template.png')
    img_target = load_image('../Images/target.png')

    img_aligned, h = deskewing(img_target, img_template)
    write_image('../Images/aligned.png', img_aligned)

    print('estimated homography: \n', h)