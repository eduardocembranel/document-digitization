from preprocessing import *
from ocr import *

if __name__ == '__main__':
    img_template = load_image('../Images/template.jpg')
    img_target = load_image('../Images/target.png')

    img_aligned, h = align_images(img_target, img_template)
    write_image('../Images/aligned.png', img_aligned)

    print('estimated homography: \n', h)

    teste = ocr_cnh(img_aligned)
    print(teste)
