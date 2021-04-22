import pytesseract
from collections import namedtuple
from preprocessing import *

def ocr_cnh(img):
    #call preprocessing
    img = gray_scale(img)
    img = clahe(img, 1.0, 7)
    img = sharpen(img)
    img = noise_removal(img)

    #write_image('wtf.png', img)
    #input()

    Ocr_location = namedtuple('location', ['key', 'bbox', 'segmentation_mode'])
    locations = [
        Ocr_location('nome', (154, 185, 854, 42), '7'),
        Ocr_location('data nascimento', (797, 325, 210, 45), '7'),
        Ocr_location('cpf', (510, 329, 290, 45), '7'),
        Ocr_location('pai', (510, 410, 460, 75), '6'),
        Ocr_location('mae', (520, 485, 460, 80), '6'),
        Ocr_location('registro', (160, 675, 340, 35), '7'),
        Ocr_location('validade', (510, 672, 221, 40), '7'),
        Ocr_location('primeira habilitacao', (750, 672, 245, 40), '7'),
        Ocr_location('categoria', (883, 600, 110, 36), '7'),
        Ocr_location('data emissao', (767, 1185, 230, 45), '7'),
        Ocr_location('local', (170, 1175, 580, 49), '7'),
    ]

    parsing_txt = []
    for loc in locations:
        (x, y, w, h) = loc.bbox
        roi = img[y:y + h, x:x + w]
        #write_image('wtf.png', roi) #remover isso dps
        tesseract_conf = '--psm {} --oem 3'.format(loc.segmentation_mode)
        text = pytesseract.image_to_string(roi, config=tesseract_conf)
        parsing_txt.append(text)
        #input()

    return parsing_txt