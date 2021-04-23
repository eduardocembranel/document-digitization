import pytesseract
from collections import namedtuple
from preprocessing import *
from posprocessing import *

class OCR_CNH:
    def __init__(self):
        self.__create_locations()
        
    def __create_locations(self):
        OCR_location = namedtuple(
            'location', ['key', 'bbox', 'segmentation_mode', 'data_type'])
        self.locations = [
            OCR_location('nome', (154, 185, 854, 42), '7', 'name'),
            OCR_location('data nascimento', (797, 325, 210, 45), '7', 'date'),
            OCR_location('cpf', (510, 329, 290, 45), '7', 'cpf'),
            OCR_location('pai', (510, 407, 460, 78), '6', 'name'),
            OCR_location('mae', (520, 485, 460, 80), '6', 'name'),
            OCR_location('registro', (160, 675, 340, 35), '7', 'number'),
            OCR_location('validade', (510, 672, 221, 40), '7', 'date'),
            OCR_location('primeira hab.', (750, 672, 245, 40), '7', 'date'),
            OCR_location('categoria', (883, 600, 110, 36), '7', 'category'),
            OCR_location('data emissao', (767, 1182, 230, 48), '7', 'date'),
            OCR_location('local', (170, 1175, 580, 49), '7', 'address'),
        ]
    
    def recognize(self, img_path):
        #colocar try catch, retornar um json vazio
        #arrumar o pos processamento
        img_target = load_image(img_path)
        img_preprocessed = preprocess(img_target)
        ocr_result_arr = self.__ocr(img_preprocessed)
        #print(ocr_result_arr)
        parsed_ocr_result = posprocess(ocr_result_arr)
        return parsed_ocr_result

    def __ocr(self, img):
        result = []
        for loc in self.locations:
            (x, y, w, h) = loc.bbox
            roi = img[y:y + h, x:x + w]
            #write_image('wtf.png', roi)
            ang, roi = correct_skew(roi)
            #write_image('wtf2.png', roi)
            #print(ang)
            tesseract_conf = '--psm {} --oem 3'.format(loc.segmentation_mode)
            text = pytesseract.image_to_string(roi, config=tesseract_conf)
            result.append((loc, text))
            #input()

        return result