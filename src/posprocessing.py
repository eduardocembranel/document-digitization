import re

def posprocess(ocr_result):
    res = {}
    for description, text in ocr_result:
        text = parse_text(text, description.data_type)
        res[description.key] = text
    
    return res

def parse_text(text, data_type):
    text = clean_text(text, data_type)
    
    if data_type == 'name':
        return parse_name(text)
    elif data_type == 'number':
        return parse_number(text)
    elif data_type == 'category':
        return text
    elif data_type == 'date':
        return parse_date(text)
    elif data_type == 'cpf':
        return parse_cpf(text)
    elif data_type == 'address':
        return parse_address(text)

def clean_text(text, data_type):
    def clean_condition(c, data_type): #return true if 'c' should be removed
        if data_type == 'name':
            return not (c.isupper() or c.isspace())
        elif data_type == 'number':
            return not c.isdigit()
        elif data_type == 'category':
            return not (c >= 'A' and c <= 'E')
        elif data_type == 'date':
            return not (c.isdigit() or c == '/')
        elif data_type == 'cpf':
            return not (c.isdigit() or c == '.' or c == '-')
        elif data_type == 'address':
            return not (c.isupper() or c.isspace() or c == ',')

    text = text.replace('\n', ' ')
    text = ''.join(['' if clean_condition(c, data_type) 
        else c for c in text]).strip()
    return remove_multiple_spaces(text)

def remove_multiple_spaces(text):
    return re.sub(' +', ' ', text)

def parse_name(text):
    arr = text.split(' ')
    if len(arr) < 2 or len(text) < 6:
        return ''
    if len(arr[-1]) == 1:
        text = text.rsplit(' ', 1)[0]

    return text.lower().title()

def parse_number(text):
    return text if len(text) > 9 else ''

def parse_date(text):
    return re.search('(\d\d/\d\d/\d\d\d\d)|$', text).group()

def parse_cpf(text):
    return re.search('(\d\d\d.\d\d\d.\d\d\d-\d\d)|$', text).group()

def parse_address(text):
    arr = text.split(',')
    if len(arr) != 2: 
        return ''

    arr[0] = arr[0].strip()
    arr[1] = arr[1].strip()

    if len(arr[0]) < 3 or len(arr[1]) != 2:
        return ''

    return arr[0].title() + ' - ' + arr[1]