from ocr import OCR_CNH
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

#localhost:5000/api/ocr_cnh?url=../Images/target.png
@app.route('/api/ocr_cnh', methods=['GET'])
def api_ocr():
    img_path = request.args.get('url')
    scanner = OCR_CNH()
    res = scanner.recognize(img_path) 
    return res

if __name__ == '__main__':
    app.run(debug=True, port=5000)