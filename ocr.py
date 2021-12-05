import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

def ocr(filename:str) -> str:
    '''
    Optical Character Recognition.
    Image to String.
    '''
    try:
        string = pytesseract.image_to_string(Image.open(filename))
    except:
        string = 'No text found'
    return string