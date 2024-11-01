import re
import pytesseract as pts

from conf import get_tesseract_version
from PIL import Image

# https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy
# Note that you should use '--psm' or '-psm' depending on your tesseract version
TESSERACT_CONFIG = '' if not (v := get_tesseract_version()) else '-psm 6' if (v[0] < 4 and v[1] < 5) else '--psm 6'
print('TESSERACT_CONFIG=', TESSERACT_CONFIG)

def convert_image_to_pdf(img: Image) -> bytes:
    return pts.image_to_pdf_or_hocr(
        image=img,
        extension='pdf',
        config=TESSERACT_CONFIG
    )

def extract_text_from_image(
        img: Image,
        box: tuple[float, float, float, float] = None,
        rotated: bool = False
) -> str:
    if rotated:
        im = img.crop(box=(box[1], box[0], box[3], box[2])) if box else img
        im = im.rotate(angle=90, expand=1)
    else:
        im = img.crop(box=(box[0], box[1], box[2], box[3])) if box else img

    text = pts.image_to_string(
        image=im.crop(box=(2, 0, im.width - 2, im.height)),
        config=TESSERACT_CONFIG
    )
    if text:
        # Fix OCR typos
        text = text.replace('|', 'I')
        text = re.sub(r'Il+(?![A-Za-z])', lambda x: x.group().replace('l', 'I'), text)
    return text

