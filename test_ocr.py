from pathlib import Path

from PIL import Image

from ocr_utils import extract_text_from_image

path = '/Users/oleg/Projects/PPC/home/ragagent/documents/temp/5acab0fc-8fd6-4c1b-9ad3-cdd6cb4e259e-16.png'
img = Image.open(path)

# Vertical position of the top of the page content
text_v_top = 0
text_v_bottom = img.height * 0.938  # cutoff footers

text = extract_text_from_image(img, box=(0, text_v_top, img.width, text_v_bottom))

print(text)

import PyPDF2

pdf = PyPDF2.PdfReader(Path(path).with_suffix(".pdf"))
p: PyPDF2.PageObject = pdf.pages[0]
print(p.extract_text())

