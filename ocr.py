from PIL import Image
import pytesseract

def run_ocr(image_path: str) -> str:
    return pytesseract.image_to_string(
        Image.open(image_path),
        lang="amh+eng"
    )


# import pytesseract
# from PIL import Image

# def run_ocr(image_path):
#     return pytesseract.image_to_string(
#         Image.open(image_path),
#         lang="amh+eng"
#     )
