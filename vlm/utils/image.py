from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import io


def pdf_to_images_from_path(fname):
    pil_image_lst = convert_from_path(fname, dpi=200)
    return pil_image_lst


def pdf_to_images_from_bytes(fname):
    pil_image_lst = convert_from_bytes(fname, dpi=200)
    return pil_image_lst


def read_byte_io(file):
    image = Image.open(io.BytesIO(file.read()))
    return image