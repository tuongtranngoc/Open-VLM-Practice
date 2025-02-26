from pdf2image import convert_from_path, convert_from_bytes


def pdf_to_images_from_path(fname):
    pil_image_lst = convert_from_path(fname, dpi=500)
    return pil_image_lst


def pdf_to_images_from_bytes(fname):
    pil_image_lst = convert_from_bytes(fname, dpi=500)
    return pil_image_lst