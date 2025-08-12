#!/usr/bin/env python2.7

import sys
import os
import logging

sys.path.insert(0, '..')

from tightocr.adapters.api_adapter import TessApi
from tightocr.adapters.lept_adapter import pix_read
from tightocr.constants import RIL_PARA


def init_logger():
    log_obj = logging.getLogger()
    log_obj.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log_obj.addHandler(handler)
    return log_obj


def main():
    os.environ['DYLD_LIBRARY_PATH'] = (
        '/usr/local/Cellar/tesseract/3.02.02/lib:'
        '/Users/dustin/development/cpp/ctesseract/build'
    )

    logger = init_logger()
    logger.info("Starting OCR process...")

    ocr_engine = TessApi(None, 'eng')
    ocr_engine.set_variable(
        'tessedit_char_whitelist',
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'".,;?!/\\$-()&%@'
    )

    image_obj = pix_read('Out.png')
    ocr_engine.set_image_pix(image_obj)
    ocr_engine.recognize()

    avg_conf = ocr_engine.mean_text_confidence()
    logger.info("Mean confidence score: %d", avg_conf)

    if avg_conf < 60:
        raise RuntimeError("Confidence too low: %d" % avg_conf)

    for para in ocr_engine.iterate(RIL_PARA):
        print(para)


if __name__ == '__main__':
    main()
