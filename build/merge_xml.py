#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import os
import sys
import logging
import traceback
from xml.etree.ElementTree import parse

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def main():
    if len(sys.argv) != 3:
        logging.error("arguments error")
        sys.exit(1)
    build_dir = os.path.dirname(os.path.abspath(__file__))
    input_xml = sys.argv[1]
    output_xml = sys.argv[2]
    try:
        doc = parse(input_xml)
        doc_insert = parse(os.path.join(
            build_dir, 'toolkit_xml', 'ascend-ide-backend.xml'))
        config = doc.getroot()
        file_info_to_insert = doc_insert.getroot()
        config.append(file_info_to_insert)
        doc.write(output_xml, xml_declaration=True, encoding='UTF-8')
    except Exception:
        logging.error(traceback.format_stack())
        sys.exit(1)


if __name__ == '__main__':
    main()
