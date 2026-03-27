# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
from msprobe.core.common.file_utils import create_directory
from msprobe.core.parse.factory import ParserFactory



_parser_factory = ParserFactory()


def _parse_parser(parser):
    parser.add_argument("-d", "--dump_path", dest="dump_path", type=str,
                        help="<Required> The file path or directory path to parse", required=True)
    parser.add_argument("-t", "--type", dest="parse_type", type=str,
                        choices=['npy', 'pt'],
                        help="The file type after parse. Supported types: npy, pt (default: pt)", required=False, default='pt')
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="The output file path after parse",
                        required=False, default="./output", nargs="?", const="./output")


def parse_cli(args):
    dump_path = args.dump_path
    parse_type = args.parse_type
    output_path = args.output_path
    
    create_directory(output_path)
    parser = _parser_factory.get_parser(dump_path)
    parser.parse(dump_path, output_path, parse_type)

