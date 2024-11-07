
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
RemoveInplaceLayerProcess class.
This class mainly involves the remove_inplace_layer function.
"""

import sys
import stat
import argparse
import os

import google.protobuf.text_format
import caffe.proto.caffe_pb2 as caffe_pb2

from cmp_utils.constant.compare_error import CompareError
from cmp_utils import log, file_utils
from cmp_utils import path_check
from cmp_utils.utils import safe_path_string

MAX_SIZE = 10 * 1024 * 1024 * 1024


class RemoveInplaceLayerProcess:
    """
    The class for remove inplace layer for caffe prototxt
    """

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    def __init__(self: any) -> None:
        parse = argparse.ArgumentParser()
        parse.add_argument("-i", dest="input_file_path", help="<Required> the prototxt file path",
                           type=safe_path_string, required=True)
        parse.add_argument("-o", dest="output_file_path", help="<Optional> the output file path", type=safe_path_string)
        args, _ = parse.parse_known_args(sys.argv[1:])
        self.input_file_path = os.path.realpath(args.input_file_path)
        if args.output_file_path:
            if os.path.islink(os.path.abspath(args.output_file_path)):
                log.print_error_log('The path "%r" is a softlink, not permitted.' % args.output_file_path)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
            self.output_file_path = os.path.realpath(args.output_file_path)
        else:
            output_file_path = os.path.join(
                os.path.dirname(self.input_file_path),
                "new_" + os.path.basename(self.input_file_path))
            self.output_file_path = os.path.realpath(output_file_path)
        self.net_param = None
        self.cur_layer_idx = -1

    @staticmethod
    def _check_input_file_valid(path: str) -> None:
        ret = path_check.check_path_valid(
            path, exist=True, have_write_permission=False, path_type=path_check.PathType.File
        )
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

        if os.path.getsize(path) > MAX_SIZE:
            log.print_error_log("The file '%r' is too large." % path)
            raise CompareError(CompareError.MSACCUCMP_FILE_TOO_LARGE_ERROR)

    @staticmethod
    def _check_output_file_valid(path: str) -> None:
        ret = path_check.check_path_valid(
            path, exist=False, have_write_permission=True, path_type=path_check.PathType.File
        )
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

        if os.path.exists(path):
            if not os.path.isfile(path):
                log.print_error_log("Provided output_file_path exists but is not a file.")
                raise CompareError(CompareError.MSACCUCMP_SYMLINK_ERROR)
            os.remove(path)
            log.print_warn_log("The file '%r' already exists" % path)

    def check_arguments_valid(self: any) -> None:
        """
        Check file valid.
        """
        self._check_input_file_valid(self.input_file_path)
        self._check_output_file_valid(self.output_file_path)

    def remove_inplace_layer(self: any) -> None:
        """
        remove inplace layer and save new layer to file
        """
        # check path valid
        self.check_arguments_valid()

        # read prototxt file
        self.net_param = caffe_pb2.NetParameter()
        try:
            with open(self.input_file_path, 'rb') as model_file:
                google.protobuf.text_format.Parse(model_file.read(), self.net_param)
        except (google.protobuf.text_format.ParseError, UnicodeDecodeError) as error:
            log.print_error_log("Provided input_file_path is not a valid protobuf text file")
            raise CompareError(CompareError.MSACCUCMP_INVALID_FORMAT_ERROR) from error

        # parse net
        while True:
            old_name, new_name = self._find_name()
            if not old_name and not new_name:
                break
            self._parse_name(old_name, new_name)

        # remove Dropout type
        for (_, layer_item) in enumerate(self.net_param.layer):
            if layer_item.type == 'Dropout':
                self.net_param.layer.remove(layer_item)

        # write file to new path
        with os.fdopen(os.open(self.output_file_path, self.WRITE_FLAGS, self.WRITE_MODES), 'w') as open_file:
            file_content = str(self.net_param)
            open_file.write(file_content)
        log.print_info_log('The "%r" has removed inplace layer.' % self.input_file_path)
        log.print_info_log('The new prototxt file has been saved to "%r".' % self.output_file_path)

    def _handle_top(self: any, layer_item: any, layer_idx: int) -> (bool, str, str):
        for (top_index, top_item) in enumerate(layer_item.top):
            if layer_item.type == 'Dropout':
                if len(layer_item.bottom) != 1:
                    return True, '', ''
                bottom_item = layer_item.bottom[0]
                if top_item != bottom_item:
                    layer_item.top[top_index] = bottom_item
                    old_name = top_item
                    new_name = bottom_item
                    self.cur_layer_idx = layer_idx
                    return True, old_name, new_name
            elif top_item != layer_item.name:
                if len(layer_item.top) == 1 and len(layer_item.bottom) == 1 \
                        and layer_item.top[0] == layer_item.bottom[0]:
                    layer_item.top[top_index] = layer_item.name
                    old_name = top_item
                    new_name = layer_item.name
                    self.cur_layer_idx = layer_idx
                    return True, old_name, new_name
        return False, '', ''

    def _find_name(self: any) -> (str, str):
        for (layer_idx, layer_item) in enumerate(self.net_param.layer):
            if layer_idx <= self.cur_layer_idx:
                continue
            ok, old_name, new_name = self._handle_top(layer_item, layer_idx)
            if ok:
                return old_name, new_name
        return '', ''

    def _parse_name(self: any, old_name: str, new_name: str) -> None:
        for (layer_idx, layer_item) in enumerate(self.net_param.layer):
            if layer_idx <= self.cur_layer_idx:
                continue
            for (bottom_index, bottom_item) in enumerate(layer_item.bottom):
                if bottom_item == old_name:
                    layer_item.bottom[bottom_index] = new_name
            for (top_index, top_item) in enumerate(layer_item.top):
                if top_item == old_name:
                    layer_item.top[top_index] = new_name


if __name__ == "__main__":
    with file_utils.UmaskWrapper():
        try:
            RemoveInplaceLayerProcess().remove_inplace_layer()
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)
