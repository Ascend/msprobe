#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
TensorConversion class. This class mainly involves the convert_shape function.
"""
from src.compare.dump_parse.dump import CompareData
from format_manager import FormatManager
from format_manager import SrcToDest
from format_manager import ShapeConversion
from fusion_op import FusionOp
from fusion_op import Tensor

from cmp_utils import log
import utils
import dump_data_pb2 as DD

from cmp_utils import common

from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


class TensorConversion:
    """
    The class for tensor conversion
    """

    def __init__(self: any, fusion_op: FusionOp, format_manager: FormatManager, is_detail: bool) -> None:
        self.fusion_op = fusion_op
        self.shape_conversion = ShapeConversion(format_manager)
        self.is_detail = is_detail

    @staticmethod
    def _translate_shape_to_array(shape_tuple: list) -> (list, bool):
        all_shape_is_1 = False
        remove_1_shape = []
        for item in shape_tuple:
            if item != 1:
                remove_1_shape.append(item)
        if len(remove_1_shape) == 0:
            return shape_tuple, True
        return remove_1_shape, all_shape_is_1

    @staticmethod
    def _padding_shape(origin_shape: list, expect_length: int) -> (list, int, int):
        shape = []
        size = 1
        one_count = 0
        for dim in origin_shape:
            shape.append(dim)
            size *= dim
            if dim == 1:
                one_count += 1
        if expect_length > len(shape):
            left_count = expect_length - len(shape)
            for _ in range(left_count):
                shape.append(1)
        return shape, size, one_count

    @staticmethod
    def _change_format(ground_truth_tensor: Tensor, origin_shape: any) -> None:
        # if fusion rule is (1,224,224, 3),origin is (1,3, 224,224),
        is_original_shape_nhwc = ground_truth_tensor.shape[0] == origin_shape[0] and \
                                 ground_truth_tensor.shape[1] == origin_shape[2] and \
                                 ground_truth_tensor.shape[2] == origin_shape[3] and \
                                 ground_truth_tensor.shape[3] == origin_shape[1]
        # need change format to NCHW
        if ground_truth_tensor.tensor_format == 'NHWC' and is_original_shape_nhwc:
            ground_truth_tensor.tensor_format = 'NCHW'
        # if fusion rule is (1,3,224,224),origin is (1,224,224,3),
        is_original_shape_nchw = ground_truth_tensor.shape[0] == origin_shape[0] and \
                                 ground_truth_tensor.shape[1] == origin_shape[3] and \
                                 ground_truth_tensor.shape[2] == origin_shape[1] and \
                                 ground_truth_tensor.shape[3] == origin_shape[2]
        # need change format to NHWC
        if ground_truth_tensor.tensor_format == 'NCHW' and is_original_shape_nchw:
            ground_truth_tensor.tensor_format = 'NHWC'

    @staticmethod
    def _make_detail_dest_format(my_output_tensor: any, ground_truth_format: int) -> (any, any):
        # convert my output and ground truth format to nchw
        if common.contain_depth_dimension(my_output_tensor.tensor_format):
            my_output_dest_format = DD.FORMAT_NCDHW
            ground_truth_dest_format = DD.FORMAT_NCDHW
        else:
            if my_output_tensor.tensor_format == DD.FORMAT_FRACTAL_Z:
                my_output_dest_format = ground_truth_format
                ground_truth_dest_format = ground_truth_format
            else:
                my_output_dest_format = DD.FORMAT_NCHW
                ground_truth_dest_format = DD.FORMAT_NCHW
        return my_output_dest_format, ground_truth_dest_format

    def slice_data(self: any, my_output_np: any, adjusted_shape: list) -> any:
        """
        Slice data by right shape
        :param my_output_np: my output numpy data
        :param adjusted_shape: the shape to slice
        :return the sliced numpy data
        """
        old_my_output_shape_str = utils.convert_shape_to_string(my_output_np.shape)
        if len(my_output_np.shape) < len(adjusted_shape):
            self._check_ground_truth_shape_by_my_output_shape(my_output_np.shape, adjusted_shape)
            return my_output_np
        # padding to 4d
        my_output_shape, my_output_size, one_count = self._padding_shape(
            my_output_np.shape, ConstManager.FOUR_DIMS_LENGTH)

        # padding to my output shape
        slice_shape, ground_truth_size, _ = self._padding_shape(adjusted_shape, len(my_output_shape))

        if ground_truth_size == my_output_size:
            if one_count == len(my_output_np.shape) or one_count == len(my_output_np.shape) - 1:
                return my_output_np

        # my output shape large then ground truth dump data, need slice data
        if self._should_slice(my_output_shape, slice_shape):
            # slice data
            my_output_np = my_output_np.reshape(my_output_shape)
            if len(my_output_shape) == ConstManager.FOUR_DIMS_LENGTH:
                my_output_np = my_output_np[:slice_shape[0], :slice_shape[1], :slice_shape[2], :slice_shape[3]]
            elif len(my_output_shape) == ConstManager.FIVE_DIMS_LENGTH:
                my_output_np = \
                    my_output_np[:slice_shape[0], :slice_shape[1], :slice_shape[2], :slice_shape[3], :slice_shape[4]]
            if self.fusion_op:
                log.print_info_log('[%s] The left dump data has been sliced from %s to %s.' %
                                   (self.fusion_op.op_name, old_my_output_shape_str,
                                    utils.convert_shape_to_string(slice_shape)))
        return my_output_np

    def get_my_output_and_ground_truth_data(self: any, compare_data: CompareData, my_output_tensor: any,
                                            ground_truth_tensor: Tensor) -> (any, any, any):
        """
        Deserialize the my output and ground truth tensor to  array
        :param compare_data: the compare data
        :param my_output_tensor: my output tensor
        :param ground_truth_tensor: ground truth tensor
        :return: my_output_array, ground_truth_array, shape
        """
        # get ground truth format
        origin_format = my_output_tensor.tensor_format
        if ground_truth_tensor.tensor_format != "":
            origin_format = self._get_ground_truth_format(ground_truth_tensor)
        is_tensor = (utils.get_shape_type(my_output_tensor.shape) == utils.ShapeType.Tensor)
        my_output_shape = utils.convert_shape_to_string(my_output_tensor.shape)

        # when compare quant and origin
        if compare_data.is_standard_quant_vs_origin():
            # deserialize dump data to np array
            my_output_array = my_output_tensor.data
            ground_truth_array = ground_truth_tensor.data.data
            log.print_info_log('[%s] Left %s <======> Right %s'
                               % (self.fusion_op.op_name, my_output_shape,
                                  utils.convert_shape_to_string(ground_truth_tensor.data.shape)))
            self._check_shape_valid(my_output_tensor.shape, ground_truth_tensor.data.shape)
            return my_output_array, ground_truth_array, my_output_tensor.shape

        log.print_info_log('[%s] Before shape convert, Left %s%s <======> Right %s%s'
                           % (self.fusion_op.op_name, common.get_format_string(my_output_tensor.tensor_format),
                              my_output_shape, ground_truth_tensor.tensor_format,
                              utils.convert_shape_to_string(ground_truth_tensor.data.shape)))
        # convert shape
        my_output_np, ground_truth_np = self._convert_shape(my_output_tensor, ground_truth_tensor, origin_format)

        # slice data
        if (my_output_tensor.tensor_format != DD.FORMAT_ND or
           (is_tensor and (my_output_np.size != ground_truth_np.size))):
            my_output_np = self.slice_data(my_output_np, ground_truth_np.shape)

        my_output_array = my_output_np.flatten()
        ground_truth_array = ground_truth_np.flatten()
        if len(my_output_array) != len(ground_truth_array):
            message = log.print_cannot_compare_warning(
                self.fusion_op.op_name, '(%d)' % len(my_output_array), '(%d)' % len(ground_truth_array))
            raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR, message)
        return my_output_array, ground_truth_array, my_output_np.shape

    def _check_ground_truth_shape_by_my_output_shape(self: any, my_output_shape: any,
                                                     ground_truth_shape: any) -> None:
        my_output_shape_array, my_output_all_shape_is_1 = self._translate_shape_to_array(my_output_shape)
        ground_truth_shape_array, ground_truth_all_shape_is_1 = self._translate_shape_to_array(ground_truth_shape)
        if my_output_all_shape_is_1 and ground_truth_all_shape_is_1:
            return
        self._check_shape_valid(my_output_shape_array, ground_truth_shape_array)

    def _should_slice(self: any, my_output_shape: list, slice_shape: list) -> bool:
        is_slice = False
        for my_output_dim, ground_truth_dim in zip(my_output_shape, slice_shape):
            if my_output_dim < ground_truth_dim:
                if self.fusion_op:
                    old_my_output_shape_str = utils.convert_shape_to_string(my_output_shape)
                    slice_shape_str = utils.convert_shape_to_string(slice_shape)
                    message = log.print_cannot_compare_warning(self.fusion_op.op_name, old_my_output_shape_str,
                                                               slice_shape_str)
                    raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR, message)
                raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)
            if my_output_dim > ground_truth_dim:
                is_slice = True
                break
        return is_slice

    def _change_format_by_origin_shape(self: any, ground_truth_tensor: Tensor) -> None:
        origin_shape = ground_truth_tensor.data.shape
        if len(ground_truth_tensor.shape) == ConstManager.FOUR_DIMS_LENGTH and \
                len(origin_shape) == ConstManager.FOUR_DIMS_LENGTH:
            match = True
            for (ground_truth_dim, origin_dim) in zip(ground_truth_tensor.shape, origin_shape):
                if ground_truth_dim != origin_dim:
                    match = False
                    break
            if not match:
                self._change_format(ground_truth_tensor, origin_shape)

    def _get_ground_truth_format(self: any, ground_truth_tensor: Tensor) -> int:
        if ground_truth_tensor.tensor_format not in ConstManager.STRING_TO_FORMAT_MAP:
            message = 'The format " %s " is not supported.' % ground_truth_tensor.tensor_format
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_FORMAT_ERROR, message)
        if ground_truth_tensor.tensor_format in ('NHWC', 'NCHW'):
            self._change_format_by_origin_shape(ground_truth_tensor)
        return ConstManager.STRING_TO_FORMAT_MAP.get(ground_truth_tensor.tensor_format)

    def _check_shape_valid(self: any, my_output_shape: any, ground_truth_shape: any) -> None:
        if len(my_output_shape) != len(ground_truth_shape):
            op_name = self.fusion_op.op_name if self.fusion_op else ''
            message = log.print_cannot_compare_warning(
                op_name, utils.convert_shape_to_string(my_output_shape),
                utils.convert_shape_to_string(ground_truth_shape))
            raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR, message)
        for (my_output_dim, ground_truth_dim) in zip(my_output_shape, ground_truth_shape):
            if my_output_dim != ground_truth_dim:
                if self.fusion_op:
                    message = log.print_cannot_compare_warning(self.fusion_op.op_name,
                                                               utils.convert_shape_to_string(my_output_shape),
                                                               utils.convert_shape_to_string(ground_truth_shape))
                    raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR, message)
                raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def _convert_shape(self: any, my_output_tensor: any, ground_truth_tensor: Tensor,
                       ground_truth_format: int) -> (any, any):
        my_output_dest_format = my_output_tensor.tensor_format
        ground_truth_dest_format = ground_truth_format
        if self.is_detail:
            # convert my output and ground truth format to nchw
            my_output_dest_format, ground_truth_dest_format = self._make_detail_dest_format(
                my_output_tensor, ground_truth_format)
        else:
            if my_output_tensor.tensor_format != ground_truth_tensor.tensor_format:
                my_output_dest_format = ground_truth_format

        # deserialize dump data to np array
        my_output_array = my_output_tensor.data
        ground_truth_array = ground_truth_tensor.data.data

        # ND format no need to convert, except FORMAT_FRACTAL_NZ to ND
        if (my_output_dest_format == DD.FORMAT_ND and my_output_tensor.tensor_format != DD.FORMAT_FRACTAL_NZ) \
                or utils.get_shape_type(my_output_tensor.shape) != utils.ShapeType.Tensor:
            return my_output_array, ground_truth_array

        # shape convert for my output
        my_output_group = common.get_sub_format(my_output_tensor)
        my_output_src_to_dest = SrcToDest(my_output_tensor.tensor_format, my_output_dest_format, my_output_tensor.shape,
                                          ground_truth_tensor.data.shape)
        my_output_np = self.shape_conversion.convert_shape(my_output_src_to_dest, my_output_array,
                                                           {'group': my_output_group})

        # shape convert for ground truth
        ground_truth_group = common.get_sub_format(ground_truth_tensor, my_output_group)
        ground_truth_src_to_dest = SrcToDest(ground_truth_format, ground_truth_dest_format,
                                             ground_truth_tensor.data.shape,
                                             ground_truth_tensor.data.shape)
        ground_truth_np = self.shape_conversion.convert_shape(ground_truth_src_to_dest, ground_truth_array,
                                                              {'group': ground_truth_group})

        log.print_info_log('[%s] After shape convert, Left %s%s <======> Right %s%s'
                           % (self.fusion_op.op_name, common.get_format_string(my_output_dest_format),
                              str(my_output_np.shape), common.get_format_string(ground_truth_dest_format),
                              str(ground_truth_np.shape)))
        return my_output_np, ground_truth_np
