import unittest
import pytest
from unittest import mock

from vector_cmp.fusion_manager import fusion_op
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.range_manager.range_manager import RangeManager
from vector_cmp.range_manager.range_mode import RangeMode
from vector_cmp.range_manager.select_mode import SelectMode


class TestUtilsMethods(unittest.TestCase):

    def test_parse_range1(self):
        with pytest.raises(CompareError) as error:
            RangeMode(",")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range2(self):
        with pytest.raises(CompareError) as error:
            RangeMode(",-1,xx")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range3(self):
        with pytest.raises(CompareError) as error:
            RangeMode("xx,,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range4(self):
        with pytest.raises(CompareError) as error:
            RangeMode(",xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range5(self):
        range_manager = RangeMode(",,")
        self.assertEqual(range_manager.step, 1)
        self.assertEqual(range_manager.end, -1)
        self.assertEqual(range_manager.start, 1)

    def test_parse_range6(self):
        range_manager = RangeMode("5,10,2")
        self.assertEqual(range_manager.start, 5)
        self.assertEqual(range_manager.step, 2)
        self.assertEqual(range_manager.end, 10)

    def test_parse_selected_op1(self):
        with pytest.raises(CompareError) as error:
            SelectMode("xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op2(self):
        with pytest.raises(CompareError) as error:
            SelectMode(",-1,xx")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op3(self):
        with pytest.raises(CompareError) as error:
            SelectMode("xx,,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op4(self):
        with pytest.raises(CompareError) as error:
            SelectMode(",xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op5(self):
        range_manager = SelectMode("1,2,3")
        self.assertEqual(range_manager.selected_op[0], 1)
        self.assertEqual(range_manager.selected_op[1], 2)
        self.assertEqual(range_manager.selected_op[2], 3)

    def test_parse_selected_op6(self):
        range_manager = SelectMode("10,5,7")
        self.assertEqual(range_manager.selected_op[0], 5)
        self.assertEqual(range_manager.selected_op[1], 7)
        self.assertEqual(range_manager.selected_op[2], 10)


    def test_check_range_valid1(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("0,100,2")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_range_valid2(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("100,100,2")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_range_valid3(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("3,2,2")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_range_valid4(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("3,11,2")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_range_valid5(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("3,10,0")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_range_valid6(self):
        with pytest.raises(CompareError) as error:
            range_manager = RangeMode("3,10,10")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_selected_valid1(self):
        with pytest.raises(CompareError) as error:
            range_manager = SelectMode("99")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_has_range1(self):
        args = ['-r']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_has_range2(self):
        args = ['--range']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_has_range3(self):
        args = ['-m']
        with mock.patch('sys.argv', args):
            self.assertFalse(RangeManager._has_cmd())

    def test_has_range4(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_has_range5(self):
        args = ['--select']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_adjust_header1(self):
        args = ['-m']
        with mock.patch('sys.argv', args):
            header = ['h1', 'h2', 'h3']
            RangeManager.adjust_header(header)
        self.assertEqual(['h1', 'h2', 'h3'], header)

    def test_adjust_header2(self):
        args = ['-r']
        with mock.patch('sys.argv', args):
            header = ['h1', 'h2', 'h3']
            RangeManager.adjust_header(header)
        self.assertEqual(['h1', 'OpSequence', 'h2', 'h3'], header)

    def test_adjust_header3(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            header = ['h1', 'h2', 'h3']
            RangeManager.adjust_header(header)
        self.assertEqual(['h1', 'OpSequence', 'h2', 'h3'], header)

    def test_adjust_data1(self):
        args = ['-m']
        with mock.patch('sys.argv', args):
            data = ['h1', 'h2', 'h3']
            RangeManager.adjust_data(data, 2)
        self.assertEqual(['h1', 'h2', 'h3'], data)

    def test_adjust_data2(self):
        args = ['-r']
        with mock.patch('sys.argv', args):
            data = ['h1', 'h2', 'h3']
            RangeManager.adjust_data(data, 1)
        self.assertEqual(['h1', '1', 'h2', 'h3'], data)

    def test_adjust_data3(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            data = ['h1', 'h2', 'h3']
            RangeManager.adjust_data(data, 1)
        self.assertEqual(['h1', '1', 'h2', 'h3'], data)

    def test_get_range_ops1(self):
        compare_rule = mock.Mock
        compare_rule.fusion_info = mock.Mock
        op1 = fusion_op.FusionOp(0, 'a', [], 'data', [], fusion_op.OpAttr([], '', False, 1))
        op2 = fusion_op.FusionOp(0, 'b', [], 'data', [], fusion_op.OpAttr([], '', False, 2))
        op3 = fusion_op.FusionOp(0, 'c', [], 'data', [], fusion_op.OpAttr([], '', False, 3))
        op4 = fusion_op.FusionOp(0, 'd', [], 'data', [], fusion_op.OpAttr([], '', False, 4))
        op5 = fusion_op.FusionOp(0, 'e', [], 'data', [], fusion_op.OpAttr([], '', False, 5))
        op6 = fusion_op.FusionOp(0, 'f', [], 'data', [], fusion_op.OpAttr([], '', False, 6))
        op7 = fusion_op.FusionOp(0, 'g', [], 'data', [], fusion_op.OpAttr([], '', False, 7))
        op8 = fusion_op.FusionOp(0, 'h', [], 'data', [], fusion_op.OpAttr([], '', False, 8))
        op9 = fusion_op.FusionOp(0, 'i', [], 'data', [], fusion_op.OpAttr([], '', False, 9))
        op10 = fusion_op.FusionOp(0, 'j', [], 'data', [], fusion_op.OpAttr([], '', False, 10))
        fusion_list = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10]
        compare_rule.fusion_info.op_list = fusion_list
        compare_rule.fusion_info.op_name_to_fusion_op_name_map = {"a": 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e',
                                                                  'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j'}
        range_manager = RangeMode("3,10,2")
        range_manager.check_input_valid(100)
        op_name_list = range_manager.get_all_ops(compare_rule)
        self.assertEqual(5, len(op_name_list))
        self.assertEqual(['c', 'e', 'g', 'i', 'j'], op_name_list)

    def test_get_range_ops2(self):
        compare_rule = mock.Mock
        compare_rule.fusion_info = mock.Mock
        op1 = fusion_op.FusionOp(0, 'a', [], 'data', [], fusion_op.OpAttr([], '', False, 1))
        op2 = fusion_op.FusionOp(0, 'b', [], 'data', [], fusion_op.OpAttr([], '', False, 2))
        op3 = fusion_op.FusionOp(0, 'c', [], 'data', [], fusion_op.OpAttr([], '', False, 3))
        op4 = fusion_op.FusionOp(0, 'd', [], 'data', [], fusion_op.OpAttr([], '', False, 4))
        op5 = fusion_op.FusionOp(0, 'e', [], 'data', [], fusion_op.OpAttr([], '', False, 5))
        op6 = fusion_op.FusionOp(0, 'f', [], 'data', [], fusion_op.OpAttr([], '', False, 6))
        op7 = fusion_op.FusionOp(0, 'g', [], 'data', [], fusion_op.OpAttr([], '', False, 7))
        op8 = fusion_op.FusionOp(0, 'h', [], 'data', [], fusion_op.OpAttr([], '', False, 8))
        op9 = fusion_op.FusionOp(0, 'i', [], 'data', [], fusion_op.OpAttr([], '', False, 9))
        op10 = fusion_op.FusionOp(0, 'j', [], 'data', [], fusion_op.OpAttr([], '', False, 10))
        fusion_list = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10]
        compare_rule.fusion_info.op_list = fusion_list
        compare_rule.fusion_info.op_name_to_fusion_op_name_map = {"a": 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e',
                                                                  'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j'}
        range_manager = SelectMode("3,5,2")
        range_manager.check_input_valid(100)
        op_name_list = range_manager.get_all_ops(compare_rule)
        self.assertEqual(3, len(op_name_list))
        self.assertEqual(['b', 'c', 'e'], op_name_list)

    def test_get_range_ops3(self):
        compare_rule = mock.Mock
        compare_rule.fusion_info = mock.Mock
        op1 = fusion_op.FusionOp(0, 'a', [], 'data', [], fusion_op.OpAttr([], '', False, 1))
        op2 = fusion_op.FusionOp(0, 'b', [], 'data', [], fusion_op.OpAttr([], '', False, 2))
        op3 = fusion_op.FusionOp(0, 'c', [], 'data', [], fusion_op.OpAttr([], '', False, 3))
        op4 = fusion_op.FusionOp(0, 'd', [], 'data', [], fusion_op.OpAttr([], '', False, 4))
        op5 = fusion_op.FusionOp(0, 'e', [], 'data', [], fusion_op.OpAttr([], '', False, 5))
        op6 = fusion_op.FusionOp(0, 'f', [], 'data', [], fusion_op.OpAttr([], '', False, 6))
        op7 = fusion_op.FusionOp(0, 'g', [], 'data', [], fusion_op.OpAttr([], '', False, 7))
        op8 = fusion_op.FusionOp(0, 'h', [], 'data', [], fusion_op.OpAttr([], '', False, 8))
        op9 = fusion_op.FusionOp(0, 'i', [], 'data', [], fusion_op.OpAttr([], '', False, 9))
        op10 = fusion_op.FusionOp(0, 'j', [], 'data', [], fusion_op.OpAttr([], '', False, 10))
        fusion_list = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10]
        compare_rule.fusion_info.op_list = fusion_list
        compare_rule.fusion_info.op_name_to_fusion_op_name_map = {"a": 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e',
                                                                  'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j'}
        range_manager = SelectMode("3,5,20")
        range_manager.check_input_valid(100)
        op_name_list = range_manager.get_all_ops(compare_rule)
        self.assertEqual(2, len(op_name_list))
        self.assertEqual(['c', 'e'], op_name_list)
