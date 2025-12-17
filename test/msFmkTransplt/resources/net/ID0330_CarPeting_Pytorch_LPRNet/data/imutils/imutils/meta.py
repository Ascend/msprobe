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

from __future__ import print_function
import cv2
import re

def find_function(name, pretty_print=True, module=None):
	# if the module is None, initialize it to to the root `cv2`
	# library
	if module is None:
		module = cv2

	# grab all function names that contain `name` from the module
	p = ".*{}.*".format(name)
	filtered = filter(lambda x: re.search(p, x, re.IGNORECASE), dir(module))
	
	# check to see if the filtered names should be returned to the
	# calling function
	if not pretty_print:
		return filtered

	# otherwise, loop over the function names and print them
	for (i, funcName) in enumerate(filtered):
		print("{}. {}".format(i + 1, funcName))