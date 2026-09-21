# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import os
import subprocess
import xml.etree.ElementTree as ET


def get_diff_added_lines():
    """获取PR中新增的代码行，并输出调试信息"""
    diff_cmd = ["git", "diff", "origin/master...HEAD", "--unified=0", "--", ":!test_*"]
    print(f"Running command: {' '.join(diff_cmd)}")
    diff_output = subprocess.run(diff_cmd, capture_output=True, text=True).stdout

    added_lines = {}  # 存储每个文件的新增行号
    current_file = None

    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):  # 新文件名
            current_file = line[6:]
            if "test/" in current_file or current_file.startswith("test_") or current_file.endswith(".md"):
                print(f"Ignoring test file: {current_file}")
                current_file = None  # 置空当前文件，避免处理后续内容
                continue
            added_lines[current_file] = []
            print(f"Detected new file: {current_file}")  # 调试信息：新文件
        elif line.startswith("@@") and current_file is not None:
            # 提取行号信息，形如 @@ -73,0 +74,5 @@
            hunk_info = line.split()[2]
            added_start_line = int(hunk_info.split(",")[0][1:])
            line_count = int(hunk_info.split(",")[1]) if "," in hunk_info else 1
            added_lines[current_file].extend(range(added_start_line, added_start_line + line_count))
            # print(f"Added lines for {current_file}: {added_lines[current_file]}")  # 调试信息：新增行号

    # print(f"Added lines: {added_lines}")  # 输出所有新增行
    return added_lines


def parse_coverage(coverage_file):
    """解析coverage.xml并提取覆盖的行，并输出调试信息"""
    print(f"Parsing coverage file: {coverage_file}")
    if not os.path.exists(coverage_file):
        print(f"Coverage file {coverage_file} does not exist!")
        return {}

    tree = ET.parse(coverage_file)
    root = tree.getroot()

    covered_lines = {}

    # 查找文件覆盖信息
    for file_elem in root.findall(".//class"):
        filename = file_elem.attrib['filename']
        lines = {"covered": [], "uncovered": []}
        for line_elem in file_elem.findall(".//line"):
            line_num = int(line_elem.attrib['number'])
            if line_elem.attrib['hits'] == "0":
                lines["uncovered"].append(line_num)
            else:
                lines["covered"].append(line_num)

        covered_lines[filename] = lines
        # print(f"Covered lines in {filename}: {lines}")  # 调试信息：文件的覆盖行

    # print(f"Covered lines: {covered_lines}")  # 输出所有覆盖行
    return covered_lines


def normalize_path(file_path, keep_parts=2):
    """
    移除路径的前缀，只保留文件名的相对部分。

    :param file_path: 原始文件路径
    :param keep_parts: 需要保留的路径部分数（从后向前数）
    :return: 标准化并截取后的路径
    """
    # 将路径标准化
    normalized_path = os.path.normpath(file_path)

    # 使用 os.path.split 分割路径成列表
    path_parts = normalized_path.split(os.sep)

    # 保留最后的 keep_parts 部分
    return os.path.join(*path_parts[-keep_parts:])


def calculate_coverage(added_lines, covered_lines):
    """计算新增代码的覆盖率，并输出调试信息"""
    total_added = 0
    total_covered = 0
    total_annotation = 0

    for filename, lines in added_lines.items():
        # 去掉路径的前缀，只比较末尾部分的路径
        normalized_filename = normalize_path(filename)
        print(f"normalized_filename:{normalized_filename}")

        found = False
        for covered_file in covered_lines:

            if normalize_path(covered_file).endswith(normalized_filename):
                found = True
                print(f"Processing file: {filename}")  # 调试信息：处理文件
                covered_file_lines = covered_lines[covered_file]["covered"]
                uncovered_file_lines = covered_lines[covered_file]["uncovered"]
                for line in lines:
                    total_added += 1
                    if line in covered_file_lines:
                        total_covered += 1
                        print(f"Line {line} in {filename} is covered.")  # 调试信息：被覆盖的行
                    elif line in uncovered_file_lines:
                        pass
                        print(f"Line {line} in {filename} is NOT covered.")  # 调试信息：未覆盖的行
                    else:
                        total_annotation += 1
                        print(f"Line {line} in {filename} is not present in coverage report.")  # 调试信息：未列入的行
                break

        if not found:
            print(f"File {filename} not found in coverage report.")  # 调试信息：找不到的文件
    # 计算有效代码行
    total_effective_lines = total_added - total_annotation

    # 输出总的新增行和覆盖行
    print(f"Total added lines: {total_added}")
    print(f"Total annotation lines: {total_annotation}")
    print(f"Total effective lines: {total_effective_lines}")
    print(f"Total covered lines: {total_covered}")

    coverage_rate = (total_covered / total_effective_lines * 100) if total_effective_lines > 0 else 0

    return coverage_rate


if __name__ == "__main__":
    # 请先运行UT
    added_lines = get_diff_added_lines()  # 获取新增代码行
    cur_dir = os.path.realpath(os.path.dirname(__file__))  # 当前脚本的目录
    coverage_file = os.path.join(cur_dir, "report", "coverage.xml")  # 假设coverage.xml生成在report目录下
    covered_lines = parse_coverage(coverage_file)  # 获取覆盖的代码行
    coverage_rate = calculate_coverage(added_lines, covered_lines)  # 计算覆盖率
    print(f"Coverage rate of added code: {coverage_rate:.2f}%")