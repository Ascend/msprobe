/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

export interface BuildInfoFormInterface {
  framework: string;
  npu_path: string;
  bench_path?: string;
  output_path?: string;
  is_print_compare_log?: boolean;
  parallel_merge?: boolean;
  npu?: { rank_size: number; tp: number; pp: number; vpp: number; order: string };
  bench?: { rank_size: number; tp: number; pp: number; vpp: number; order: string };
  layerMappingItems?: string;
  overflow_check?: boolean;
  fuzzy_match?: boolean;
}
