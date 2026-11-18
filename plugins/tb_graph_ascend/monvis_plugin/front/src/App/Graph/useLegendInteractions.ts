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

import { useCallback, useRef } from 'react';
import { useGlobalStore } from '../../store/useGlobalStore';

/** 双击判定窗口（ms），与操作系统标准双击阈值一致：窗口内同一条图例再次点击 = 删除 */
const DOUBLE_CLICK_MS = 500;

/**
 * 构建 legend.selected：只列出被隐藏的项(false)，其余默认显示。
 * 通过 getState() 现取而非订阅——单击隐藏时 echarts 已原生切换，无需重渲染；
 * 该状态只在图表重建（加/删曲线、切模式）时注入 option。
 */
export const getLegendSelected = (): Record<string, boolean> => {
  const selected: Record<string, boolean> = {};
  useGlobalStore
    .getState()
    .hiddenLegendNames.forEach((name) => {
      selected[name] = false;
    });
  return selected;
};

/**
 * 图例统一交互（LineChart 三模式共用）：
 * 单击 = 隐藏/显示（持久化到 store，加新曲线不重置）；
 * 快速双击 = 删除该曲线（原始曲线连带其 diff，diff 曲线删自身）。
 * 不订阅隐藏状态：单击后 echarts 原生切换即见效，避免 option 重建的卡顿与动画回放。
 */
export default function useLegendInteractions() {
  const setHiddenLegendNames = useGlobalStore((state) => state.setHiddenLegendNames);
  const removeSeriesByName = useGlobalStore((state) => state.removeSeriesByName);
  const lastClickRef = useRef<{ name: string; time: number }>({ name: '', time: 0 });

  const onLegendSelectChanged = useCallback(
    (params: { name: string; selected: Record<string, boolean> }) => {
      const { name, selected } = params;
      const now = Date.now();
      const isDoubleClick = lastClickRef.current.name === name && now - lastClickRef.current.time < DOUBLE_CLICK_MS;
      lastClickRef.current = { name, time: now };
      if (isDoubleClick) {
        // 快速双击：删除该曲线（原始曲线连带其 diff，diff 曲线删自身）
        removeSeriesByName(name);
        return;
      }
      // 单击：params.selected 只含当前图表图例的曲线名，其他图表/模式已隐藏的记录需保留。
      // 若整体替换，会丢弃本图 selected 之外的旧隐藏记录（如叠加模式隐藏的差值曲线，
      // 切到原始模式再隐藏任一原始曲线后，切回时差值曲线会重新显示）。
      const prevHidden = useGlobalStore.getState().hiddenLegendNames;
      const hiddenSet = new Set(prevHidden);
      Object.entries(selected).forEach(([name, isSelected]) => {
        if (isSelected) {
          hiddenSet.delete(name);
        } else {
          hiddenSet.add(name);
        }
      });
      setHiddenLegendNames(Array.from(hiddenSet));
    },
    [removeSeriesByName, setHiddenLegendNames],
  );

  return { onLegendSelectChanged };
}
