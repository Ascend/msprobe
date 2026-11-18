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

import React, { memo, useMemo, useState } from 'react';
import { Button, message, Select, Spin, Splitter } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { useGlobalStore } from '../../store/useGlobalStore';
import { LINE_MODE_OPTIONS, CLEAR_ICON } from '../../common/constant';
import { formatCurveLabel } from '../../utils';
import HeatMap from './HeatMap';
import LineChart from './LineChart';

import './index.less';

const Graph = () => {
  const loadingHeatMap = useGlobalStore((state) => state.loadingHeatMap);
  const loadingLineChart = useGlobalStore((state) => state.loadingLineChart);
  const viewMode = useGlobalStore((state) => state.viewMode); // 折线图模式：全部/原始/Diff（折线图顶部工具栏选择）
  const setViewMode = useGlobalStore((state) => state.setViewMode);
  const trendData = useGlobalStore((state) => state.trendData);
  const dimension = useGlobalStore((state) => state.dimension);
  const diffPairs = useGlobalStore((state) => state.diffPairs);
  const addDiffPair = useGlobalStore((state) => state.addDiffPair);
  const setTrendData = useGlobalStore((state) => state.setTrendData);
  const setDiffPairs = useGlobalStore((state) => state.setDiffPairs);
  const setHiddenLegendNames = useGlobalStore((state) => state.setHiddenLegendNames);

  // diff 曲线选择状态（折线图顶部工具栏，与模式选择/清空同一行）
  const [curveASelection, setCurveASelection] = useState<string | undefined>(undefined);
  const [curveBSelection, setCurveBSelection] = useState<string | undefined>(undefined);

  // 构建可选曲线列表（与图例共用 formatCurveLabel，保证 A/B 选中与图例名一致）
  const curveOptions = useMemo(() => {
    return trendData.map((trend) => {
      const key = `${trend.dimX}_${trend.dimY}`;
      const label = formatCurveLabel(trend, dimension);
      return { value: key, label };
    });
  }, [trendData, dimension]);

  const handleAddDiff = () => {
    if (!curveASelection || !curveBSelection) return;
    if (curveASelection === curveBSelection) return;
    // 校验 A/B 仍存在于当前曲线列表：曲线被删除（双击图例）后残留的选中值不应
    // 再进 diffPairs，否则生成差值曲线用 findCurve 匹配不到直接整条消失
    const validKeys = new Set(curveOptions.map((o) => o.value));
    if (!validKeys.has(curveASelection) || !validKeys.has(curveBSelection)) {
      message.warning('所选曲线已不存在，请重新选择');
      setCurveASelection(undefined);
      setCurveBSelection(undefined);
      return;
    }
    // 已存在相同 pair（含 A/B 互换）→ 提示而非静默，避免用户误以为已添加
    const dup = diffPairs.some(
      (p) =>
        (p.curveA === curveASelection && p.curveB === curveBSelection) ||
        (p.curveA === curveBSelection && p.curveB === curveASelection),
    );
    if (dup) {
      message.info('该差值曲线已存在');
      return;
    }
    addDiffPair(curveASelection, curveBSelection);
    setCurveASelection(undefined);
    setCurveBSelection(undefined);
  };

  // 清空：清掉所有曲线与 diff（图标保持 toolbox 样式，与模式/差值选择框同一水平线）
  const handleClear = () => {
    setTrendData([]);
    setDiffPairs([]);
    setHiddenLegendNames([]);
  };

  return (
    <div className="graph-container">
      <div className="main-graph">
        <Splitter layout="vertical" style={{ height: '100%', boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)' }}>
          <Splitter.Panel style={{ height: '100%' }} defaultSize="60%">
            <Spin size="middle" spinning={loadingHeatMap}>
              <HeatMap />
            </Spin>
          </Splitter.Panel>
          <Splitter.Panel style={{ height: '100%' }} defaultSize="40%">
            <div className="line-panel">
              {/* 折线图顶部工具栏：模式选择 / 差值曲线选择 / 清空 同一行 */}
              <div className="line-toolbar">
                <div className="toolbar-item mode-item">
                  <span className="toolbar-label">折线图模式</span>
                  <Select
                    className="mode-select"
                    size="small"
                    style={{ width: 110 }}
                    value={viewMode}
                    onChange={setViewMode}
                    options={LINE_MODE_OPTIONS}
                  />
                </div>
                <div className="toolbar-item diff-item">
                  <Select
                    className="diff-select"
                    size="small"
                    style={{ width: 150 }}
                    placeholder="曲线 A"
                    value={curveASelection}
                    onChange={setCurveASelection}
                    options={curveOptions}
                    showSearch
                    filterOption={(input, option) =>
                      (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    allowClear
                    notFoundContent="无可用曲线"
                  />
                  <span className="diff-minus">－</span>
                  <Select
                    className="diff-select"
                    size="small"
                    style={{ width: 150 }}
                    placeholder="曲线 B"
                    value={curveBSelection}
                    onChange={setCurveBSelection}
                    options={curveOptions}
                    showSearch
                    filterOption={(input, option) =>
                      (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    allowClear
                    notFoundContent="无可用曲线"
                  />
                  <Button
                    type="primary"
                    size="small"
                    icon={<PlusOutlined />}
                    onClick={handleAddDiff}
                    disabled={!curveASelection || !curveBSelection || curveASelection === curveBSelection}
                  >
                    添加差值
                  </Button>
                </div>
                {/* 清空：原 toolbox 图标样式，与模式/差值选择框同一水平线 */}
                <div className="toolbar-item clear-item">
                  <Button
                    className="clear-btn"
                    type="text"
                    size="small"
                    title="清空"
                    onClick={handleClear}
                    icon={
                      <svg viewBox="0 0 100 100" width="14" height="14" aria-hidden>
                        <path
                          d={CLEAR_ICON.replace('path://', '')}
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="10"
                        />
                      </svg>
                    }
                  />
                </div>
              </div>
              {/* 折线图主体（三模式由 LineChart 内部按 viewMode 分支） */}
              <div className="line-body">
                <Spin size="middle" spinning={loadingLineChart}>
                  <LineChart />
                </Spin>
              </div>
            </div>
          </Splitter.Panel>
        </Splitter>
      </div>
    </div>
  );
};

export default memo(Graph);
