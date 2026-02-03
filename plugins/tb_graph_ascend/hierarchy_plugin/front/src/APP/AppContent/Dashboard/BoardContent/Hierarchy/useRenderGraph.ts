import * as d3 from 'd3';
import { isEmpty } from 'lodash';
import {
  BASE_NODE_COLOR,
  BENCH_NODE_COLOR,
  DURATION_TIME,
  INIT_TRANSFORM,
  NO_MATCHED_NODE_COLOR,
  NODE_TYPE,
  NODE_TYPE_STYLES,
  OVERFLOW_COLOR,
  PREFIX_MAP,
  PRECIS_SELECTED_STROKE_COLOR,
  OVERFLOW_SELECTED_STROKE_COLOR,
  SELECTED_STROKE_WIDTH,
  STROKE_WIDTH,
} from '../../../../../common/constant';
import type {
  GraphType,
  HierarchyNodeType,
  HierarchyObjectType,
  PreProcessDataConfigType,
  PreProcessDataType,
  UseGraphType,
} from '../../type';

import { darkenColor, maybeTruncateString } from '../../..//../../common/utils';

const getPrecisionColor = (
  node: HierarchyNodeType,
  colors: PreProcessDataConfigType['colors'],
  graphType: GraphType,
) => {
  if (!colors || !graphType) return NO_MATCHED_NODE_COLOR;
  if (isEmpty(node.matchedNodeLink))
    return Object.keys(colors).find((color) => colors[color].value === '无匹配节点') ?? NO_MATCHED_NODE_COLOR;
  if (graphType === 'Bench') return BENCH_NODE_COLOR;

  const precisionValue = parseFloat(node.precisionIndex);
  return calcColorByPrecision(precisionValue, colors);
};

const getOverflowColor = (node: HierarchyNodeType) => {
  switch (node.overflowLevel) {
    case 'medium':
      return OVERFLOW_COLOR.medium;
    case 'high':
      return OVERFLOW_COLOR.high;
    case 'critical':
      return OVERFLOW_COLOR.critical;
    default:
      return OVERFLOW_COLOR.default;
  }
};

const preProcessData: PreProcessDataType = (
  hierarchyObject: { [key: string]: HierarchyNodeType },
  selectedNode: string,
  config: PreProcessDataConfigType,
  transform: { x: number; y: number; scale: number },
) => {
  const { colors, isOverflowFilter, graphType } = config;
  const data = Object.values(hierarchyObject);

  const virtualNodes = data.filter(
    (d) =>
      d.y >= (-Number(transform.y) - 1000) / Number(transform.scale) &&
      d.y <= (-Number(transform.y) + 2000) / Number(transform.scale),
  );

  const parentsVirtualNodes: Array<HierarchyNodeType> = [];
  virtualNodes.forEach((d) => {
    let node: HierarchyNodeType | undefined = d;
    while (node?.parentNode) {
      const parent: HierarchyNodeType = hierarchyObject[node.parentNode];
      if (parent && !virtualNodes.includes(parent) && !parentsVirtualNodes.includes(parent)) {
        parentsVirtualNodes.push(parent);
      }
      node = parent;
    }
  });

  const orderedNodes = [...new Set([...parentsVirtualNodes.reverse(), ...virtualNodes])];
  return orderedNodes.map((d) => {
    let precisionColor = isOverflowFilter ? getOverflowColor(d) : getPrecisionColor(d, colors, graphType);
    const selected_stroke_color = isOverflowFilter ? OVERFLOW_SELECTED_STROKE_COLOR : PRECIS_SELECTED_STROKE_COLOR;
    let strokeColor = d.name === selectedNode ? selected_stroke_color : darkenColor(precisionColor, 40);

    if (d.nodeType === NODE_TYPE.API_LIST || d.nodeType === NODE_TYPE.MULTI_COLLECTION) {
      precisionColor = 'white';
    }

    return {
      ...d,
      ...NODE_TYPE_STYLES[d.nodeType],
      color: precisionColor,
      stroke: strokeColor,
      strokeWidth: d.name === selectedNode ? SELECTED_STROKE_WIDTH : STROKE_WIDTH,
    };
  });
};

// ======================
// D3 渲染函数
// ======================
const bindInnerRect: UseGraphType['bindInnerRect'] = (container, data) => {
  const innerRect = container.selectAll('.inner-rect').data(data, (d: any) => d.name);
  innerRect
    .transition()
    .duration(DURATION_TIME)
    .attr('opacity', 1)
    .attr('x', (d: any) => d.x)
    .attr('y', (d: any) => d.y)
    .attr('width', (d: any) => d.width)
    .attr('fill', (d: any) => d.color);

  innerRect
    .enter()
    .append('rect')
    .attr('name', (d: any) => d.name)
    .attr('class', 'inner-rect')
    .attr('rx', (d: any) => d.rx)
    .attr('ry', (d: any) => d.ry)
    .attr('fill', (d: any) => d.color)
    .attr('x', (d: any) => d.x)
    .attr('y', (d: any) => d.y)
    .attr('width', (d: any) => d.width)
    .attr('height', 15)
    .attr('opacity', 0)
    .transition()
    .duration(DURATION_TIME + 60)
    .attr('opacity', 1);

  innerRect
    .exit()
    .transition()
    .duration(DURATION_TIME - 60)
    .attr('opacity', 0)
    .remove();
  innerRect.order();
};

const bindOuterRect: UseGraphType['bindOuterRect'] = (container, data) => {
  const outerRect = container.selectAll('.outer-rect').data(data, (d: any) => d.name);
  outerRect
    .transition()
    .duration(DURATION_TIME)
    .attr('opacity', 1)
    .attr('x', (d: any) => d.x)
    .attr('y', (d: any) => d.y)
    .attr('width', (d: any) => d.width)
    .attr('height', (d: any) => d.height)
    .attr('stroke', (d: any) => d.stroke)
    .attr('stroke-width', (d: any) => d.strokeWidth);

  outerRect
    .enter()
    .append('rect')
    .attr('name', (d: any) => d.name)
    .attr('class', 'outer-rect')
    .attr('rx', (d: any) => d.rx)
    .attr('ry', (d: any) => d.ry)
    .attr('fill', 'transparent')
    .attr('stroke', (d: any) => d.stroke)
    .attr('stroke-width', (d: any) => d.strokeWidth)
    .attr('stroke-dasharray', (d: any) => d.strokeDasharray)
    .attr('width', (d: any) => d.width)
    .attr('height', 15)
    .attr('x', (d: any) => d.x)
    .attr('y', (d: any) => d.y)
    .transition()
    .duration(DURATION_TIME + 60)
    .attr('height', (d: any) => d.height)
    .attr('opacity', 1);

  outerRect
    .exit()
    .transition()
    .duration(DURATION_TIME - 60)
    .attr('opacity', 0)
    .remove();
  outerRect.order();
};

const bindText: UseGraphType['bindText'] = (container, data) => {
  const texts = container.selectAll('text').data(data, (d: any) => d.name);
  texts
    .transition()
    .duration(DURATION_TIME)
    .attr('opacity', 1)
    .attr('x', (d: any) => d.x + d.width / 2)
    .attr('y', (d: any) => d.y + 8);

  texts
    .enter()
    .append('text')
    .attr('name', (d: any) => d.name)
    .attr('x', (d: any) => d.x + d.width / 2)
    .attr('y', (d: any) => d.y + 8)
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')

    .text((d: any) => maybeTruncateString(d.label, 9, d.width))
    .each(function (d: HierarchyNodeType) {
      // @ts-expect-error
      d3.select(this).append('title').text(d.label);
    })
    .style('font-size', (d: any) => `${d.fontSize}px`)
    .attr('opacity', 0)
    .transition()
    .duration(DURATION_TIME + 60)
    .attr('opacity', 1);

  texts
    .exit()
    .transition()
    .duration(DURATION_TIME - 60)
    .attr('opacity', 0)
    .remove();
  texts.order();
};

export const calcColorByPrecision = (precisionValue: number, colors: PreProcessDataConfigType['colors']) => {
  if (isNaN(precisionValue)) return BASE_NODE_COLOR;
  for (const [color, config] of Object.entries(colors || {})) {
    if (Array.isArray(config.value)) {
      const [min, max] = config.value;
      const isWithinRange = precisionValue >= min && precisionValue < max;
      const isMaxRange = max === 1 && precisionValue === 1;
      const isMinRange = min === 0 && precisionValue === 0;
      if (isWithinRange || isMaxRange || isMinRange) return color;
    }
  }
  return NO_MATCHED_NODE_COLOR;
};
// 渲染逻辑
export const renderGraph = (
  hierarchyObject: HierarchyObjectType,
  selectedNode: string,
  transform = INIT_TRANSFORM,
  container: SVGGElement | null,
  config: PreProcessDataConfigType,
) => {
  if (!container) return;
  const prefix = PREFIX_MAP[config.graphType];
  const selectedNodeName = selectedNode?.startsWith(prefix) ? selectedNode : `${prefix}${selectedNode}`;
  const renderData = preProcessData(hierarchyObject, selectedNodeName, config, transform);
  const containerD3 = d3.select(container);
  bindInnerRect(containerD3, renderData);
  bindOuterRect(containerD3, renderData);
  bindText(containerD3, renderData);
};
