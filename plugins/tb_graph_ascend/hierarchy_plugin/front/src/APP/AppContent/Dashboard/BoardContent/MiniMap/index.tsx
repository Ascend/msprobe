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
import { useEffect, useRef, useCallback } from 'react';
import type { HierarchyObjectType, HierarchyNodeType, PreProcessDataConfigType, GraphType } from '../../type';
import styles from './index.module.less';
import { isEmpty, throttle } from 'lodash';
import { changeGraphPosition } from '../../../../../common/utils';
import {
  BASE_NODE_COLOR,
  BENCH_NODE_COLOR,
  MOVE_STEP,
  NO_MATCHED_NODE_COLOR,
  NODE_TYPE,
  OVERFLOW_COLOR,
} from '../../../../../common/constant';
import useGraphStore from '../../../../../store/useGraphStore';

interface MiniMapProps {
  transform: { x: number; y: number; scale: number };
  setTransform: (transform: { x: number; y: number; scale: number }) => void;
  graphType: GraphType;
  graph: SVGSVGElement | null;
  container: SVGGElement | null;
  hierarchyObject: HierarchyObjectType;
}

const CANVAS_WIDTH = 100;
const CANVAS_HEIGHT = 400;

const MiniMap = (props: MiniMapProps) => {
  const { hierarchyObject, graph, container, graphType, transform, setTransform } = props;
  const colors = useGraphStore((state) => state.colors);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const startYRef = useRef<number>(0);
  const scaleRef = useRef<number>(1);
  // 拖拽相关 refs
  const isDraggingRef = useRef(false);
  const dragOffsetXRef = useRef(0);
  const dragOffsetYRef = useRef(0);
  const currentViewportRef = useRef({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  });

  // ================== 绘制函数 ==================
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

  const calcColorByPrecision = (precisionValue: number, colors: PreProcessDataConfigType['colors']) => {
    if (isNaN(precisionValue)) return BASE_NODE_COLOR;
    for (const [color, config] of Object.entries(colors)) {
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

  const drawMinimap = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;
    const isOverflowFilter = false;
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    const nodes = Object.values(hierarchyObject);
    if (isEmpty(nodes)) return;
    // 计算比例关系
    const svgRect = graph?.getBoundingClientRect();
    const viewWidth = svgRect?.width || 600;
    const viewHeight = svgRect?.height || 800;

    const scale = CANVAS_WIDTH / nodes[0]?.width || 1;
    scaleRef.current = scale;
    const { height } = currentViewportRef.current;
    // 计算视口（用于绘制 & 拖拽检测）
    const containerTransform = transform; // 获取当前的transform，不要通过container获取，因为container的transform可能不是最新的
    const vx = -(containerTransform.x / containerTransform.scale) * scaleRef.current;
    const vy = -(containerTransform.y / containerTransform.scale) * scaleRef.current;
    const viewportWidth = (viewWidth / containerTransform.scale) * scaleRef.current;
    const viewportHeight = (viewHeight / containerTransform.scale) * scaleRef.current;
    // 缓存当前视口位置（用于拖拽）
    currentViewportRef.current = { x: vx, y: vy, width: viewportWidth, height: viewportHeight };
    if (vy + height > CANVAS_HEIGHT) {
      startYRef.current = vy + height - CANVAS_HEIGHT;
    } else if (vy < 0) {
      startYRef.current = vy;
    } else {
      startYRef.current = 0;
    }
    // 绘制节点
    for (const node of nodes) {
      const x = node.x * scale;
      const y = node.y * scale - startYRef.current;
      const w = node.width * scale;
      const h = 15 * scale;

      if (y > CANVAS_HEIGHT || y + h < 0) continue;
      let precisionColor = isOverflowFilter ? getOverflowColor(node) : getPrecisionColor(node, colors, graphType);
      ctx.fillStyle = precisionColor;
      if (node.nodeType === NODE_TYPE.UNEXPANDED_NODE) {
        ctx.beginPath();
        ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#999999ff';
        ctx.lineWidth = 1;
        ctx.stroke();
      } else {
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#999999ff';
        ctx.strokeRect(x, y, w, h);
      }

      if (node.nodeType !== NODE_TYPE.UNEXPANDED_NODE && node.expand) {
        const totalHeight = node.height * scale;
        ctx.strokeStyle = '#999999ff';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, w, totalHeight);
      }
    }

    // 绘制视口
    ctx.fillStyle = '#268f323d';
    ctx.fillRect(vx, vy - startYRef.current, viewportWidth, viewportHeight);
    ctx.strokeRect(vx, vy - startYRef.current, viewportWidth, viewportHeight);
  }, [hierarchyObject, graph, container, transform, colors]);

  // ================== 拖拽逻辑 ==================

  const handleMouseDown = useCallback(
    (e: MouseEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const { x, y, width, height } = currentViewportRef.current;
      const viewportTop = y - startYRef.current;
      // 判断是否点击在视口框内
      if (mouseX >= x && mouseX <= x + width && mouseY >= viewportTop && mouseY <= viewportTop + height) {
        isDraggingRef.current = true;
        dragOffsetXRef.current = mouseX - x;
        dragOffsetYRef.current = mouseY - y + startYRef.current;
        canvas.style.cursor = 'grabbing';
      } else {
        const newVx = mouseX - width / 2;
        const newVy = mouseY - height / 2 + startYRef.current;
        // 反向映射到主图的世界坐标
        const worldX = -(newVx / scaleRef.current) * transform.scale;
        const worldY = -(newVy / scaleRef.current) * transform.scale;
        changeGraphPosition(container as unknown as HTMLElement, worldX, worldY, transform.scale, 16);
        // 更新主图 transform（保持 scale 不变）
        setTransform({
          x: worldX,
          y: worldY,
          scale: transform.scale,
        });
      }
    },
    [transform.scale, setTransform, container, transform],
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      // 新的视口左上角（canvas 坐标）
      const newVx = mouseX - dragOffsetXRef.current;
      const newVy = mouseY - dragOffsetYRef.current + startYRef.current;
      // 反向映射到主图的世界坐标
      const worldX = -(newVx / scaleRef.current) * transform.scale;
      const worldY = -(newVy / scaleRef.current) * transform.scale;
      changeGraphPosition(container as unknown as HTMLElement, worldX, worldY, transform.scale, 16);
      // 更新主图 transform（保持 scale 不变）
      setTransform({
        x: worldX,
        y: worldY,
        scale: transform.scale,
      });
    },
    [transform.scale, setTransform, container],
  );

  const handleMouseUp = useCallback(() => {
    if (isDraggingRef.current) {
      isDraggingRef.current = false;
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'default';
      }
    }
  }, []);

  // ================== 事件绑定 ==================

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctxRef.current = ctx;

    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;

    // Wheel 事件（垂直滚动）
    const throttledWheel = throttle((e: WheelEvent) => {
      e.preventDefault();
      const y = (transform.y += e.deltaY > 0 ? -MOVE_STEP : MOVE_STEP);
      // 此处需要抽离为公方法
      changeGraphPosition(container as unknown as HTMLElement, transform.x, transform.y, transform.scale, 16);
      // 更新主图 transform（保持 scale 不变）
      setTransform({
        x: transform.x,
        y: y,
        scale: transform.scale,
      });
    }, 16);
    canvas.addEventListener('wheel', throttledWheel, { passive: false });
    canvas.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      canvas.removeEventListener('wheel', throttledWheel);
      canvas.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [drawMinimap, handleMouseDown, handleMouseMove, handleMouseUp]);

  // ================== 响应主图变化 ==================

  useEffect(() => {
    drawMinimap();
  }, [transform, hierarchyObject, colors]);

  return (
    <div className={styles.miniMap} id="minimap">
      <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
    </div>
  );
};

export default MiniMap;
