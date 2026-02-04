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

import { expect, Page } from '@playwright/test';
import { test } from './entity';
import { MAX_DIFF_PIXELS } from './constants';

// 滚轮操作，拆解动作并设置间隔以提高测试稳定性
const mouseWheelMultipleTimes = async (page: Page, offsetX: number, offsetY: number, times: number) => {
  for (let i = 0; i < times; i++) {
    await page.mouse.wheel(offsetX, offsetY);
    await page.waitForTimeout(100);
  }
};

// 按数次键，设置间隔以提高测试稳定性
const pressKeyMultipleTimes = async (page: Page, key: string, times: number) => {
  for (let i = 0; i < times; i++) {
    await page.keyboard.press(key);
    await page.waitForTimeout(100);
  }
};

test.describe('MainGraphTest', () => {
  test.beforeEach(async ({ page, mainPage, metaContentPanel }) => {
    const allParsedPromise = page.waitForResponse(
      (response) => response.url().includes('/loadGraphConfigInfo') && response.status() === 200,
    );
    const { dirSelector, fileSelector, dirOptions, fileOptions } = metaContentPanel;
    // 概率性出现长时间加载中状态而导致page.goto超时，但其实界面已加载完成，不影响后续测试
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await allParsedPromise;
    await mainPage.fileSiderButton.click();
    await expect(metaContentPanel.panel).toBeVisible();
    await dirSelector.click();
    await dirOptions.compareDirOption.click();
    await fileSelector.click();
    await fileOptions.compareFileOption.click();
  });

  test('test_node_expand_and_collapse', async ({ mainPage, page }) => {
    const { mainArea, debugGraph, benchGraph, syncCheckBox } = mainPage;
    const npuNode = debugGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0');
    const benchNode = benchGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0');
    const npuSubNode = debugGraph.getByText('layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0');
    const benchSubNode = benchGraph.getByText('layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0');
    // 默认勾选同步展开对应侧节点，任意侧节点双击关联节点都会展开/收起
    await npuNode.dblclick();
    await expect(npuSubNode).toBeVisible();
    await expect(benchSubNode).toBeVisible();
    await expect(mainArea).toHaveScreenshot('nodeExpandedNpuAndBench.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await benchNode.dblclick();
    await expect(npuSubNode).toBeHidden();
    // 取消勾选同步展开对应侧节点，只能控制自身
    await syncCheckBox.click();
    await expect(syncCheckBox).not.toHaveCSS('background-color', 'rgba(230, 244, 255, 1) ');
    await npuNode.dblclick();
    await expect(npuSubNode).toBeVisible();
    await expect(benchSubNode).toBeHidden();
    await expect(mainArea).toHaveScreenshot('nodeExpandedOnlyNpu.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试模型图中鼠标事件，包含拖拽，图放缩
  test('test_mouse_drag_and_wheel_graph', async ({ page, mainPage }) => {
    const { mainArea } = mainPage;
    const { npuArea, benchArea } = await mainPage.getBoundingBoxes();
    const { x: startXNpu, y: startYNpu } = npuArea;
    const { x: startXBench, y: startYBench } = benchArea;
    const DRAG_START_OFFSET = 200;
    const DRAG_END_OFFSET = 400;
    // 测试鼠标拖拽移动
    await page.mouse.move(startXNpu + DRAG_START_OFFSET, startYNpu + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startXNpu + DRAG_END_OFFSET, startYNpu + DRAG_END_OFFSET);
    await page.mouse.up();
    await page.mouse.move(startXBench + DRAG_START_OFFSET, startYBench + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startXBench + DRAG_END_OFFSET, startYBench + DRAG_END_OFFSET);
    await page.mouse.up();
    await expect(mainArea).toHaveScreenshot('mouseDragDisplay.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    // 测试鼠标滚轮移动
    await page.mouse.move(startXNpu + DRAG_START_OFFSET, startYNpu + DRAG_START_OFFSET);
    await mouseWheelMultipleTimes(page, 0, 100, 4);
    await page.mouse.move(startXBench + DRAG_START_OFFSET, startYBench + DRAG_START_OFFSET);
    await mouseWheelMultipleTimes(page, 0, -100, 4);
    await expect(mainArea).toHaveScreenshot('mouseWheelDisplay.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试鼠标拖拽分割线
  test('test_mouse_drag_splitter', async ({ page, mainPage }) => {
    const { mainArea, splitter } = mainPage;
    const splitterBox = await splitter.boundingBox();
    if (!splitterBox) {
      throw new Error('Test failed because the splitter was not rendered correctly.');
    }
    await splitter.hover();
    await page.mouse.down();
    await page.mouse.move(splitterBox.x - 400, splitterBox.y);
    await page.mouse.up();
    await expect(mainArea).toHaveScreenshot('mouseDragSplitter.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试键盘控制左右移动和放缩
  test('test_move_and_scale_with_keyboard', async ({ page, mainPage }) => {
    const { mainArea } = mainPage;
    const { npuArea, benchArea } = await mainPage.getBoundingBoxes();
    const { x: startXNpu, y: startYNpu } = npuArea;
    const { x: startXBench, y: startYBench } = benchArea;
    await page.mouse.move(startXNpu, startYNpu);
    await pressKeyMultipleTimes(page, 'D', 4); // 左移
    await pressKeyMultipleTimes(page, 'W', 10); // 放大
    await page.mouse.move(startXBench, startYBench);
    await pressKeyMultipleTimes(page, 'A', 3); // 右移
    await pressKeyMultipleTimes(page, 'S', 3); // 缩小
    await expect(mainArea).toHaveScreenshot('keyboardOperations.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 测试缩略图相关
  test('test_minimap_display_and_controls', async ({ page, mainPage }) => {
    const { mainArea, npuMinimap } = mainPage;
    const { npuArea } = await mainPage.getBoundingBoxes();
    await page.mouse.move(npuArea.x, npuArea.y);
    await pressKeyMultipleTimes(page, 'D', 4);
    await pressKeyMultipleTimes(page, 'W', 10);
    const npuMinimapArea = await npuMinimap.boundingBox();
    if (!npuMinimapArea) {
      throw new Error('Test failed because the minimap was not rendered correctly.');
    }
    const { x: startX, y: startY } = npuMinimapArea;
    const DRAG_START_OFFSET = 50;
    const DRAG_END_OFFSET = 100;
    await page.mouse.move(startX + DRAG_START_OFFSET, startY + DRAG_START_OFFSET);
    await page.mouse.down();
    await page.mouse.move(startX + DRAG_END_OFFSET, startY + DRAG_END_OFFSET);
    await page.mouse.up();
    // 设置延迟等待动画完成
    await page.waitForTimeout(200);
    await expect(mainArea).toHaveScreenshot('mouseDragMinimap.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });
});
