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
import { expect } from '@playwright/test';
import { test } from './entity';
import { MAX_DIFF_PIXELS, SIDER_TYPE } from './constants';

const setupBeforeTest = (sider: SIDER_TYPE) => {
  return test.beforeEach(async ({ page, mainPage, metaContentPanel }) => {
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
    await mainPage.siderButtons[sider].click();
  });
};

test.describe('FileSelectSiderTest', () => {
  test.beforeEach(async ({ page, mainPage, metaContentPanel }) => {
    const allParsedPromise = page.waitForResponse(
      (response) => response.url().includes('/loadGraphConfigInfo') && response.status() === 200,
    );
    // 概率性出现长时间加载中状态而导致page.goto超时，但其实界面已加载完成，不影响后续测试
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await allParsedPromise;
    await mainPage.fileSiderButton.click();
    await expect(metaContentPanel.panel).toBeVisible();
  });

  // 文件选择看板相关测试
  test('testFileSelectPanel', async ({ metaContentPanel }) => {
    const {
      panel,
      dirSelector,
      fileSelector,
      dirOptions,
      fileOptions,
      stepSelector,
      rankSelector,
      microStepSelector,
      getSelectOption,
    } = metaContentPanel;
    await dirSelector.click();
    await dirOptions.communicationDirOption.click();
    await fileSelector.click();
    await fileOptions.communicationFileOption.click();
    await expect(stepSelector).toBeVisible();
    await expect(rankSelector).toBeVisible();
    await expect(microStepSelector).toBeVisible();
    await stepSelector.click();
    await getSelectOption('0').click();
    await rankSelector.click();
    await getSelectOption('1').click();
    await microStepSelector.click();
    await getSelectOption('2').nth(1).click();
    await expect(panel).toHaveScreenshot('fileSelectPanel.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });
});

test.describe('NodeSearchSiderTest', () => {
  setupBeforeTest(SIDER_TYPE.SEARCH);

  // 双图比对和单图展示场景下标杆侧按钮是否可用测试
  test('benchRadioShouldDisabledInSingleGraph', async ({ mainPage, metaContentPanel, nodeSearchPanel }) => {
    const { dirSelector, fileSelector, dirOptions, fileOptions } = metaContentPanel;
    const { benchRadio } = nodeSearchPanel;
    // 双图比对场景下两个勾选框都可用
    await expect(benchRadio).toBeEnabled();
    // 单图展示场景下标杆侧勾选框不可用
    await mainPage.fileSiderButton.click();
    await expect(metaContentPanel.panel).toBeVisible();
    await dirSelector.click();
    await dirOptions.singleDirOption.click();
    await fileSelector.click();
    await fileOptions.singleFileOption.click();
    await expect(benchRadio).toBeDisabled();
  });

  // 节点搜索功能测试
  test('displayedNodeListShouldContainSearchWord', async ({ nodeSearchPanel }) => {
    const { panel, nodeSearch, nodeCountLabel, clearButton } = nodeSearchPanel;
    await expect(nodeCountLabel).toContainText('151');
    await nodeSearch.fill('conv1');
    await expect(nodeCountLabel).toContainText('18');
    await nodeSearch.fill('RELU');
    await expect(nodeCountLabel).toContainText('34');
    await expect(panel).toHaveScreenshot('nodeSearchResultWithCondition.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await clearButton.click();
    await expect(nodeCountLabel).toContainText('151');
  });

  // 选中节点列表和图上关联测试
  test('selectedNodeInGraphShouldSyncInList', async ({ page, mainPage, nodeSearchPanel }) => {
    const { mainArea, benchGraph } = mainPage;
    const { panel, benchRadio } = nodeSearchPanel;
    await panel.getByText('Module.relu.ReLU.forward.0').click();
    // 等待节点展开
    await page.waitForTimeout(2000);
    await expect(mainArea).toHaveScreenshot('locatedToSelectedNodeWhenClickDebugNodeList.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await panel.getByText('Module.layer1.1.conv1.Conv2d.').click();
    await page.waitForTimeout(2000);
    await expect(mainArea).toHaveScreenshot('changeSelectedNodeWhenClickNodeList.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await benchRadio.click();
    await panel.getByText('Module.layer1.0.bn1.').click();
    await page.waitForTimeout(2000);
    await expect(mainArea).toHaveScreenshot('locatedToSelectedNodeWhenClickBenchNodeList.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await benchGraph.getByText('Module.a…Module.avgpool.AdaptiveAvgPool2d.forward.0').click();
    await expect(panel.getByText('Module.avgpool.AdaptiveAvgPool2d.forward.0')).toBeVisible();
    await expect(mainArea).toHaveScreenshot('locatedToSelectedNodeWhenClickBenchNodeGraph.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
  });

  // 使用上一个/下一个图标进行选择节点切换
  test('switchSelectedNodeByUpAndDownIcon', async ({ nodeSearchPanel }) => {
    const { panel, upIcon, downIcon, nodeSearch } = nodeSearchPanel;
    const firstNodeDiv = panel.locator('div').filter({ hasText: /^Module\.conv1\.Conv2d\.forward\.0$/ });
    const secondNodeDiv = panel.locator('div').filter({ hasText: /^Module\.bn1\.BatchNorm2d\.forward\.0$/ });
    const secondToLastNodeDiv = panel
      .locator('div')
      .filter({ hasText: /^Module\.layer1\.0\.conv1\.Conv2d\.backward\.0$/ });
    const lastNodeDiv = panel.locator('div').filter({ hasText: /^Module\.conv1\.Conv2d\.backward\.0$/ });
    const nodeSelectedHoverColor = 'rgb(186, 224, 255)';
    const nodeSelectedColor = 'rgb(230, 244, 255)';
    const nodeNotSelectedColor = 'rgba(0, 0, 0, 0)';
    await firstNodeDiv.click();
    await expect(firstNodeDiv).toHaveCSS('background-color', nodeSelectedHoverColor);
    await upIcon.click();
    await expect(firstNodeDiv).toHaveCSS('background-color', nodeSelectedColor);
    await downIcon.click();
    await expect(firstNodeDiv).toHaveCSS('background-color', nodeNotSelectedColor);
    await expect(secondNodeDiv).toHaveCSS('background-color', nodeSelectedColor);
    await nodeSearch.fill('conv1');
    await lastNodeDiv.click();
    await downIcon.click();
    await expect(lastNodeDiv).toHaveCSS('background-color', nodeSelectedColor);
    await upIcon.click();
    await expect(lastNodeDiv).toHaveCSS('background-color', nodeNotSelectedColor);
    await expect(secondToLastNodeDiv).toHaveCSS('background-color', nodeSelectedColor);
  });
});
