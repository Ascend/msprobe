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
import { expect, Page, Locator } from '@playwright/test';
import { test } from './entity';
import { FILES, MAX_DIFF_PIXELS, SIDER_TYPE } from './constants';

const setupBeforeTest = (sider: SIDER_TYPE) => {
  return test.beforeEach(async ({ mainPage, metaContentPanel }) => {
    const allParsedPromise = mainPage.page.waitForResponse(
      (response) => response.url().includes('/loadGraphConfigInfo') && response.status() === 200,
    );
    const { dirSelector, fileSelector, dirOptions, fileOptions } = metaContentPanel;
    // 概率性出现长时间加载中状态而导致page.goto超时，但其实界面已加载完成，不影响后续测试
    await mainPage.page.goto('/', { waitUntil: 'domcontentloaded' });
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

// 文件选择侧边栏
test.describe('FileSelectSiderTest', () => {
  test.beforeEach(async ({ mainPage, metaContentPanel }) => {
    const allParsedPromise = mainPage.page.waitForResponse(
      (response) => response.url().includes('/loadGraphConfigInfo') && response.status() === 200,
    );
    // 概率性出现长时间加载中状态而导致page.goto超时，但其实界面已加载完成，不影响后续测试
    await mainPage.page.goto('/', { waitUntil: 'domcontentloaded' });
    await allParsedPromise;
    await mainPage.fileSiderButton.click();
    await expect(metaContentPanel.panel).toBeVisible();
  });

  // 文件选择看板相关测试
  test('testFileSelectPanel', async ({ metaContentPanel, mainPage }) => {
    const { panel, dirSelector, fileSelector, dirOptions, fileOptions, stepSelector, rankSelector, microStepSelector } =
      metaContentPanel;
    const { getSelectOption } = mainPage;
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

// 节点筛选侧边栏
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
    await mainPage.searchSiderButton.click();
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
  test('selectedNodeInGraphShouldSyncInList', async ({ mainPage, nodeSearchPanel }) => {
    const { page, mainArea, benchGraph } = mainPage;
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

// 精度筛选和溢出检测侧边栏
test.describe('PrecisionSiderTest', () => {
  setupBeforeTest(SIDER_TYPE.PRECISION);

  // 不同数据会有不同呈现效果
  test('shouldDisplayCorrectlyInDifferentGraphType', async ({ mainPage, metaContentPanel, precisionFilertPanel }) => {
    const { panel, errorTab, overflowTab, precisionErrorTooltip, precisionErrorCheckBoxes, overflowCheckBoxes } =
      precisionFilertPanel;
    // 精度比对，未执行溢出检测数据场景，溢出筛选不可用，鼠标悬浮会有提示
    await expect(errorTab).toBeEnabled();
    await expect(overflowTab).toBeDisabled();
    await expect(precisionErrorTooltip).toBeVisible();
    await precisionErrorTooltip.hover();
    // 等待提示信息出现
    await mainPage.page.waitForTimeout(1000);
    await expect(panel).toHaveScreenshot('precisionErrorTooltip.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    Object.values(precisionErrorCheckBoxes).forEach(async (checkbox) => {
      await expect(checkbox).toBeEnabled();
    });

    // 执行了溢出检测数据场景，溢出筛选可用
    await mainPage.fileSiderButton.click();
    await metaContentPanel.dirSelector.click();
    await metaContentPanel.dirOptions.overflowDirOption.click();
    await expect(overflowTab).toBeEnabled();
    await mainPage.precisionSiderButton.click();
    await overflowTab.click();
    await expect(panel).toHaveScreenshot('displayInOverflowMode.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await expect(precisionErrorTooltip).toBeAttached({ attached: false });
    Object.values(overflowCheckBoxes).forEach(async (checkbox) => {
      await expect(checkbox).toBeEnabled();
    });

    // 单图场景，未执行溢出检测，溢出筛选不可用，所有节点不存在匹配关系，所以仅未匹配勾选框可用
    await mainPage.fileSiderButton.click();
    await metaContentPanel.dirSelector.click();
    await metaContentPanel.dirOptions.singleDirOption.click();
    await expect(overflowTab).toBeDisabled();
    await expect(precisionErrorTooltip).toBeAttached({ attached: false });
    await expect(precisionErrorCheckBoxes.pass).toBeDisabled();
    await expect(precisionErrorCheckBoxes.warning).toBeDisabled();
    await expect(precisionErrorCheckBoxes.error).toBeDisabled();
    await expect(precisionErrorCheckBoxes.unmatched).toBeEnabled();
  });

  // 精度误差筛选功能
  test('nodeListDisplayWithDifferentAccuracyFilterConditions', async ({ precisionFilertPanel }) => {
    const { panel, precisionErrorCheckBoxes, nodeCountLabel } = precisionFilertPanel;
    await expect(nodeCountLabel).toContainText('(0)');
    await precisionErrorCheckBoxes.pass.check();
    await expect(nodeCountLabel).toContainText('(112)');
    await precisionErrorCheckBoxes.pass.uncheck();
    await precisionErrorCheckBoxes.warning.check();
    await expect(nodeCountLabel).toContainText('(2)');
    await precisionErrorCheckBoxes.warning.uncheck();
    await precisionErrorCheckBoxes.error.check();
    await expect(nodeCountLabel).toContainText('(4)');
    await expect(panel).toHaveScreenshot('precisionPanelWithErrorCondition.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    await precisionErrorCheckBoxes.unmatched.check();
    await expect(nodeCountLabel).toContainText('(6)');
    await precisionErrorCheckBoxes.pass.check();
    await precisionErrorCheckBoxes.warning.check();
    await expect(nodeCountLabel).toContainText('(120)');
  });

  // 溢出筛选功能
  test('nodeListDisplayWithDifferentOverflowFilterConditions', async ({
    mainPage,
    metaContentPanel,
    precisionFilertPanel,
  }) => {
    const { panel, overflowTab, overflowCheckBoxes, nodeCountLabel } = precisionFilertPanel;
    await mainPage.fileSiderButton.click();
    await metaContentPanel.dirSelector.click();
    await metaContentPanel.dirOptions.overflowDirOption.click();
    await mainPage.precisionSiderButton.click();
    await overflowTab.click();
    await expect(nodeCountLabel).toContainText('(0)');
    await overflowCheckBoxes.medium.check();
    await expect(nodeCountLabel).toContainText('(1)');
    await overflowCheckBoxes.medium.uncheck();
    await overflowCheckBoxes.high.check();
    await expect(nodeCountLabel).toContainText('(41)');
    await overflowCheckBoxes.high.uncheck();
    await overflowCheckBoxes.ciritical.check();
    await expect(nodeCountLabel).toContainText('(1)');
    await panel.getByText('Tensor.__mul__.144.forward').click();
    await expect(mainPage.mainArea).toHaveScreenshot('overflowWithCriticalLevel.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await overflowCheckBoxes.medium.check();
    await overflowCheckBoxes.high.check();
    await expect(nodeCountLabel).toContainText('(43)');
  });
});

// 节点匹配侧边栏
test.describe('MatchSiderTest', () => {
  setupBeforeTest(SIDER_TYPE.MATCH);

  const locatedAndExpandMatchedModuleNodeByName = async (
    moduleNodeDisplayName: string,
    page: Page,
    debugGraph: Locator,
    benchGraph: Locator,
  ): Promise<void> => {
    const debugMatchedModuleNode = debugGraph.getByText(moduleNodeDisplayName);
    await expect(debugMatchedModuleNode).toBeVisible();
    await debugMatchedModuleNode.click({ button: 'right' });
    await page.getByRole('button', { name: 'aim Positioning matched node' }).click();
    const benchMatchedModuleNode = benchGraph.getByText(moduleNodeDisplayName);
    await expect(benchMatchedModuleNode).toBeVisible();
    await benchMatchedModuleNode.dblclick();
  };

  // 节点取消匹配测试（勾选后代节点）
  test('unmatchNodeWithCheckDesendant', async ({ mainPage, matchPanel }) => {
    const {
      debugMatchedCountLabel,
      debugMatchedSelect,
      benchMatchedCountLabel,
      unmatchButton,
      debugUnmatchedCountLabel,
      benchUnmatchedCountLabel,
      debugUnmatchedSelect,
    } = matchPanel;
    const { page, mainArea, debugGraph, benchGraph, getSelectOption } = mainPage;
    const searchStr = 'Module.layer1.0.conv1.Conv2d.forward.0';
    await debugMatchedSelect.locator('#rc_select_8').fill(searchStr);
    await getSelectOption(searchStr).click();
    const moduleNodeDisplayName = 'layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0';
    await locatedAndExpandMatchedModuleNodeByName(moduleNodeDisplayName, page, debugGraph, benchGraph);
    await expect(debugMatchedCountLabel).toContainText('(149)');
    await expect(benchMatchedCountLabel).toContainText('(149)');
    await expect(debugUnmatchedCountLabel).toContainText('(2)');
    await expect(benchUnmatchedCountLabel).toContainText('(2)');
    await unmatchButton.click();
    await expect(debugUnmatchedSelect).toHaveText('Module.layer1.0.BasicBlock.forward.0');
    await expect(debugMatchedCountLabel).toContainText('(135)');
    await expect(benchMatchedCountLabel).toContainText('(135)');
    await expect(debugUnmatchedCountLabel).toContainText('(16)');
    await expect(benchUnmatchedCountLabel).toContainText('(16)');
    await expect(mainArea).toHaveScreenshot('unmatchNodeWithDesendant.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 节点取消匹配测试（不勾选后代节点）
  test('unmatchNodeWithoutCheckDesendant', async ({ mainPage, matchPanel }) => {
    const { debugMatchedSelect, unmatchButton, debugUnmatchedSelect, unmatchDesendantCheckbox } = matchPanel;
    const { page, mainArea, debugGraph, benchGraph, getSelectOption } = mainPage;
    const searchStr = 'Module.layer1.1.conv1.Conv2d.forward.0';
    await debugMatchedSelect.locator('#rc_select_8').fill(searchStr);
    await getSelectOption(searchStr).click();
    const moduleNodeDisplayName = 'layer1.1.BasicBlock.forward.0layer1.1.BasicBlock.forward.0';
    await locatedAndExpandMatchedModuleNodeByName(moduleNodeDisplayName, page, debugGraph, benchGraph);
    await unmatchDesendantCheckbox.uncheck();
    await unmatchButton.click();
    await expect(debugUnmatchedSelect).toHaveText('Module.layer1.1.BasicBlock.forward.0');
    await expect(mainArea).toHaveScreenshot('unmatchNodeWithoutDesendant.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });

  // 节点匹配测试（勾选后代节点）
  test('matchNodeWithCheckDesendant', async ({ mainPage, matchPanel }) => {
    const {
      debugMatchedCountLabel,
      debugMatchedSelect,
      benchMatchedCountLabel,
      matchButton,
      debugUnmatchedCountLabel,
      benchUnmatchedCountLabel,
      unmatchButton,
    } = matchPanel;
    const { debugGraph, benchGraph, mainArea } = mainPage;
    await debugGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0').dblclick();
    const subNodeDisplayName = 'layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0';
    const benchSubNode = benchGraph.getByText(subNodeDisplayName);
    await expect(benchSubNode).toBeAttached();
    // 取消同步展开关联节点，否则会自动展开/收起父节点，对测试带来不便
    const syncSubnode = mainArea.getByRole('button', { name: 'subnode' });
    await syncSubnode.click();
    await debugGraph.getByText(subNodeDisplayName).dblclick();
    await benchSubNode.dblclick();
    await expect(debugMatchedCountLabel).toContainText('(133)');
    await expect(benchMatchedCountLabel).toContainText('(133)');
    await expect(debugUnmatchedCountLabel).toContainText('(18)');
    await expect(benchUnmatchedCountLabel).toContainText('(18)');
    await matchButton.click();
    await expect(debugMatchedSelect).toHaveText('Module.layer1.0.BasicBlock.forward.0');
    await expect(debugMatchedCountLabel).toContainText('(147)');
    await expect(benchMatchedCountLabel).toContainText('(147)');
    await expect(debugUnmatchedCountLabel).toContainText('(4)');
    await expect(benchUnmatchedCountLabel).toContainText('(4)');
    await expect(mainArea).toHaveScreenshot('matchNodeWithDesendant.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    // 重新取消匹配，方便后续测试
    await unmatchButton.click();
  });

  // 节点匹配测试（不勾选后代节点），通过下拉列表选择节点测试
  test('matchNodeWithoutCheckDesendant', async ({ mainPage, matchPanel }) => {
    const {
      debugUnmatchedSelect,
      debugUnmatchedUpIcon,
      benchUnmatchedSelect,
      benchUnmatchedDownIcon,
      matchDesendantCheckbox,
      matchButton,
      debugUnmatchedCountLabel,
      unmatchButton,
    } = matchPanel;
    const { debugGraph, mainArea, getSelectOption } = mainPage;
    await debugGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0').dblclick();
    await debugGraph.getByText('layer1.1.BasicBlock.forward.0layer1.1.BasicBlock.forward.0').click();
    await debugUnmatchedUpIcon.click();
    await expect(debugUnmatchedSelect).toHaveText('Module.layer1.0.BasicBlock.forward.0');
    await benchUnmatchedSelect.click();
    await getSelectOption('Module.layer1.0.relu.ReLU.forward.1').click();
    await benchUnmatchedDownIcon.click();
    await matchDesendantCheckbox.uncheck();
    await matchButton.click();
    await expect(debugUnmatchedCountLabel).toContainText('(16)');
    await debugGraph.getByText('layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0').dblclick();
    await expect(debugGraph.getByText('conv1.Co…conv1.Conv2d.forward')).toBeVisible();
    await expect(mainArea).toHaveScreenshot('matchNodeWithoutCheckDesendant.png', { maxDiffPixels: MAX_DIFF_PIXELS });
    // 回退匹配前状态
    await unmatchButton.click();
  });

  // 测试生成和应用匹配配置文件
  test('generateMatchConfigurationAndMatchByConfiguration', async ({ mainPage, matchPanel }) => {
    const {
      configurationSelect,
      configurationGenerateTooltip,
      configurationGenerateButton,
      matchButton,
      unmatchButton,
    } = matchPanel;
    const { page, debugGraph, benchGraph, mainArea, getSelectOption } = mainPage;
    await debugGraph.getByText('Module.layer1.Sequential.forward.0Module.layer1.Sequential.forward.0').dblclick();
    const subNodeDisplayName1 = 'layer1.0.BasicBlock.forward.0layer1.0.BasicBlock.forward.0';
    const subNodeDisplayName2 = 'layer1.1.BasicBlock.forward.0layer1.1.BasicBlock.forward.0';
    const debugSubNode1 = debugGraph.getByText(subNodeDisplayName1);
    const debugSubNode2 = debugGraph.getByText(subNodeDisplayName2);
    const benchSubNode1 = benchGraph.getByText(subNodeDisplayName1);
    const benchSubNode2 = benchGraph.getByText(subNodeDisplayName2);
    await debugSubNode1.click();
    await benchSubNode1.click();
    await matchButton.click();
    await debugSubNode2.click();
    await benchSubNode2.click();
    await matchButton.click();
    await configurationGenerateTooltip.hover();
    // 等待提示信息出现
    await page.waitForTimeout(1000);
    await expect(mainArea).toHaveScreenshot('generateMatchConfigurationTooltip.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await configurationGenerateButton.click();
    await debugSubNode1.click();
    await unmatchButton.click();
    await debugSubNode2.click();
    await unmatchButton.click();
    await configurationSelect.click();
    await getSelectOption(`${FILES.COMPARE_FILE}_0_0.vis.config`).click();
    await page.waitForResponse(
      (response) => response.url().includes('/updateHierarchyData') && response.status() === 200,
    );
    await debugSubNode1.dblclick();
    await debugSubNode2.dblclick();
    await expect(mainArea).toHaveScreenshot('matchNodesByConfiguration.png', { maxDiffPixels: MAX_DIFF_PIXELS });
  });
});

// 主题切换和语言切换侧边栏
test.describe('ThemeAndLanguageSiderTest', () => {
  setupBeforeTest(SIDER_TYPE.THEME);

  test('displayInDarkAndZhMode', async ({ mainPage, precisionFilertPanel, nodeSearchPanel, matchPanel }) => {
    const { precisionSiderButton, searchSiderButton, matchSiderButton, translationSiderButton } = mainPage;
    const { mainArea } = mainPage;
    // 切换中文模式
    await translationSiderButton.click();
    await precisionSiderButton.click();
    await expect(precisionFilertPanel.nodeCountLabel).toContainText('节点列表');
    await expect(mainArea).toHaveScreenshot('precisionFilterDisplayInDarkAndZhMode.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await searchSiderButton.click();
    await expect(mainArea).toHaveScreenshot('nodeSearchDisplayInDarkAndZhMode.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
    await matchSiderButton.click();
    await expect(mainArea).toHaveScreenshot('nodeMatchDisplayInDarkAndZhMode.png', {
      maxDiffPixels: MAX_DIFF_PIXELS,
    });
  });
});
