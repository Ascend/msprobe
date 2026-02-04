/* -------------------------------------------------------------------------
 Copyright (c) 2026, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/
import { test as baseTest, Locator, Page } from '@playwright/test';
import { DIRS, FILES, SIDER_TYPE } from './constants';

interface AllPages {
  mainPage: MainPage;
  metaContentPanel: MetaContentPanel;
  nodeSearchPanel: NodeSearchPanel;
  precisionFilertPanel: PrecisionFilterPanel;
  matchPanel: MatchPanel;
}

interface DirOption {
  compareDirOption: Locator;
  singleDirOption: Locator;
  communicationDirOption: Locator;
  md5DirOption: Locator;
  overflowDirOption: Locator;
}

interface FileOption {
  compareFileOption: Locator;
  singleFileOption: Locator;
  communicationFileOption: Locator;
  md5FileOption: Locator;
  overflowFileOption: Locator;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

class MetaContentPanel {
  readonly page: Page;
  readonly panel: Locator;
  readonly dirSelector: Locator;
  readonly fileSelector: Locator;
  readonly stepSelector: Locator;
  readonly rankSelector: Locator;
  readonly microStepSelector: Locator;
  readonly dirOptions: DirOption;
  readonly fileOptions: FileOption;
  readonly mainPage: MainPage;

  constructor(page: Page) {
    this.page = page;
    this.panel = page.getByTestId('metaContentPanel');
    this.dirSelector = this.panel.getByTestId('runSelect');
    this.fileSelector = this.panel.getByTestId('tagSelect');
    this.stepSelector = this.panel.getByTestId('stepSelect');
    this.rankSelector = this.panel.getByTestId('rankSelect');
    this.microStepSelector = this.panel.getByTestId('microStepSelect');
    this.mainPage = new MainPage(this.page);
    this.dirOptions = {
      compareDirOption: this.mainPage.getSelectOption(DIRS.COMPARE_DIR),
      singleDirOption: this.mainPage.getSelectOption(DIRS.SINGLE_DIR),
      communicationDirOption: this.mainPage.getSelectOption(DIRS.COMMUNICATION_DIR),
      md5DirOption: this.mainPage.getSelectOption(DIRS.MD5_DIR),
      overflowDirOption: this.mainPage.getSelectOption(DIRS.OVERFLOW_DIR),
    };
    this.fileOptions = {
      compareFileOption: this.mainPage.getSelectOption(FILES.COMPARE_FILE),
      singleFileOption: this.mainPage.getSelectOption(FILES.SINGLE_FILE),
      communicationFileOption: this.mainPage.getSelectOption(FILES.COMMUNICATION_FILE),
      md5FileOption: this.mainPage.getSelectOption(FILES.MD5_FILE),
      overflowFileOption: this.mainPage.getSelectOption(FILES.OVERFLOW_FILE),
    };
  }
}

class NodeSearchPanel {
  readonly panel: Locator;
  readonly debugRadio: Locator;
  readonly benchRadio: Locator;
  readonly nodeSearch: Locator;
  readonly nodeCountLabel: Locator;
  readonly clearButton: Locator;
  readonly upIcon: Locator;
  readonly downIcon: Locator;

  constructor(page: Page) {
    this.panel = page.getByTestId('searchPanel');
    this.debugRadio = this.panel.getByRole('radio', { name: 'Debug' });
    this.benchRadio = this.panel.getByRole('radio', { name: 'Bench' });
    this.nodeSearch = this.panel.getByRole('searchbox');
    this.nodeCountLabel = this.panel.getByTestId('nodeCountLabel');
    this.clearButton = this.panel.getByRole('button', { name: 'close-circle' });
    this.upIcon = this.panel.getByRole('img', { name: 'up' });
    this.downIcon = this.panel.getByRole('img', { name: 'down' });
  }
}

class PrecisionFilterPanel {
  readonly panel: Locator;
  readonly errorTab: Locator;
  readonly overflowTab: Locator;
  readonly precisionErrorTooltip: Locator;
  readonly precisionErrorCheckBoxes: {
    pass: Locator;
    warning: Locator;
    error: Locator;
    unmatched: Locator;
  };
  readonly overflowCheckBoxes: {
    medium: Locator;
    high: Locator;
    ciritical: Locator;
  };
  readonly nodeCountLabel: Locator;

  constructor(page: Page) {
    this.panel = page.getByTestId('precisionPanel');
    const precisionSemented = this.panel.getByTestId('precisionSemented');
    this.errorTab = precisionSemented.getByText('Accuracy Error');
    this.overflowTab = precisionSemented.getByText('Overflow');
    this.precisionErrorTooltip = this.panel.getByTestId('precisionErrorTooltip');
    this.precisionErrorCheckBoxes = {
      pass: this.panel.getByRole('checkbox', { name: 'Pass' }),
      warning: this.panel.getByRole('checkbox', { name: 'Warning' }),
      error: this.panel.getByRole('checkbox', { name: 'Error' }),
      unmatched: this.panel.getByRole('checkbox', { name: 'Unmatched' }),
    };
    this.overflowCheckBoxes = {
      medium: this.panel.getByRole('checkbox', { name: 'Medium' }),
      high: this.panel.getByRole('checkbox', { name: 'High' }),
      ciritical: this.panel.getByRole('checkbox', { name: 'Critical' }),
    };
    this.nodeCountLabel = this.panel.getByTestId('nodeCountLabel');
  }
}

class MatchPanel {
  readonly panel: Locator;
  readonly configurationSelect: Locator;
  readonly configurationGenerateButton: Locator;
  readonly configurationGenerateTooltip: Locator;
  readonly debugUnmatchedCountLabel: Locator;
  readonly debugUnmatchedSelect: Locator;
  readonly debugUnmatchedUpIcon: Locator;
  readonly debugUnmatchedDownIcon: Locator;
  readonly benchUnmatchedCountLabel: Locator;
  readonly benchUnmatchedSelect: Locator;
  readonly benchUnmatchedUpIcon: Locator;
  readonly benchUnmatchedDownIcon: Locator;
  readonly debugMatchedCountLabel: Locator;
  readonly debugMatchedSelect: Locator;
  readonly debugMatchedUpIcon: Locator;
  readonly debugMatchedDownIcon: Locator;
  readonly benchMatchedCountLabel: Locator;
  readonly benchMatchedSelect: Locator;
  readonly benchMatchedUpIcon: Locator;
  readonly benchMatchedDownIcon: Locator;
  readonly matchButton: Locator;
  readonly unmatchButton: Locator;
  readonly matchDesendantCheckbox: Locator;
  readonly unmatchDesendantCheckbox: Locator;

  constructor(page: Page) {
    this.panel = page.getByTestId('matchPanel');
    this.configurationSelect = this.panel.getByTestId('configurationSelect');
    this.configurationGenerateButton = this.panel.getByTestId('configurationGenerateButton');
    this.configurationGenerateTooltip = this.panel.getByTestId('configurationGenerateTooltip');
    this.debugUnmatchedCountLabel = this.panel.getByTestId('debugUnmatchedCount');
    this.debugUnmatchedSelect = this.panel.getByTestId('debugUnmatchedSelect');
    this.debugUnmatchedUpIcon = this.panel.getByTestId('debugUnmatchedUp');
    this.debugUnmatchedDownIcon = this.panel.getByTestId('debugUnmatchedDown');
    this.benchUnmatchedCountLabel = this.panel.getByTestId('benchUnmatchedCount');
    this.benchUnmatchedSelect = this.panel.getByTestId('benchUnmatchedSelect');
    this.benchUnmatchedUpIcon = this.panel.getByTestId('benchUnmatchedUp');
    this.benchUnmatchedDownIcon = this.panel.getByTestId('benchUnmatchedDown');
    this.debugMatchedCountLabel = this.panel.getByTestId('debugMatchedCount');
    this.debugMatchedSelect = this.panel.getByTestId('debugMatchedSelect');
    this.debugMatchedUpIcon = this.panel.getByTestId('debugMatchedUp');
    this.debugMatchedDownIcon = this.panel.getByTestId('debugMatchedDown');
    this.benchMatchedCountLabel = this.panel.getByTestId('benchMatchedCount');
    this.benchMatchedSelect = this.panel.getByTestId('benchMatchedSelect');
    this.benchMatchedUpIcon = this.panel.getByTestId('benchMatchedUp');
    this.benchMatchedDownIcon = this.panel.getByTestId('benchMatchedDown');
    this.matchButton = this.panel.getByTestId('matchButton');
    this.unmatchButton = this.panel.getByTestId('unmatchButton');
    this.matchDesendantCheckbox = this.panel.getByTestId('matchDesendantCheckbox');
    this.unmatchDesendantCheckbox = this.panel.getByTestId('unmatchDesendantCheckbox');
  }
}

class MainPage {
  readonly page: Page;
  readonly mainArea: Locator;
  readonly fileSiderButton: Locator;
  readonly precisionSiderButton: Locator;
  readonly matchSiderButton: Locator;
  readonly searchSiderButton: Locator;
  readonly conversionSiderButton: Locator;
  readonly themeSiderButton: Locator;
  readonly translationSiderButton: Locator;
  readonly debugGraph: Locator;
  readonly benchGraph: Locator;
  readonly siderButtons: Record<SIDER_TYPE, Locator>;
  readonly syncCheckBox: Locator;
  readonly splitter: Locator;
  readonly npuMinimap: Locator;
  readonly benchMinimap: Locator;

  constructor(page: Page) {
    this.page = page;
    this.fileSiderButton = page.getByRole('button', { name: 'file' });
    this.precisionSiderButton = page.getByTestId('precisionSiderButton');
    this.matchSiderButton = page.getByTestId('matchSiderButton');
    this.searchSiderButton = page.getByTestId('searchSiderButton');
    this.conversionSiderButton = page.getByTestId('conversionSiderButton');
    this.themeSiderButton = page.getByTestId('themeSiderButton');
    this.translationSiderButton = page.getByRole('button', { name: 'translation' });
    this.siderButtons = {
      [SIDER_TYPE.FILE]: this.fileSiderButton,
      [SIDER_TYPE.SEARCH]: this.searchSiderButton,
      [SIDER_TYPE.PRECISION]: this.precisionSiderButton,
      [SIDER_TYPE.MATCH]: this.matchSiderButton,
      [SIDER_TYPE.THEME]: this.themeSiderButton,
      [SIDER_TYPE.LANGUAGE]: this.translationSiderButton,
    };
    this.mainArea = page.locator('body');
    this.debugGraph = page.getByTestId('debugGraph');
    this.benchGraph = page.getByTestId('benchGraph');
    this.syncCheckBox = page.getByRole('button', { name: 'subnode' });
    this.splitter = page.locator(
      '._boardContentWrapper_uns83_17 > .ant-splitter > .ant-splitter-bar > .ant-splitter-bar-dragger',
    );
    this.npuMinimap = this.debugGraph.locator('#minimap');
    this.benchMinimap = this.benchGraph.locator('#minimap');
  }

  getBoundingBoxes = async (): Promise<{ npuArea: BoundingBox; benchArea: BoundingBox }> => {
    const npuArea = await this.debugGraph.boundingBox();
    const benchArea = await this.benchGraph.boundingBox();
    if (!npuArea || !benchArea) {
      throw new Error('Test failed because the graph area was not rendered correctly.');
    }
    return { npuArea, benchArea };
  };

  getSelectOption = (value: string): Locator => {
    return this.page.locator(`.ant-select-item-option[title='${value}']`);
  };
}

export const test = baseTest.extend<AllPages>({
  mainPage: async ({ page }, use) => {
    const mainPage = new MainPage(page);
    await use(mainPage);
  },
  metaContentPanel: async ({ page }, use) => {
    const metaContentPanel = new MetaContentPanel(page);
    await use(metaContentPanel);
  },
  nodeSearchPanel: async ({ page }, use) => {
    const nodeSearchPanel = new NodeSearchPanel(page);
    await use(nodeSearchPanel);
  },
  precisionFilertPanel: async ({ page }, use) => {
    const precisionFilterPanel = new PrecisionFilterPanel(page);
    await use(precisionFilterPanel);
  },
  matchPanel: async ({ page }, use) => {
    const matchPanel = new MatchPanel(page);
    await use(matchPanel);
  },
});
