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
}

interface FileOption {
  compareFileOption: Locator;
  singleFileOption: Locator;
  communicationFileOption: Locator;
  md5FileOption: Locator;
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

  constructor(page: Page) {
    this.page = page;
    this.panel = page.getByTestId('metaContentPanel');
    this.dirSelector = this.panel.getByTestId('runSelect');
    this.fileSelector = this.panel.getByTestId('tagSelect');
    this.stepSelector = this.panel.getByTestId('stepSelect');
    this.rankSelector = this.panel.getByTestId('rankSelect');
    this.microStepSelector = this.panel.getByTestId('microStepSelect');
    this.dirOptions = {
      compareDirOption: page.locator(`.ant-select-item-option[title='${DIRS.COMPARE_DIR}']`),
      singleDirOption: page.locator(`.ant-select-item-option[title='${DIRS.SINGLE_DIR}']`),
      communicationDirOption: page.locator(`.ant-select-item-option[title='${DIRS.COMMUNICATION_DIR}']`),
      md5DirOption: page.locator(`.ant-select-item-option[title='${DIRS.MD5_DIR}']`),
    };
    this.fileOptions = {
      compareFileOption: page.locator(`.ant-select-item-option[title='${FILES.COMPARE_FILE}']`),
      singleFileOption: page.locator(`.ant-select-item-option[title='${FILES.SINGLE_FILE}']`),
      communicationFileOption: page.locator(`.ant-select-item-option[title='${FILES.COMMUNICATION_FILE}']`),
      md5FileOption: page.locator(`.ant-select-item-option[title='${FILES.MD5_FILE}']`),
    };
  }

  getSelectOption = (title: string): Locator => {
    return this.page.locator(`.ant-select-item-option[title='${title}']`);
  };
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

  constructor(page: Page) {
    this.panel = page.getByTestId('precisionPanel');
  }
}

class MatchPanel {
  readonly panel: Locator;

  constructor(page: Page) {
    this.panel = page.getByTestId('matchPanel');
  }
}

class MainPage {
  readonly page: Page;
  readonly mainArea: Locator;
  readonly fileSiderButton: Locator;
  readonly accuracySiderButton: Locator;
  readonly matchSiderButton: Locator;
  readonly searchSiderButton: Locator;
  readonly conversionSiderButton: Locator;
  readonly themeSiderButton: Locator;
  readonly translationSiderButton: Locator;
  readonly metaContent: MetaContentPanel;
  readonly npuGraph: Locator;
  readonly benchGraph: Locator;
  readonly siderButtons: Record<SIDER_TYPE, Locator>;

  constructor(page: Page) {
    this.page = page;
    this.fileSiderButton = page.getByRole('button', { name: 'file' });
    this.accuracySiderButton = page.getByTestId('accuracyErrorFilter');
    this.matchSiderButton = page.getByTestId('matchSiderButton');
    this.searchSiderButton = page.getByTestId('searchSiderButton');
    this.conversionSiderButton = page.getByTestId('conversionSiderButton');
    this.themeSiderButton = page.getByTestId('themeSiderButton');
    this.translationSiderButton = page.getByRole('button', { name: 'translation' });
    this.siderButtons = {
      [SIDER_TYPE.FILE]: this.fileSiderButton,
      [SIDER_TYPE.SEARCH]: this.searchSiderButton,
      [SIDER_TYPE.PRECISION]: this.accuracySiderButton,
      [SIDER_TYPE.MATCH]: this.matchSiderButton,
      [SIDER_TYPE.THEME]: this.themeSiderButton,
      [SIDER_TYPE.LANGUAGE]: this.translationSiderButton,
    };
    this.metaContent = new MetaContentPanel(page);
    this.mainArea = page.locator('body');
    this.npuGraph = page.getByTestId('debugGraph');
    this.benchGraph = page.getByTestId('benchGraph');
  }

  async getBoundingBoxes(): Promise<{ npuArea: BoundingBox; benchArea: BoundingBox }> {
    const npuArea = await this.npuGraph.boundingBox();
    const benchArea = await this.benchGraph.boundingBox();
    if (!npuArea || !benchArea) {
      throw new Error('Test failed because the graph area was not rendered correctly.');
    }
    return { npuArea, benchArea };
  }
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
});
