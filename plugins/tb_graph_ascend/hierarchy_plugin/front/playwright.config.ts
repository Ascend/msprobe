/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
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
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60 * 1000,
  expect: {
    timeout: 5000,
  },
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: 1,
  reporter: [['html', { open: 'never' }], ['./e2e/myReporter.ts']],
  use: {
    /* Base URL to use in actions like `await page.goto('')`. */
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    headless: true,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        // 赋予粘贴板写入权限才能使用复制功能
        permissions: ['clipboard-write'],
      },
    },
  ],

  /* Run your local dev server before starting the tests */
  // 通过tensorboard指令拉起后端，需确保运行的环境能够执行该指令
  webServer: [
    {
      command: 'tensorboard --logdir ~/GUI_DATA',
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'npm run dev',
      url: 'http://127.0.0.1:3000',
      reuseExistingServer: !process.env.CI,
    },
  ],
});
