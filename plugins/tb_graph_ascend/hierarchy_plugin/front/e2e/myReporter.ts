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
import type { Reporter, FullConfig, Suite, TestCase, TestResult, FullResult } from '@playwright/test/reporter';

class MyReporter implements Reporter {
  constructor() {
    console.log('Starting');
  }

  onBegin(config: FullConfig, suite: Suite): void {
    console.log(`Starting all tests: ${suite.allTests().length}`);
  }

  onTestBegin(test: TestCase): void {
    console.log(`Starting test: ${test.parent.title} ${test.title}`);
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    console.log(`Finish the test:  ${test.parent.title}  ${test.title}  ${result.status}`);
  }

  onEnd(result: FullResult): void {
    console.log('Finished all tests');
  }
}
export default MyReporter;
