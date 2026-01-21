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