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
-----------------------------------------------------------------------------*/

export enum SIDER_TYPE {
  FILE,
  SEARCH,
  PRECISION,
  MATCH,
  THEME,
  LANGUAGE,
}

export const enum DIRS {
  COMPARE_DIR = 'Compare',
  SINGLE_DIR = 'Single',
  MD5_DIR = 'Md5',
  COMMUNICATION_DIR = 'Communication',
  OVERFLOW_DIR = 'Overflow',
}

export const enum FILES {
  COMPARE_FILE = 'compare_20250902171456',
  SINGLE_FILE = 'build_20251227174822',
  MD5_FILE = 'compare_20260104171250',
  COMMUNICATION_FILE = 'build_collective',
  OVERFLOW_FILE = 'build_20260110160141',
}

export const MAX_DIFF_PIXELS = 100;
