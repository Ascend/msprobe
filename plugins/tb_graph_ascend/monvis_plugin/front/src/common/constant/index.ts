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

export const STEP_DIMENSION = 'step';
export const RANK_DIMENSION = 'rank';
export const MODULE_NAME_DIMENSION = 'module_name';

export const CONTINUOUS = 'continuous';
export const PIECEWISE = 'piecewise';

export const DIMENSIONS_OPTIONS = [
  { value: STEP_DIMENSION, label: 'Step' },
  { value: RANK_DIMENSION, label: 'Rank' },
  { value: MODULE_NAME_DIMENSION, label: 'Module Name' },
];

export const HEATMAP_TYPE = [
  { value: CONTINUOUS, label: '渐变模式' },
  { value: PIECEWISE, label: '分段模式' },
];

export const DIMENSIONS_AXIS_MAP = {
  [STEP_DIMENSION]: {
    x: 'Rank',
    y: 'Module Name',
  },
  [RANK_DIMENSION]: {
    x: 'Step',
    y: 'Module Name',
  },
  [MODULE_NAME_DIMENSION]: {
    x: 'Step',
    y: 'Rank',
  },
};

export const CLEAR_ICON = 'path://M10 10 H90 V90 H10 Z M25 25 L75 75 M75 25 L25 75';
