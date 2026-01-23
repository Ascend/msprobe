/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState, useEffect } from 'react';
import type { ThemeConfig } from 'antd';

type ThemeType = 'light' | 'dark';

interface ParentThemeResult {
  themeType: ThemeType;
  themeToken: NonNullable<ThemeConfig['token']>;
  toggleTheme: () => void;
}

const useTheme = (): ParentThemeResult => {
  const [themeType, setThemeType] = useState<ThemeType>('light');

  useEffect(() => {
    let parentBody: HTMLElement | null = null;
    try {
      parentBody = window?.parent?.document?.body;
    } catch (e) {
      console.warn('无法访问父页面 DOM，可能为跨域或非 iframe 环境');
    }
    if (!parentBody) {
      setThemeType('light');
      return;
    }

    const checkDarkMode = () => {
      const hasDarkClass = parentBody!.classList.contains('dark-mode');
      const newTheme = hasDarkClass ? 'dark' : 'light';
      setThemeType(newTheme);
    };

    checkDarkMode();

    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          checkDarkMode();
          break;
        }
      }
    });

    observer.observe(parentBody, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  const toggleTheme = () => {
    setThemeType((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // 根据 themeType 返回 token
  const themeToken =
    themeType === 'dark'
      ? {
          colorBgContainer: '#1a1a1a',
          colorBgLayout: '#0f0f0f',
          colorListHover: '#1d1d1d',
          colorListSelected: '#15325b',
          colorListSelectedHover: '#15417e',
          colorPanelBorder: '#ffffff',
          colorDebugTableRowBg: '#0f0f0f',
          colorBenchTableRowBg: '#2b2b2b',
          colorGroupByBorder: '#7b7b7b',
          colorBlockBorder: 'rgba(255, 255, 255, 0.25)',
        }
      : {
          colorBgContainer: '#ffffff',
          colorBgLayout: '#ffffff',
          colorListHover: '#fafafa',
          colorListSelected: '#e6f4ff',
          colorListSelectedHover: '#bae0ff',
          colorPanelBorder: '#c5c3c3ff',
          colorDebugTableRowBg: '#ffffff',
          colorBenchTableRowBg: '#f5f5f5',
          colorGroupByBorder: '#bfbfbf',
          colorBlockBorder: 'rgba(0, 0, 0, 0.15)',
        };

  return { themeType, themeToken, toggleTheme };
};

export default useTheme;
