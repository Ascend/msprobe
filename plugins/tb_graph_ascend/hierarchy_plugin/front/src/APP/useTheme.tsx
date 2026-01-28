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
