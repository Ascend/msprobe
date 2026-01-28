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
import { create } from 'zustand';
import { CURRENT_PAGE, CURRENT_TAB } from '../common/constant';

interface GlobalStateType {
  currentTab: CURRENT_TAB;
  currentPage: CURRENT_PAGE;
  setCurrentTab: (tab: GlobalStateType['currentTab']) => void;
  setCurrentPage: (page: GlobalStateType['currentPage']) => void;
}

const useGlobalStore = create<GlobalStateType>()((set) => ({
  currentTab: CURRENT_TAB.PRECISION_TAB,
  currentPage: CURRENT_PAGE.DASHBOARD,
  setCurrentTab: (tab: GlobalStateType['currentTab']) => set({ currentTab: tab }),
  setCurrentPage: (page: GlobalStateType['currentPage']) => set({ currentPage: page }),
}));

export default useGlobalStore;
