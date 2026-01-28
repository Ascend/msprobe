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

import { CURRENT_PAGE } from '../../common/constant';

import Dashboard from './Dashboard';
import Visualization from './Visualization';
import useGlobalStore from '../../store/useGlobalStore';

const AppContent = () => {
  const currentPage = useGlobalStore((state) => state.currentPage);
  const showContent = () => {
    switch (currentPage) {
      case CURRENT_PAGE.DASHBOARD:
        return <Dashboard />;
      case CURRENT_PAGE.VISUALIZATION:
        return <Visualization />;
    }
  };
  return <div>{showContent()}</div>;
};

export default AppContent;
