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

import { CURRENT_TAB } from '../../../../common/constant';
import useGlobalStore from '../../../../store/useGlobalStore';
import MatchSider from './MatchSider';
import PrecisionSider from './PrecisionSider';
import SearchSider from './SearchSider';

const BoardSider = () => {
  const currentTab = useGlobalStore((state) => state.currentTab);

  const showBoardSider = () => {
    switch (currentTab) {
      case CURRENT_TAB.PRECISION_TAB:
        return <PrecisionSider />;
      case CURRENT_TAB.MATCH_TAB:
        return <MatchSider />;
      case CURRENT_TAB.SEARCH_TAB:
        return <SearchSider />;
    }
  };

  return <div>{showBoardSider()}</div>;
};

export default BoardSider;
