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
