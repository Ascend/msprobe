# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import sqlite3
from tensorboard.util import tb_logging
from ..common.utils import Utils
logger = tb_logging.get_logger()

class DBConnection:
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self._initialize_db_connection()

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.conn is not None
    
    def _initialize_db_connection(self) -> None:
        """Initialize database connection."""
        try:
            # 目录安全校验
            directory = str(os.path.dirname(self.db_path))
            success, error = Utils.safe_check_load_file_path(directory, True)
            if not success:
                raise PermissionError(error)
            # 文件安全校验
            success, error = Utils.safe_check_load_file_path(self.db_path)
            if not success:
                raise PermissionError(error)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None


