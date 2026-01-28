# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
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


