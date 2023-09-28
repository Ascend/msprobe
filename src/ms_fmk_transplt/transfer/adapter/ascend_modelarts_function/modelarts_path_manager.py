#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import atexit
import os
import shutil
from enum import Enum, auto
import pathlib
from collections import namedtuple
from utils import trans_utils as utils

try:
    import moxing as mox
except ImportError:
    mox = None

from .path_mapping_config import PATH_MAPPING_CONFIG


class ModelArtsPathManager:
    class PathType(Enum):
        UNDEFINED = auto()
        DIR = auto()
        FILE = auto()

    CACHE_DIR = '/cache/modelarts_cache'

    PathPair = namedtuple('PathPair', ['type', 'local_path', 'obs_path'])

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self._is_run_on_modelarts = (mox is not None)
        if not self._is_run_on_modelarts:
            return

        self.project_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        self._input_path_mapping = self._parse_input_path_mapping()
        self._output_path_mapping = self._parse_output_path_mapping()

        self._download_before_training()
        atexit.register(self._upload_after_training)

    @staticmethod
    def log_info(msg):
        print('[INFO] ' + str(msg))

    @staticmethod
    def log_warning(msg):
        print('[WARNING] ' + str(msg))

    @staticmethod
    def log_error(msg):
        print('[ERROR] ' + str(msg))

    @staticmethod
    def _check_obs_path_size_valid(obs_path, local_path):
        local_path_free_size = shutil.disk_usage(local_path).free
        obs_path_total_size = mox.file.get_size(obs_path, recursive=True)
        if obs_path_total_size >= local_path_free_size:
            raise ValueError('The obs path is too large, and the remaining disk space is not enough.')

    def get_path(self, *args, **kwargs):
        if len(args) > 0:
            return self._get_path(args[0])
        else:
            return self._get_path(list(kwargs.values())[0])

    def _get_path(self, local_path):
        if not self._is_run_on_modelarts:
            return local_path

        if isinstance(local_path, pathlib.Path):
            local_path = str(local_path)
        if not isinstance(local_path, str):
            self.log_warning(f'Unsupported path argument type {type(local_path)}')
            return local_path

        if not os.path.isabs(local_path):
            return os.path.realpath(os.path.join(self.project_path, local_path))

        path_mapping = {**self._input_path_mapping, **self._output_path_mapping}
        for key, value in path_mapping.items():
            if value.type == ModelArtsPathManager.PathType.FILE and local_path == key:
                return value.local_path
            if value.type == ModelArtsPathManager.PathType.DIR:
                local_path = os.path.realpath(local_path)
                if local_path == key or local_path.startswith(key + os.path.sep):
                    mapping_path = os.path.realpath(os.path.join(value.local_path, os.path.relpath(local_path, key)))
                    return mapping_path

        return local_path

    def _parse_input_path_mapping(self):
        input_path_mapping = {}
        for local_path, obs_path in PATH_MAPPING_CONFIG.get('input').items():
            if not os.path.isabs(local_path):
                self.log_warning('Relative path cannot be input path.')
                continue
            if not mox.file.exists(obs_path):
                self.log_warning(f'Invalid obs/s3 path {obs_path}, skip.')
                continue

            local_path = os.path.realpath(local_path)
            if mox.file.is_directory(obs_path):
                path_type = ModelArtsPathManager.PathType.DIR
            else:
                path_type = ModelArtsPathManager.PathType.FILE
            cache_path = os.path.realpath(os.path.join(self.CACHE_DIR, local_path.lstrip('/')))
            input_path_mapping.update(
                {local_path: ModelArtsPathManager.PathPair(path_type, cache_path, obs_path)})

        return input_path_mapping

    def _parse_output_path_mapping(self):
        output_path_mapping = {}
        for local_path, obs_path in PATH_MAPPING_CONFIG.get('output').items():
            if not mox.file.exists(obs_path):
                path_type = ModelArtsPathManager.PathType.UNDEFINED
            elif mox.file.is_directory(obs_path):
                path_type = ModelArtsPathManager.PathType.DIR
            else:
                path_type = ModelArtsPathManager.PathType.FILE

            if not os.path.isabs(local_path):
                output_path_mapping.update(
                    {local_path: ModelArtsPathManager.PathPair(
                        path_type, os.path.join(self.project_path, local_path), obs_path)})
            else:
                cache_path = os.path.realpath(os.path.join(self.CACHE_DIR, local_path.lstrip('/')))
                if path_type == ModelArtsPathManager.PathType.DIR:
                    utils.make_dir_safety(cache_path)
                else:
                    utils.make_dir_safety(os.path.dirname(cache_path))
                output_path_mapping.update(
                    {os.path.realpath(local_path): ModelArtsPathManager.PathPair(path_type, cache_path, obs_path)})

        return output_path_mapping

    def _download_before_training(self):
        for _, (path_type, local_path, obs_path) in self._input_path_mapping.items():
            if path_type == ModelArtsPathManager.PathType.DIR:
                utils.make_dir_safety(local_path)
                self.log_info(f'Download directory from {obs_path} to {local_path} ...')
                self._check_obs_path_size_valid(obs_path, local_path)
                mox.file.copy_parallel(obs_path, local_path)
            else:
                utils.make_dir_safety(os.path.dirname(local_path))
                self.log_info(f'Download file from {obs_path} to {local_path} ...')
                self._check_obs_path_size_valid(obs_path, local_path)
                mox.file.copy(obs_path, local_path)
            self.log_info('Done.')

    def _upload_after_training(self):
        if not self._is_run_on_modelarts:
            return

        for _, path_pair in self._output_path_mapping.items():
            local_path, obs_path = path_pair.local_path, path_pair.obs_path
            if not os.path.exists(local_path):
                self.log_warning(f'Path {local_path} does not exist.')
                continue
            if os.path.isdir(local_path):
                self.log_info(f'Upload directory from {local_path} to {obs_path} ...')
                mox.file.copy_parallel(local_path, obs_path)
            else:
                self.log_info(f'Upload file from {local_path} to {obs_path} ...')
                mox.file.copy(local_path, obs_path)
            self.log_info('Done.')
