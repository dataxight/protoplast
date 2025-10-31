#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import logging
import os
from urllib.parse import urlparse

import fsspec
import h5py
import s3fs
import torch
from fsspec.implementations.cached import WholeFileCacheFileSystem

logger = logging.getLogger(__name__)


IS_PROTORAY = bool(int(os.getenv("IS_PROTORAY", "0")))

PROTORAY_CACHE_FS = os.environ.get("PROTORAY_CACHE_FS", "wholefile")
PROTORAY_CACHE_DIR = os.environ.get("PROTORAY_CACHE_DIR", "/mnt/cache/1/fsspec_cache")

CACHED_FS = os.environ.get("CACHED_FS", "readahead")
CACHED_DIR = os.environ.get("CACHED_DIR", "/tmp/fsspec_cache")


def _get_protoray_fs(fs):
    logger.debug("Using Protoray Caching config")
    if torch.cuda.is_available():
        if PROTORAY_CACHE_FS == "wholefile":
            fs = WholeFileCacheFileSystem(fs=fs, cache_storage=PROTORAY_CACHE_DIR)
        else:
            raise ValueError(f"Unsupported CACHED_FS value: {PROTORAY_CACHE_FS}")
    else:
        logger.debug("Not using caching for S3 no GPU detected")
    return fs


def _get_cached_fs(fs):
    logger.debug("Using Caching config")
    if CACHED_FS == "wholefile":
        fs = WholeFileCacheFileSystem(fs=fs, cache_storage=CACHED_DIR)
    else:
        raise ValueError(f"Unsupported CACHED_FS value: {CACHED_FS}")
    return fs


def get_fsspec(filename: str, mode="rb"):
    parsed = urlparse(filename)
    scheme = parsed.scheme.lower()
    if scheme == "dnanexus":
        fs = fsspec.filesystem("dnanexus")
        file = fs.open(filename, mode=mode)
    elif scheme == "s3":
        fs = s3fs.S3FileSystem(anon=False)
        full_s3_path = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
        if IS_PROTORAY:
            fs = _get_protoray_fs(fs)
        else:
            fs = _get_cached_fs(fs)
        file = fs.open(full_s3_path, mode=mode)
    else:
        # For local files or other supported fsspec schemes
        fs, path = fsspec.core.url_to_fs(filename)
        file = fs.open(path, mode=mode)
    return file


def open_fsspec(filename: str):
    file = get_fsspec(filename)
    return h5py.File(file, "r")
