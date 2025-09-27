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


from urllib.parse import urlparse

import fsspec
import h5py


def open_fsspec(filename: str):
    parsed = urlparse(filename)
    scheme = parsed.scheme.lower()

    if scheme == "dnanexus":
        fs = fsspec.filesystem("dnanexus")
        file = fs.open(filename, mode="rb")
    else:
        # For local files or other supported fsspec schemes
        fs, path = fsspec.core.url_to_fs(filename)
        file = fs.open(path, mode="rb")

    return h5py.File(file, "r")
