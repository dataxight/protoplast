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
import sys
from typing import Any
from urllib.parse import urlparse

from daft import DataType
from daft.expressions import Expression, ExpressionVisitor


class ExpressionVisitorWithRequiredColumns(ExpressionVisitor[None]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_columns: set[str] = set()

    def get_required_columns(self, expr: Expression | None) -> list[str]:
        if expr is None:
            return []

        self.visit(expr)
        required_columns = list(self.required_columns)
        self.required_columns.clear()
        return required_columns

    def visit_col(self, name: str) -> None:
        self.required_columns.add(name)

    def visit_lit(self, value: Any) -> None:
        pass

    def visit_alias(self, expr: Expression, alias: str) -> None:
        self.visit(expr)

    def visit_cast(self, expr: Expression, dtype: DataType) -> None:
        self.visit(expr)

    def visit_function(self, name: str, args: list[Expression]) -> None:
        for arg in args:
            self.visit(arg)


def setup_console_logging():
    """
    Configures the root logger to output to the console (stderr)
    based on the LOG_LEVEL environment variable.
    """
    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    if sys.version_info >= (3, 11):
        level_map = logging.getLevelNamesMapping()
        if env_level not in level_map:
            log_level = logging.INFO
            print(f"Warning: Invalid LOG_LEVEL '{env_level}' provided. Defaulting to INFO.", file=sys.stderr)
        else:
            log_level = level_map[env_level]
    else:
        log_level = logging.getLevelName(env_level)
        if not isinstance(log_level, int):
            # The level was invalid, so log_level is still a string.
            print(f"Warning: Invalid LOG_LEVEL '{env_level}' provided. Defaulting to INFO.", file=sys.stderr)
            log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Logging initialized. Current level is: {logging.getLevelName(log_level)}")


def resolve_path_or_url(path_or_url: str | None) -> str | None:
    """
    Returns the absolute path for local files,
    otherwise returns the original URL for remote resources.

    Args:
        path_or_url (str): The input path or URL string.

    Returns:
        str: The resolved absolute path or the original URL/URI.
    """
    if path_or_url is None:
        return None
    parsed = urlparse(path_or_url)
    # Handle the special case where a scheme exists but netloc doesn't (e.g., 's3:data/file.txt')
    # If the scheme is NOT 'file' and it's present, treat it as remote.
    if parsed.scheme and parsed.scheme.lower() != "file":
        return path_or_url
    else:
        return os.path.abspath(path_or_url)
