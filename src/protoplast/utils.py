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
    level_map = logging.getLevelNamesMapping()

    if env_level not in level_map:
        log_level = logging.INFO
        print(f"Warning: Invalid LOG_LEVEL '{env_level}' provided. Defaulting to INFO.", file=sys.stderr)
    else:
        log_level = level_map[env_level]

    logging.basicConfig(
        level=log_level,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Logging initialized. Current level is: {logging.getLevelName(log_level)}")
