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
