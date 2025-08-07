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
