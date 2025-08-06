"""Console script for protoplast."""

import typer
from rich.console import Console

from protoplast import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for protoplast."""
    console.print("Replace this message by putting your code into protoplast.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
