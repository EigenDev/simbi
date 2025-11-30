from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress_bar(console: Optional[Console] = None):
    """
    create a progress object sized for the given console (or the current terminal).

    - console: optional rich.Console; if omitted a new Console() is used.
    - the bar_width=None and expand=True let rich allocate remaining terminal
      width to the progress bar so the bar visually fills the terminal line.
    """
    if console is None:
        console = Console()

    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TaskProgressColumn(),
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )
