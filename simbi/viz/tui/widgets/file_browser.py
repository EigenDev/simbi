# =============================================================================
# simbi/viz/tui/widgets/file_browser.py
#
# directory tree + checkpoint file selection widget.
# users navigate directories and select checkpoint files for plotting.
# supports single file (snapshot), multiple files (overlay), or
# directory (animation/time series).
# =============================================================================
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DirectoryTree, Label, Static


class _CheckpointTree(DirectoryTree):
    """directory tree filtered to show only checkpoint files and directories."""

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        return sorted(
            p
            for p in paths
            if p.is_dir()
            or (p.suffix in (".h5", ".hdf5") and "chkpt" in p.name)
        )


class FileBrowser(Static):
    """file browser for selecting checkpoint files."""

    class FilesSelected(Message):
        """emitted when the user confirms file selection."""

        def __init__(self, files: list[Path]) -> None:
            super().__init__()
            self.files = files

    DEFAULT_CSS = """
    FileBrowser {
        height: 100%;
        width: 100%;
    }

    FileBrowser #file-tree {
        height: 1fr;
        min-height: 10;
        border: solid $primary;
    }

    FileBrowser #selection-info {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }

    FileBrowser #file-buttons {
        height: 3;
        margin-top: 1;
    }

    FileBrowser #file-buttons Button {
        min-width: 12;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        initial_path: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._root = initial_path or Path.cwd()
        if self._root.is_file():
            self._root = self._root.parent
        self._selected: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Label("Files", classes="section-title")
        with Vertical():
            yield _CheckpointTree(self._root, id="file-tree")
            yield Static("no files selected", id="selection-info")
            with Horizontal(id="file-buttons"):
                yield Button("Select Dir", id="select-dir", variant="default")
                yield Button("Clear", id="clear-selection", variant="error")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        path = Path(event.path)
        if path in self._selected:
            self._selected.remove(path)
        else:
            self._selected.append(path)
        self._update_info()
        self.post_message(self.FilesSelected(list(self._selected)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select-dir":
            tree = self.query_one("#file-tree", _CheckpointTree)
            dir_path = tree.path
            # glob all checkpoint files in the selected directory
            files = sorted(
                p
                for p in dir_path.iterdir()
                if p.suffix in (".h5", ".hdf5") and "chkpt" in p.name
            )
            self._selected = files
            self._update_info()
            self.post_message(self.FilesSelected(list(self._selected)))
        elif event.button.id == "clear-selection":
            self._selected.clear()
            self._update_info()
            self.post_message(self.FilesSelected([]))

    def _update_info(self) -> None:
        info = self.query_one("#selection-info", Static)
        n = len(self._selected)
        if n == 0:
            info.update("no files selected")
        elif n == 1:
            info.update(f"1 file: {self._selected[0].name}")
        else:
            info.update(f"{n} files selected (first: {self._selected[0].name})")

    @property
    def selected_files(self) -> list[Path]:
        return list(self._selected)
