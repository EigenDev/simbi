# Components developer guide

This file documents the expected contract between visualization components and
the `Figure` / `FigureFormatter` orchestration layers.

Goal
----
Provide a short, authoritative specification so component authors know:

- what to return from `render()` (the `RenderResult` contract),
- what metadata keys the `FigureFormatter` recognizes,
- which semantic artist keys are expected,
- how to migrate legacy components returning plain `dict`/`list`,
- a minimal example and a migration checklist.

Design principles
-----------------
- single responsibility: components create artists and are responsible for
  updating them in-place. layout and presentation decisions belong to the
  `FigureFormatter`.
- explicit contract: components SHOULD return `RenderResult`. legacy returns
  (plain dicts) are tolerated but migration is encouraged.
- metadata is advisory: formatter may ignore unknown keys. metadata exists to
  communicate intent, not to perform side effects.

RenderResult (canonical return)
-------------------------------
The `RenderResult` is the canonical return value for `Component.render()`.

Contents
- `artists` (dict[str, object]) — mapping of semantic keys to matplotlib
  artist objects (or other renderables).
- `metadata` (Optional[dict[str, object]]) — optional hints for the formatter.

Rationale: this typed structure makes it easy for `Figure` to normalize output,
and for `FigureFormatter` to decide whether to create colorbars, legends, and
axis labels.

Example (conceptual)
```/dev/null/example_renderresult.md#L1-14
RenderResult(
    artists={
        "mesh": quadmesh,            # pcolormesh / QuadMesh
        "quiver": quiver_obj,        # optional vector overlay
        "refs": [vline1, vline2],    # auxiliary artists
    },
    metadata={
        "mappable": quadmesh,        # explicit mappable for colorbar
        "label": "$\rho$",          # preferred label for colorbar or legend
        "preferred_cmap": "viridis", # advisory colormap
        "color_range": {"min": 0.0, "max": 10.0},
    },
)
```

Semantic artist keys
--------------------
Use these conventional keys when returning `artists`. The formatter and other
utilities may look for them.

- `"mesh"`: QuadMesh / pcolormesh (ScalarMappable) — typically used for 2D fields
- `"collection"`: PolyCollection / PatchCollection — alternative mappable for polygons
- `"line"`: Line2D single
- `"lines"`: list of Line2D objects
- `"quiver"`: Quiver object
- `"streamplot"`: the streamplot return (dictionary or specialized object)
- `"refs"`, `"vlines"`, `"annotations"`: auxiliary artists belonging to the component

Metadata keys (recommended)
---------------------------
These keys are recognized by the `FigureFormatter` and should be used by
components to communicate display intentions.

- `mappable`:
  - type: matplotlib ScalarMappable-like object (e.g., QuadMesh or PolyCollection)
  - use: direct reference for `fig.colorbar(...)` to create the colorbar
- `label`:
  - type: str
  - use: preferred descriptive label (used for colorbar label, ylabel, legend entries)
- `is_line`:
  - type: bool
  - use: explicitly mark a component as line-like; used to decide legend presence
- `is_vector`:
  - type: bool
  - use: mark vector visuals (quiver/stream) to avoid colorbar/legend confusion
- `preferred_cmap`:
  - type: str or Colormap
  - use: suggestion for colormap used when component creates its mappable
- `color_range`:
  - type: dict with keys `min` and `max` or a `ColorRange` object
  - use: explicit vmin/vmax to stabilize color scaling across frames

Notes:
- metadata is advisory. The formatter tries to respect keys but will fall back to
  conservative defaults if keys are missing/invalid.
- prefer including `mappable` in metadata when your component creates a mesh or
  collection intended for a colorbar. this avoids guesswork.

Formatter behavior (summary)
---------------------------
The `FigureFormatter`:
- collects normalized `(artists, metadata)` tuples from components,
- chooses the first available `mappable` (from `mappable` metadata or `'mesh'`/`'collection'` artist keys) to make a colorbar,
- will show a legend only when line-like artists are present (or `label`/`is_line` metadata is set),
- uses `FieldData.axis_names` and `metadata.label` to derive axis labels when possible,
- places colorbars differently for Cartesian vs. Polar axes (handled internally).

Minimal component template
--------------------------
This template shows the minimal responsibilities of a component. It assumes the
`Component` protocol from `components/interface.py` is used.

```/dev/null/component_template.py#L1-80
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from simbi.viz.components.interface import Component, ComponentProps
from simbi.viz.types import RenderResult, FieldData
from simbi.viz.config import StyleConfig

class MyLineComponent(Component):
    def __init__(self, props: ComponentProps):
        self.props = props
        self._line: Line2D | None = None
        self._initialized = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    def update(self, props: ComponentProps) -> None:
        self.props = props

    def render(self, data: FieldData, style: StyleConfig) -> RenderResult:
        if not self._initialized:
            raise RuntimeError("component not initialized")
        # create or update line artist in-place
        if self._line is None:
            self._line = self.ax.plot(data.domain[0], data.values, label=data.name)[0]
        else:
            self._line.set_data(data.domain[0], data.values)
        return RenderResult(artists={"line": self._line}, metadata={"label": data.name, "is_line": True})
```

Migration checklist (legacy -> RenderResult)
-------------------------------------------
If a component currently returns a plain `dict` or `list`, migrate it with
these steps:

1. Replace the loose return with `RenderResult(artists=..., metadata=...)`.
2. Populate `artists` with semantic keys: `"mesh"`, `"collection"`, `"line"`, etc.
3. Add `metadata["mappable"]` when your component has a mappable intended for a colorbar.
4. If your component represents a dataset conceptually single-valued, set `metadata["label"]`.
5. If your component draws lines, set `metadata["is_line"] = True`.
6. If your component draws vectors (quiver/stream), set `metadata["is_vector"] = True`.
7. Run `Figure.render()` and visually verify: legend presence, colorbar, labels.
8. Add a small unit test asserting that `component.render(...)` returns a `RenderResult`
   and that `artists` contains expected keys.

Testing suggestions
-------------------
- Unit test `_normalize_render_output` behavior:
  - pass `None`, legacy `dict`, `RenderResult`, `(artists, metadata)` tuples and mapping-like objects and assert normalization.
- Test `FigureFormatter.apply_figure_formatting` with synthetic artists:
  - create a fake Axes and Figure (`matplotlib.figure.Figure`) in headless mode,
  - supply a RenderResult with `mappable` and assert a Colorbar was created,
  - supply a line artist with `is_line` and assert legend appears.
- For animation: test `Figure.animate()` draws the first frame with full formatting present.

FAQ
---
Q: What if my component has no visible artists (pure metadata)?
A: Return `RenderResult(artists={}, metadata={"label": "info"})`. The formatter will
handle empty artists conservatively.

Q: Should components create colorbars themselves?
A: No. Components should create mappable artists and provide them via `artists`
and `metadata["mappable"]`. The formatter manages creation and placement of the
colorbar.

Q: How do I stabilize color scaling across frames?
A: Components can read the global style config (`style.color_range`) and/or
return `metadata["color_range"]` with explicit `min`/`max`. An automated
global-scan option is a separate feature; prefer explicit ranges for now.

Notes for reviewers
-------------------
- The RenderResult contract is intentionally permissive (metadata is advisory)
  — the goal is to communicate intent without strict enforcement.
- When in doubt, include a `mappable` reference and a `label`. This covers the
  most common formatter needs.

Contact
-------
If you want me to:
- scan `simbi/simbi/viz/components` for remaining legacy returns and produce a short report, or
- open PRs converting select components to `RenderResult` with minimal changes,

tell me which components to prioritize and I’ll proceed.
