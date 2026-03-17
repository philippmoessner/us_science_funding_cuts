"""legend_builders.py
====================
Minimal‑dependency helpers that generate **static** legend HTML + CSS matching
DataMapPlot’s own class names.  Drop the returned strings straight into
``create_interactive_plot(custom_html=…, custom_css=…)``.

Highlights
----------
* **continuous_legend_html_css** – full control over ticks, label, size &
  absolute position (via an *anchor* keyword).
* **categorical_legend_html_css** – swatch legend with optional title, flexible
  anchor, row/column layout and custom swatch size.

Both helpers return ``(html, css)`` so you can concatenate multiple legends.
No JavaScript is injected – they render statically but look native.  If you
later add JS (e.g. DMP’s `ColorLegend` behaviour), the class names already fit.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union
from datetime import datetime, date
import matplotlib.cm as _cm
from matplotlib.colors import to_hex, to_rgb

Colour = Union[str, tuple]
__all__ = ["continuous_legend_html_css", "categorical_legend_html_css"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _hex(c: Colour) -> str:
    """Convert *c* to #RRGGBB hex (handles any Matplotlib‑parsable colour)."""
    return c if isinstance(c, str) else to_hex(to_rgb(c))


def _gradient(cmap: Union[str, _cm.Colormap, Sequence[str]], *, vertical: bool = True) -> str:
    """Return a CSS linear‑gradient from a Matplotlib cmap or explicit colour list."""
    if isinstance(cmap, (list, tuple)):
        stops = [_hex(c) for c in cmap]
    else:
        cmap = _cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        stops = [to_hex(cmap(i / 255)) for i in range(256)]
    direction = "to top" if vertical else "to right"
    return f"linear-gradient({direction}, {', '.join(stops)})"


_ANCHOR_CSS: Dict[str, str] = {
    "top-left": "top:10px; left:10px;",
    "top-right": "top:10px; right:10px;",
    "bottom-left": "bottom:10px; left:10px;",
    "bottom-right": "bottom:10px; right:10px;",
    "middle-left": "top:50%; left:10px; transform:translateY(-50%);",
    "middle-right": "top:50%; right:10px; transform:translateY(-50%);",
    "middle-center": "top:50%; left:50%; transform:translate(-50%,-50%);",
}

# ---------------------------------------------------------------------------
# continuous legend
# ---------------------------------------------------------------------------

def continuous_legend_html_css(
    cmap: Union[str, _cm.Colormap, Sequence[str]],
    vmin: Union[int, float, datetime, date],
    vmax: Union[int, float, datetime, date],
    *,
    ticks: Sequence[Union[int, float, datetime, date]] | None = None,
    label: str | None = None,
    bar_size: tuple[int, int] = (10, 200),
    anchor: str = "top-right",
    container_id: str = "dmp-colorbar",
) -> Tuple[str, str]:
    """Return *(html, css)* snippet for a static colour‑bar legend."""

    # ---------- ticks -----------------------------------------------------
    if ticks is None:
        ticks = [vmin + (vmax - vmin) * i / 4 for i in range(5)]  # type: ignore

    def _fmt(val):
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y")
        sci = max(abs(float(vmin)), abs(float(vmax))) >= 1e5 or 0 < abs(float(vmin)) <= 1e-4
        if sci:
            return f"{val:.1e}"
        return f"{val:.0f}" if float(val).is_integer() else f"{val:.2f}"

    tick_labels = [_fmt(t) for t in ticks]

    # relative positions (0% top, 100% bottom) -----------------------------
    def _rel(val):
        if isinstance(val, (datetime, date)):
            rng = (ticks[-1] - ticks[0]).total_seconds() or 1
            return (ticks[-1] - val).total_seconds() / rng * 100
        rng = float(ticks[-1] - ticks[0]) or 1
        return (ticks[-1] - val) / rng * 100

    # ---------- HTML ------------------------------------------------------
    w, h = bar_size
    html: List[str] = [f'<div id="{container_id}" class="colorbar-container">']

    if label:
        html.append(
            f'  <div class="colorbar-label" style="writing-mode:vertical-rl; transform:rotate(180deg); margin-right:8px;">{label}</div>'
        )

    html.append(f'  <div class="colorbar" style="width:{w}px; height:{h}px; background:{_gradient(cmap)};"></div>')
    html.append('  <div class="colorbar-tick-container">')

    for pos, lab in zip([_rel(t) for t in ticks], tick_labels):
        html.append(
            f'    <div class="colorbar-tick" style="top:{pos:.2f}%;">'
            '      <div class="colorbar-tick-line"></div>'
            f'      <div class="colorbar-tick-label">{lab}</div>'
            '    </div>'
        )

    html.extend(['  </div>', '</div>'])

    # ---------- CSS -------------------------------------------------------
    pos_css = _ANCHOR_CSS.get(anchor, _ANCHOR_CSS["top-right"])
    css = f"""
#{container_id} {{position:absolute; {pos_css} z-index:100; display:flex; align-items:center; gap:4px; padding:10px;}}
#{container_id} .colorbar-tick-container {{position:relative; width:40px; height:{h}px;}}
#{container_id} .colorbar-tick {{position:absolute; display:flex; align-items:center; gap:4px; transform:translateY(-50%); font-size:12px;}}
#{container_id} .colorbar-tick-line {{width:8px; height:1px; background:#333;}}
#{container_id} .colorbar-label {{font-size:12px;}}
"""

    return "\n".join(html), css

# ---------------------------------------------------------------------------
# categorical legend
# ---------------------------------------------------------------------------

def categorical_legend_html_css(
    color_mapping: Dict[str, Colour],
    *,
    title: str | None = None,
    swatch: int = 12,
    anchor: str = "bottom-left",
    container_id: str = "dmp-catlegend",
    rows: bool = True,
) -> Tuple[str, str]:
    """Return *(html, css)* for a swatch legend."""

    html: List[str] = [f'<div id="{container_id}" class="color-legend-container">']
    if title:
        html.append(f'  <div class="legend-title">{title}</div>')
    for lbl, col in color_mapping.items():
        html.append(
            '  <div class="legend-item">'
            f'    <div class="color-swatch-box" style="background:{_hex(col)};"></div>'
            f'    <div class="legend-label">{lbl}</div>'
            '  </div>'
        )
    html.append('</div>')

    pos_css = _ANCHOR_CSS.get(anchor, _ANCHOR_CSS["bottom-left"])
    css = f"""
#{container_id} {{position:absolute; {pos_css} z-index:100; display:flex; flex-direction:{'column' if rows else 'row'}; gap:4px; padding:10px;}}
#{container_id} .legend-title {{font-weight:bold; margin-bottom:4px;}}
#{container_id} .legend-item {{display:flex; align-items:center; gap:4px;}}
#{container_id} .color-swatch-box {{width:{swatch}px; height:{swatch}px; border-radius:2px; border:1px solid #555;}}
#{container_id} .legend-label {{font-size:12px;}}
"""

    return "\n".join(html), css

# ---------------------------------------------------------------------------
# sample script for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # pip install datamapplot matplotlib numpy to run this demo
    import numpy as np
    from matplotlib import cm
    import datamapplot as dmp

    # dummy data ----------------------------------------------------------
    n = 400
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(n, 2))
    years = rng.integers(1990, 2025, size=n)

    # quadrant labels -----------------------------------------------------
    quad = np.where(coords[:, 0] >= 0,
                    np.where(coords[:, 1] >= 0, "A", "D"),
                    np.where(coords[:, 1] >= 0, "B", "C"))

    # colours -------------------------------------------------------------
    grey = "#bbbbbb"
    cols = np.full(n, grey, dtype=object)
    mask = rng.random(n) < 0.1
    vir = cm.get_cmap("viridis")
    cols[mask] = [to_hex(vir((y - years.min())/(years.max()-years.min()))) for y in years[mask]]

    # legends -------------------------------------------------------------
    html_bar, css_bar = continuous_legend_html_css(
        vir, years.min(), years.max(), label="Year", anchor="middle-right", ticks=[1990, 2000, 2010, 2020, 2024]
    )
    html_cat, css_cat = categorical_legend_html_css(
        {lbl: col for lbl, col in zip("ABCD", cm.tab10.colors)}, title="Quadrant", anchor="bottom-left"
    )

    custom_html = html_bar + html_cat
    custom_css = css_bar + css_cat

    # plot ---------------------------------------------------------------
    plot = dmp.create_interactive_plot(
        coords, quad,
        hover_text=np.arange(n).astype(str),
        marker_color_array=cols,
        custom_html=custom_html,
        custom_css=custom_css,
    )

    # In Jupyter this shows automatically; otherwise save:
    # with open("demo.html", "w") as f:
    #     f.write(str(plot))

    print("Demo plot generated – view in a notebook or open the saved HTML.")
