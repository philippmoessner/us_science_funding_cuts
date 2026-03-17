"""Colormap Chooser Gradio Component
===================================

A reusable, importable Gradio component that provides a **scrollable, wide-strip**
chooser for Matplotlib (and compatible) colormaps. Designed to drop into an
existing Gradio Blocks app.

Features
--------
* Long, skinny gradient bars (not square tiles).
* Smart sampling:
  - Continuous maps → ~20 sample steps (configurable) interpolated across width.
  - Categorical / qualitative maps → actual number of colors (`cmap.N`).
* Scrollable gallery (height-capped w/ CSS).
* Selection callback returns the **selected colormap name** (string) you can pass
  directly to Matplotlib (`mpl.colormaps[name]` or `plt.get_cmap(name)`).
* Optional category + search filtering UI.
* Minimal dependencies: NumPy, Matplotlib, Gradio.

Quick Start
-----------
```python
import gradio as gr
from colormap_chooser import ColormapChooser, setup_colormaps

# Set up colormaps with custom collections
categories = setup_colormaps(
    included_collections=['matplotlib', 'cmocean', 'scientific'],
    excluded_collections=['colorcet']
)

chooser = ColormapChooser(
    categories=categories,
    gallery_kwargs=dict(columns=4, allow_preview=True, height="400px")
)

with gr.Blocks() as demo:
    with gr.Row():
        chooser.render()  # inserts the component cluster
    # Use chooser.selected_name as an input to your plotting fn
    import numpy as np, matplotlib.pyplot as plt
    def show_demo(cmap_name):
        data = np.random.rand(32, 32)
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=cmap_name)
        ax.set_title(cmap_name)
        fig.colorbar(im, ax=ax)
        return fig
    out = gr.Plot()
    chooser.selected_name.change(show_demo, chooser.selected_name, out)

demo.launch()
```

Installation
------------
Drop this file in your project (e.g., `colormap_chooser.py`) and import.

Customizing
-----------
Pass your own category dict, default sampling counts, or CSS overrides at
construction time; see class docstring below.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import gradio as gr
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------
# Default category mapping (extend or replace at init)
# ------------------------------------------------------------------
DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    "Perceptually Uniform": ["viridis", "plasma", "inferno", "magma", "cividis"],
    "Sequential": ["Blues", "Greens", "Oranges", "Purples", "Reds", "Greys"],
    "Diverging": ["coolwarm", "bwr", "seismic", "PiYG", "PRGn", "RdBu"],
    "Qualitative": ["tab10", "tab20", "Set1", "Set2", "Accent"],
}


# ------------------------------------------------------------------
# Colormap setup functions
# ------------------------------------------------------------------

def load_matplotlib_colormaps():
    """
    Load matplotlib's built-in colormaps directly.
    Returns dict of colormap_name -> colormap_object
    """
    matplotlib_cmaps = {}
    
    # Get all matplotlib colormaps
    for name in plt.colormaps():
        try:
            cmap = plt.get_cmap(name)
            matplotlib_cmaps[name] = cmap
        except Exception:
            continue
    
    return matplotlib_cmaps


def load_external_colormaps():
    """
    Load colormaps from external packages (like colormaps, cmocean, etc.).
    Returns dict of colormap_name -> colormap_object
    """
    external_cmaps = {}
    
    # Try to load from colormaps package
    try:
        import colormaps
        for attr_name in dir(colormaps):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(colormaps, attr_name)
                    # Check if it looks like a colormap
                    if hasattr(attr_value, '__call__') or hasattr(attr_value, 'colors'):
                        external_cmaps[attr_name] = attr_value
                except Exception:
                    continue
    except ImportError:
        pass
    
    return external_cmaps


def categorize_colormaps(
    colormap_dict: Dict[str, any], 
    included_collections: List[str], 
    excluded_collections: List[str]
) -> Dict[str, List[str]]:
    """
    Categorize colormaps by type with priority ordering.
    
    Args:
        colormap_dict: Dict of colormap_name -> colormap_object
        included_collections: List of collection names to include
        excluded_collections: List of collection names to exclude
    
    Returns:
        Dict {"Category": [list_of_names]} with colormaps ordered by collection priority
    """
    
    # Known categorizations based on documentation
    matplotlib_sequential = {
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',  # Perceptually uniform
        'ylorbr', 'ylorrd', 'orrd', 'purd', 'rdpu', 'bupu',  # Multi-hue sequential
        'gnbu', 'pubu', 'ylgnbu', 'pubugn', 'bugn', 'ylgn',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',  # Sequential (2)
        'spring', 'summer', 'autumn', 'winter', 'cool', 'wistia',
        'hot', 'afmhot', 'gist_heat', 'copper'
    }
    
    # Single-color sequential maps to exclude
    single_color_sequential = {
        'blues', 'greens', 'oranges', 'purples', 'reds', 'greys'
    }
    
    matplotlib_diverging = {
        'piyg', 'prgn', 'brbg', 'puor', 'rdgy', 'rdbu',
        'rdylbu', 'rdylgn', 'spectral', 'coolwarm', 'bwr', 'seismic',
        'berlin', 'managua', 'vanimo'
    }
    
    matplotlib_qualitative = {
        'pastel1', 'pastel2', 'paired', 'accent',
        'dark2', 'set1', 'set2', 'set3',
        'tab10', 'tab20', 'tab20b', 'tab20c'
    }
    
    matplotlib_miscellaneous = {
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'cmrmap', 'cubehelix', 'brg',
        'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
        'gist_ncar', 'twilight', 'twilight_shifted', 'hsv'
    }
    
    # External colormap collections
    cmocean_sequential = {
        'thermal', 'haline', 'solar', 'ice', 'gray', 'oxy', 'deep', 'dense', 
        'algae', 'matter', 'turbid', 'speed', 'amp', 'tempo', 'rain'
    }
    cmocean_diverging = {'balance', 'delta', 'curl', 'diff', 'tarn'}
    cmocean_other = {'phase', 'topo'}
    
    scientific_sequential = {
        'batlow', 'batlowK', 'batlowW', 'acton', 'bamako', 'bilbao', 'buda', 'davos',
        'devon', 'grayC', 'hawaii', 'imola', 'lajolla', 'lapaz', 'nuuk', 'oslo',
        'tokyo', 'turku', 'actonS', 'bamO', 'brocO', 'corko', 'corkO', 'davosS',
        'grayCS', 'hawaiiS', 'imolaS', 'lajollaS', 'lapazS', 'nuukS', 'osloS',
        'tokyoS', 'turkuS'
    }
    scientific_diverging = {
        'bam', 'bamo', 'berlin', 'broc', 'brocO', 'cork', 'corko', 'lisbon',
        'managua', 'roma', 'romao', 'tofino', 'vanimo', 'vik', 'viko'
    }
    
    cmasher_sequential = {
        'amber', 'amethyst', 'apple', 'arctic', 'autumn', 'bubblegum', 'chroma',
        'cosmic', 'dusk', 'ember', 'emerald', 'flamingo', 'freeze', 'gem', 'gothic',
        'heat', 'jungle', 'lavender', 'neon', 'neutral', 'nuclear', 'ocean',
        'pepper', 'plasma_r', 'rainforest', 'savanna', 'sunburst', 'swamp', 'torch',
        'toxic', 'tree', 'voltage', 'voltage_r'
    }
    cmasher_diverging = {
        'copper', 'emergency', 'fusion', 'guppy', 'holly', 'iceburn', 'infinity',
        'pride', 'prinsenvlag', 'redshift', 'seasons', 'seaweed', 'viola',
        'waterlily', 'watermelon', 'wildfire'
    }
    
    # Helper function to determine collection priority
    def get_collection_priority(name_lower):
        # Check matplotlib first (highest priority)
        if (name_lower in matplotlib_sequential or name_lower in matplotlib_diverging or 
            name_lower in matplotlib_qualitative or name_lower in matplotlib_miscellaneous):
            return 0
        # Then cmocean
        elif (name_lower in cmocean_sequential or name_lower in cmocean_diverging or name_lower in cmocean_other):
            return 1
        # Then scientific
        elif (name_lower in scientific_sequential or name_lower in scientific_diverging):
            return 2
        # Then cmasher
        elif (name_lower in cmasher_sequential or name_lower in cmasher_diverging):
            return 3
        # Everything else
        else:
            return 4
    
    # Collect all valid colormaps with their categories and priorities
    valid_colormaps = []
    
    for name, cmap_obj in colormap_dict.items():
        name_lower = name.lower()
        
        # Skip numbered variants (like brbg_9, set1_9, brbg_4_r, piyg_8_r, etc.)
        parts = name_lower.split('_')
        if len(parts) >= 2:
            # Check if second-to-last part is a digit (handles both name_4 and name_4_r)
            if parts[-2].isdigit():
                continue
            # Also check if last part is a digit (handles name_4)
            if parts[-1].isdigit():
                continue
            
        # Skip single-color sequential maps
        if name_lower in single_color_sequential:
            continue
        
        # Check if we should include this colormap based on collection filters
        include_cmap = True
        
        # Check excluded collections
        for excluded in excluded_collections:
            if excluded.lower() in name_lower:
                include_cmap = False
                break
        
        if not include_cmap:
            continue
        
        # Check included collections
        if included_collections:
            include_cmap = False
            for included in included_collections:
                if (included.lower() in name_lower or 
                    # Special handling for matplotlib colormaps
                    (included == 'matplotlib' and name in plt.colormaps()) or
                    # Special handling for known colormap sets
                    name_lower in cmocean_sequential or name_lower in cmocean_diverging or name_lower in cmocean_other or
                    name_lower in scientific_sequential or name_lower in scientific_diverging or
                    name_lower in cmasher_sequential or name_lower in cmasher_diverging):
                    include_cmap = True
                    break
        
        if not include_cmap:
            continue
        
        # Categorize the colormap
        category = None
        if (name_lower in matplotlib_qualitative or 
            any(qual in name_lower for qual in ['tab10', 'tab20', 'set1', 'set2', 'set3', 'paired', 'accent', 'pastel', 'dark2'])):
            category = "Qualitative"
        elif (name_lower in cmocean_sequential or name_lower in scientific_sequential or 
              name_lower in cmasher_sequential or name_lower in matplotlib_sequential or
              'sequential' in name_lower or
              any(seq in name_lower for seq in ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])):
            category = "Sequential"
        elif (name_lower in cmocean_diverging or name_lower in scientific_diverging or 
              name_lower in cmasher_diverging or name_lower in matplotlib_diverging or
              'diverging' in name_lower or
              any(div in name_lower for div in ['bwr', 'coolwarm', 'seismic', 'rdbu', 'rdgy', 'piyg', 'prgn', 'brbg'])):
            category = "Diverging"
        else:
            category = "Other"
        
        if category:
            priority = get_collection_priority(name_lower)
            valid_colormaps.append((name, category, priority))
    
    # Sort by category, then by priority, then by name
    valid_colormaps.sort(key=lambda x: (x[1], x[2], x[0].lower()))
    
    # Group by category while maintaining order
    categories = {
        "Sequential": [],
        "Diverging": [],
        "Qualitative": [],
        "Other": []
    }
    
    for name, category, priority in valid_colormaps:
        categories[category].append(name)
    
    # Remove empty categories and hide "Other" category
    final_categories = {}
    for cat_name, cmap_names in categories.items():
        if cmap_names and cat_name != "Other":  # Hide "Other" category
            final_categories[cat_name] = cmap_names
    
    return final_categories


def setup_colormaps(
    included_collections: Optional[List[str]] = None,
    excluded_collections: Optional[List[str]] = None,
    additional_colormaps: Optional[Dict[str, any]] = None
) -> Dict[str, List[str]]:
    """
    Set up and categorize colormaps from various sources.
    
    Args:
        included_collections: List of collection names to include 
            (e.g., ['matplotlib', 'cmocean', 'scientific'])
        excluded_collections: List of collection names to exclude
        additional_colormaps: Dict of additional colormaps to include
    
    Returns:
        Dict of {"Category": [list_of_colormap_names]} ready for ColormapChooser
    """
    if excluded_collections is None:
        excluded_collections = ['colorcet', 'carbonplan', 'sciviz']
    
    if included_collections is None:
        included_collections = ['matplotlib', 'cmocean', 'scientific', 'cmasher', 'colorbrewer', 'cartocolors']
    
    # Combine all colormaps
    all_colormaps = {}
    
    # Add matplotlib colormaps
    if 'matplotlib' in included_collections:
        matplotlib_cmaps = load_matplotlib_colormaps()
        all_colormaps.update(matplotlib_cmaps)
        print(f"Added {len(matplotlib_cmaps)} matplotlib colormaps")
    
    # Add external colormaps
    try:
        external_cmaps = load_external_colormaps()
        all_colormaps.update(external_cmaps)
        print(f"Added {len(external_cmaps)} external colormaps")
    except Exception as e:
        print(f"Could not load external colormaps: {e}")
    
    # Add any additional colormaps
    if additional_colormaps:
        all_colormaps.update(additional_colormaps)
        print(f"Added {len(additional_colormaps)} additional colormaps")
    
    # Categorize colormaps
    return categorize_colormaps(all_colormaps, included_collections, excluded_collections)


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _flatten_categories(categories: Dict[str, Sequence[str]]) -> List[str]:
    names = []
    for _, vals in categories.items():
        names.extend(vals)
    # maintain insertion order; drop dupes while preserving first occurrence
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _build_name2cat(categories: Dict[str, Sequence[str]]) -> Dict[str, str]:
    m = {}
    for cat, vals in categories.items():
        for n in vals:
            m[n] = cat
    return m


# ------------------------------------------------------------------
# Sampling policy
# ------------------------------------------------------------------

def _is_categorical_cmap(
    cmap: mcolors.Colormap,
    declared_category: Optional[str] = None,
    qualitative_label: str = "Qualitative",
    max_auto: int = 32,
) -> bool:
    """Heuristic: treat as categorical/qualitative.

    Priority:
    1. If user-declared category == qualitative_label → True.
    2. If ListedColormap with small N → True.
    3. If colormap name suggests it's qualitative → True.
    4. Else False (continuous).
    """
    # Check if explicitly declared as qualitative
    if declared_category == qualitative_label:
        return True
    
    # Check if it's a ListedColormap with small N
    if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= max_auto:
        return True
    
    # Additional check: if the colormap name suggests it's qualitative
    # This is a fallback in case the declared_category doesn't match exactly
    if hasattr(cmap, 'name'):
        name_lower = cmap.name.lower()
        qualitative_names = {
            'tab10', 'tab20', 'tab20b', 'tab20c', 'set1', 'set2', 'set3',
            'pastel1', 'pastel2', 'paired', 'accent', 'dark2'
        }
        if name_lower in qualitative_names:
            return True
    
    return False


def _cmap_strip(
    name: str,
    width: int = 10,
    height: int = 16,
    smooth_steps: int = 20,
    declared_category: Optional[str] = None,
    qualitative_label: str = "Qualitative",
    max_auto: int = 32,
):
    """Return RGB uint8 preview strip for *name* colormap.

    Continuous maps are resampled to *smooth_steps* and linearly interpolated.
    Categorical maps use actual number of colors, but adapt to available width.
    """
    cmap = mpl.colormaps[name]
    categorical = _is_categorical_cmap(
        cmap, declared_category=declared_category, qualitative_label=qualitative_label, max_auto=max_auto
    )
    
    if categorical:
        n = cmap.N
        if hasattr(cmap, "colors"):
            cols = np.asarray(cmap.colors)
            if cols.shape[1] == 4:
                cols = cols[:, :3]
        else:
            xs = np.linspace(0, 1, n, endpoint=False) + (0.5 / n)
            cols = cmap(xs)[..., :3]
        
        # Adaptive approach based on available width
        min_block_width = 3  # Minimum pixels per color block for visibility
        
        if width >= n * min_block_width:
            # We have enough width to show all colors as distinct blocks
            block_w = width // n
            selected_cols = cols
            num_blocks = n
        else:
            # Not enough width - show a representative sample
            max_colors_that_fit = max(2, width // min_block_width)  # At least 2 colors
            
            if max_colors_that_fit >= n:
                # We can fit all colors
                selected_cols = cols
                num_blocks = n
                block_w = width // n
            else:
                # Sample evenly across the colormap
                indices = np.linspace(0, n-1, max_colors_that_fit, dtype=int)
                selected_cols = cols[indices]
                num_blocks = max_colors_that_fit
                block_w = width // num_blocks
        
        # Debug output for categorical sampling
        if name.lower() in ['tab10', 'tab20', 'set1', 'set2', 'accent', 'paired']:
            print(f'CATEGORICAL SAMPLING DEBUG: {name}')
            print(f'  n (total colors): {n}')
            print(f'  width: {width}')
            print(f'  num_blocks (colors shown): {num_blocks}')
            print(f'  block_w (width per color): {block_w}')
            print(f'  showing all colors: {num_blocks == n}')
            print('---')
        
        # Create the array with discrete blocks
        arr = np.repeat(selected_cols[np.newaxis, :, :], height, axis=0)  # (h,num_blocks,3)
        arr = np.repeat(arr, block_w, axis=1)  # (h,num_blocks*block_w,3)
        
        # Handle any remaining width
        current_width = arr.shape[1]
        if current_width < width:
            # Pad by extending the last color
            pad = width - current_width
            last_color = arr[:, -1:, :]  # Get last column
            padding = np.repeat(last_color, pad, axis=1)
            arr = np.concatenate([arr, padding], axis=1)
        elif current_width > width:
            # Trim to exact width
            arr = arr[:, :width, :]
        
        return (arr * 255).astype(np.uint8)

    # continuous - unchanged
    xs = np.linspace(0, 1, smooth_steps)
    cols = cmap(xs)[..., :3]
    xi = np.linspace(0, smooth_steps - 1, width)
    lo = np.floor(xi).astype(int)
    hi = np.minimum(lo + 1, smooth_steps - 1)
    t = xi - lo
    strip = (1 - t)[:, None] * cols[lo] + t[:, None] * cols[hi]
    arr = np.repeat(strip[np.newaxis, :, :], height, axis=0)
    return (arr * 255).astype(np.uint8)


# ------------------------------------------------------------------
# ColormapChooser class
# ------------------------------------------------------------------
class ColormapChooser:
    """Reusable scrollable colormap selector for Gradio.

    Parameters
    ----------
    categories:
        Dict mapping *Category Label* → list of cmap names. If None, uses
        DEFAULT_CATEGORIES defined above. You may pass additional categories or
        override existing ones. Order preserved.
    smooth_steps:
        Approx sample count for continuous maps (default 20).
    strip_width:
        Pixel width of preview strip images (default 512).
    strip_height:
        Pixel height of preview strip images (default 16).
    css_height:
        Max CSS height (pixels) for the scrollable gallery viewport.
    qualitative_label:
        Category label used to force qualitative sampling when present.
    max_auto:
        If a ListedColormap has N <= max_auto, treat as categorical even if not
        declared Qualitative.
    elem_id:
        DOM id for the gallery (used to scope CSS overrides). Default 'cmap_gallery'.
    show_search:
        Whether to render the search Textbox.
    show_category:
        Whether to render the category Radio selector.
    show_preview:
        Show the big preview strip under the gallery.  Off by default.
    show_selected_name:
        Show the textbox that echoes the selected colormap name.  Off by default.
    show_selected_info:
        Show the markdown info line.  Off by default.
    gallery_kwargs:
        Dictionary of keyword arguments to pass to the Gradio Gallery component
        when it is created. For example, `columns=4, allow_preview=True, height="400px"`.

    Public attributes after render():
        category (optional)
        search (optional)
        gallery
        preview
        selected_name  (Textbox; value string)
        selected_info  (Markdown)
        names_state    (State of current filtered cmap names)

    Usage: see module Quick Start above.
    """

    def __init__(
        self,
        *,
        categories: Optional[Dict[str, Sequence[str]]] = None,
        smooth_steps: int = 10,
        strip_width: int = 10,
        strip_height: int = 16,
        css_height: int = 240,
        qualitative_label: str = "Qualitative",
        max_auto: int = 32,
        elem_id: str = "cmap_gallery",
        show_search: bool = True,
        show_category: bool = True,
        columns: int = 3,
        thumb_margin_px: int = 2,          # NEW
        gallery_kwargs: Optional[Dict[str, Any]] = None,
        show_preview: bool = False,
        show_selected_name: bool = False,
        show_selected_info: bool = True,
    ) -> None:
        self.categories = categories if categories is not None else DEFAULT_CATEGORIES
        self.smooth_steps = smooth_steps
        self.strip_width = strip_width
        self.strip_height = strip_height
        self.css_height = css_height
        self.qualitative_label = qualitative_label
        self.max_auto = max_auto
        self.elem_id = elem_id
        self.show_search = show_search
        self.show_category = show_category
        self.columns = columns
        self.thumb_margin_px = thumb_margin_px   # NEW
        self.gallery_kwargs = gallery_kwargs or {}
        # visibility flags
        self.show_preview = show_preview
        self.show_selected_name = show_selected_name
        self.show_selected_info = show_selected_info
        self._all_names = _flatten_categories(self.categories)
        self._name2cat = _build_name2cat(self.categories)
        self._tile_cache: Dict[str, np.ndarray] = {}

        # public gradio components (populated in render)
        self.category = None
        self.search = None
        self.gallery = None
        self.preview = None
        self.selected_name = None
        self.selected_info = None
        self.names_state = None

    # ------------------
    # internal helpers
    # ------------------
    def _tile(self, name: str) -> np.ndarray:
        if name not in self._tile_cache:
            self._tile_cache[name] = _cmap_strip(
                name,
                width=self.strip_width,
                height=self.strip_height,
                smooth_steps=self.smooth_steps,
                declared_category=self._name2cat.get(name),
                qualitative_label=self.qualitative_label,
                max_auto=self.max_auto,
            )
        return self._tile_cache[name]

    def _make_gallery_items(self, names: Sequence[str]):
        return [(self._tile(n), n) for n in names]

    # ------------------
    # event functions
    # ------------------
    def _filter(self, cat: str, s: str):
        if self.show_category and cat in self.categories:
            names = list(self.categories[cat])
        else:
            names = list(self._all_names)

        if s and self.show_search:
            sl = s.lower()
            names = [n for n in names if sl in n.lower()]

        # Remember new list for the select-callback
        self.names_state.value = names

        # 1) return an updated gallery
        gkw = {
            "value": self._make_gallery_items(names),
            "selected_index": None,
        }
        gkw.update(self.gallery_kwargs)
        gallery_update = gr.Gallery(**gkw)
        # 2) clear the other widgets so old selection disappears
        preview_update  = gr.update(value=None)
        name_update     = gr.update(value="")
        info_update     = gr.update(value="")

        return gallery_update, preview_update, name_update, info_update

    def _select(self, evt: gr.SelectData, names: Sequence[str]):
        if not names or evt.index is None or evt.index >= len(names):
            return gr.update(), "", "Nothing selected"
        name = names[evt.index]
        big = _cmap_strip(
            name,
            width=max(self.strip_width * 2, 768),
            height=max(self.strip_height * 2, 32),
            smooth_steps=self.smooth_steps,
            declared_category=self._name2cat.get(name),
            qualitative_label=self.qualitative_label,
            max_auto=self.max_auto,
        )
        info = f"**Selected:** `{name}` _(Category: {self._name2cat.get(name, '?')})_"
        return big, name, info

    # ------------------
    # CSS block builder
    # ------------------
    def css(self) -> str:
        return f"""
        /* ───── 0. easy visual check the CSS is live (remove later) ───── */
        #{self.elem_id} {{ 
        /* background:rgba(255,255,0,.05); */
        }}
        
        /* the wrapper *is* the .block, so it owns the padding var */
        #{self.elem_id}_wrap {{
            padding: 0 !important;
            --block-padding: 0 !important;
        }}
        
        /* ───── 1. the wrapper Gradio marks .fixed-height: make it scroll  ─── */
        #{self.elem_id} .grid-wrap {{
            height: {self.css_height}px;          /* kill inline 200 px or similar */
            max-height: {self.css_height}px;  /* cap the gallery’s height      */
            overflow-y: auto;                 /* rows that don’t fit will scroll */
        }}

        /* ───── 2. the real grid: keep masonry maths intact, tweak gap ─── */
        #{self.elem_id} .grid-container {{
            height: auto !important;          /* sometimes Gradio sets one     */
            gap: 7px;        /* tighter gutters (define attr) */
            grid-auto-rows:auto !important;
        }}

        /* ───── 3. thumbnail boxes keep your ultra-wide shape ──────────── */
        #{self.elem_id} .thumbnail-item {{
            aspect-ratio: 3/1;  /* e.g. 5/1 */
            height: auto !important;          /* beats Gradio’s inline 100 %   */
            margin: {self.thumb_margin_px}px !important;
            overflow: hidden;                 /* just in case                  */
        }}

        /* ───── 4. images fill each box neatly ─────────────────────────── */
        #{self.elem_id} img {{
            width: 100%;
            height: 100%;
            object-fit: cover;                /* crop to fill                  */
            object-position: left;
            display: block;                   /* kill inline-img whitespace    */
        }}

        /* ───── 5. widen the “Selected:” info line ───────────────────── */
        .cmap_selected_info {{
            max-width: 100% !important;   /* kill default 45 rem limit   */
        }}
        """

    # ------------------
    # Render into an existing Blocks context
    # ------------------
    def render(self):
        """Create Gradio UI elements and wire callbacks.

        Must be called *inside* an active `gr.Blocks()` context.
        Returns a tuple `(components_dict)` for convenience.
        """
        # initial list: first category or all
        if self.show_category:
            first_cat = next(iter(self.categories))
            init_names = list(self.categories[first_cat])
        else:
            init_names = list(self._all_names)

        # preheat tiles lazily on demand; no bulk precompute
        # (call _tile when building gallery items)

        # layout
        if self.show_category or self.show_search:
            with gr.Row():
                if self.show_category:
                    self.category = gr.Radio(list(self.categories.keys()), value=first_cat, label="Category")
                else:
                    self.category = gr.State(None)  # shim so filter signature works
                if self.show_search:
                    self.search = gr.Textbox(label="Search", placeholder="type to filter...")
                else:
                    self.search = gr.State("")
        else:
            self.category = gr.State(None)
            self.search = gr.State("")

        self.names_state = gr.State(init_names)

        gkw = {
            "value": self._make_gallery_items(init_names),
            "label": None,                   # remove label
            "allow_preview": False,
            "elem_id": self.elem_id,
            "show_share_button": False,
            "columns": getattr(self, "columns", 3),
        }
        gkw.update(self.gallery_kwargs)
        self.gallery = gr.Gallery(**gkw)

        self.preview = gr.Image(
            label="Preview", interactive=False, height=60, visible=self.show_preview
        )
        self.selected_name = gr.Textbox(
            label="Selected cmap", interactive=False, visible=self.show_selected_name
        )
        self.selected_info = gr.Markdown(
            visible=self.show_selected_info,
            elem_classes="cmap_selected_info",
        )

        # wiring
        if self.show_category or self.show_search:

            def _wrapped_filter(cat, s):
                if not self.show_category:
                    cat = None
                if not self.show_search:
                    s = ""
                return self._filter(cat, s)

            outputs = [self.gallery,
                       self.preview,
                       self.selected_name,
                       self.selected_info]

            if self.show_category:
                self.category.change(
                    _wrapped_filter,
                    [self.category, self.search],
                    outputs
                )
            if self.show_search:
                self.search.change(
                    _wrapped_filter,
                    [self.category, self.search],
                    outputs
                )

        def _wrapped_select(evt: gr.SelectData, names):
            return self._select(evt, names)

        self.gallery.select(_wrapped_select, [self.names_state],
                            [self.preview, self.selected_name, self.selected_info])

        return {
            "gallery": self.gallery,
            "selected_name": self.selected_name,
            "preview": self.preview,
            "info": self.selected_info,
            "category": self.category,
            "search": self.search,
            "names_state": self.names_state,
        }

    # ==========================================================
    # NEW TAB-BASED RENDERER
    # ==========================================================
    def render_tabs(self):
        """
        Render the chooser as one Gallery per category inside a gradio Tabs
        container.  No search box is provided – each tab already filters
        by category.

        Returns the same components dict as `render()`, plus a "galleries"
        dict that maps category → Gallery component.
        """
        galleries = {}

        with gr.Tabs() as root_tabs:

            # --- build a tab + gallery for every category -------------
            for cat, names in self.categories.items():
                with gr.TabItem(cat):
                    gkw = {
                        "value": self._make_gallery_items(names),
                       "label": None,         # remove label
                        "allow_preview": False,
                        "show_share_button": False,
                        "elem_id": self.elem_id,
                        "columns": getattr(self, "columns", 3),
                        "show_label": False
                    }
                    gkw.update(self.gallery_kwargs)
                    with gr.Row(elem_id=f"{self.elem_id}_wrap"):   # ← new wrapper
                        gal = gr.Gallery(**gkw)
                    galleries[cat] = gal

        # --- shared preview / meta area under the tabs ----------------
        self.preview = gr.Image(
            label="Preview", interactive=False, height=60, visible=self.show_preview
        )
        self.selected_name = gr.Textbox(
            label="Selected cmap", interactive=False, visible=self.show_selected_name
        )
        self.selected_info = gr.Markdown(
            visible=self.show_selected_info,
            elem_classes="cmap_selected_info",
        )

        # --- wiring: every gallery uses the same _select callback -----
        def _wrapped_select(evt: gr.SelectData, names):
            return self._select(evt, names)

        for cat, gal in galleries.items():
            gal.select(
                _wrapped_select,
                [gr.State(list(self.categories[cat]))],      # names list
                [self.preview, self.selected_name, self.selected_info],
            )

        return {
            "galleries": galleries,
            "selected_name": self.selected_name,
            "preview": self.preview,
            "info": self.selected_info,
            "tabs": root_tabs,
        }


# ------------------------------------------------------------------
# Minimal self-demo (only runs if module executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    chooser = ColormapChooser()
    with gr.Blocks(css=chooser.css()) as demo:
        gr.Markdown("## Colormap Chooser Demo")
        chooser.render()
    demo.launch()
