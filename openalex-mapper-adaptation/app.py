#import spaces #
import time
print(f"Starting up: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# source openalex_env_map/bin/activate
# Standard library imports

import os

#Enforce local cching:


# os.makedirs("./pip_cache", exist_ok=True)
# Pip:
# os.makedirs("./pip_cache", exist_ok=True)
# os.environ["PIP_CACHE_DIR"] = os.path.abspath("./pip_cache")
# # MPL:
# os.makedirs("./mpl_cache", exist_ok=True)
# os.environ["MPLCONFIGDIR"] = os.path.abspath("./mpl_cache")
# #Transformers
# os.makedirs("./transformers_cache", exist_ok=True)
# os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./transformers_cache")

# import numba
# print(numba.config)
# print("Numba threads:", numba.get_num_threads())
# numba.set_num_threads(16)
# print("Updated Numba threads:", numba.get_num_threads())

# import datamapplot.medoids


# print(help(datamapplot.medoids))



from pathlib import Path
from datetime import datetime
from itertools import chain
import ast  # Add this import at the top with the standard library imports
import hashlib
import html as html_lib
import mimetypes

import base64
import json
import pickle
import urllib.parse
import urllib.request

# Third-party imports
import numpy as np
# Compatibility shim for NumPy 2.x: some deps (e.g. pynndescent/umap) still reference `np.infty`.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]
import pandas as pd
import torch
import gradio as gr

print(f"Gradio version: {gr.__version__}")

import subprocess
import re
from color_utils import rgba_to_hex

def print_datamapplot_version():
    try:
        # On Unix systems, you can pipe commands by setting shell=True.
        version = subprocess.check_output("pip freeze | grep datamapplot", shell=True, text=True)
        print("datamapplot version:", version.strip())
    except subprocess.CalledProcessError:
        print("datamapplot not found in pip freeze output.")

print_datamapplot_version()



from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import matplotlib.pyplot as plt
import tqdm
import colormaps
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

import random

import opinionated # for fonts
plt.style.use("opinionated_rc")

from sklearn.neighbors import NearestNeighbors


def is_running_in_hf_zero_gpu():
    value = os.environ.get("SPACES_ZERO_GPU")
    print(value)
    return value is not None and value.lower() in ("1", "t", "true")
    
is_running_in_hf_zero_gpu()

def is_running_in_hf_space():
    return "SPACE_ID" in os.environ

# #if is_running_in_hf_space():
# from spaces.zero.client import _get_token
    
    
try:
    import spaces
    HAS_SPACES = True
except (ImportError, ModuleNotFoundError):
    HAS_SPACES = False

# Provide a harmless fallback so decorators don't explode
if not HAS_SPACES:
    class _Dummy:
        def GPU(self, *a, **k):
            def deco(f):  # no-op decorator
                return f
            return deco
    spaces = _Dummy()          # fake module object

def _get_token(request: gr.Request):
    """
    Return Hugging Face's per-user ZeroGPU token (JWT), if present.

    Newer `spaces` versions (>=0.47.0) no longer expose `spaces.zero.client._get_token`,
    so we read the header directly and keep a legacy fallback.
    """
    if request is None:
        return None

    headers = getattr(request, "headers", None)
    if headers is None:
        return None

    # Current ZeroGPU handshake header
    try:
        token = headers.get("x-ip-token") or headers.get("X-IP-Token")
        if token:
            return token
    except Exception:
        pass

    if not HAS_SPACES:
        return None

    # Backwards-compatible fallback for older `spaces` versions
    try:
        from spaces.zero.client import _get_token as legacy_get_token  # type: ignore
    except Exception:
        return None

    try:
        token = legacy_get_token(request)
    except Exception:
        return None

    return token or None


#if is_running_in_hf_space():
#import spaces # necessary to run on Zero.
#print(f"Spaces version: {spaces.__version__}")

import datamapplot
import pyalex

# Local imports
from openalex_utils import (
    openalex_url_to_pyalex_query, 
    get_field,
    process_records_to_df,
    openalex_url_to_filename,
    get_records_from_dois,
    openalex_url_to_readable_name
)
from ui_utils import highlight_queries
from styles import DATAMAP_CUSTOM_CSS
from data_setup import (
    download_required_files,
    setup_basemap_data,
    setup_mapper,
    setup_embedding_model,
    
)

from network_utils import create_citation_graph, draw_citation_graph

# Add colormap chooser imports
from colormap_chooser import ColormapChooser, setup_colormaps

# Add legend builder imports
try:
    from legend_builders import continuous_legend_html_css, categorical_legend_html_css
    HAS_LEGEND_BUILDERS = True
except ImportError:
    print("Warning: legend_builders.py not found. Legends will be disabled.")
    HAS_LEGEND_BUILDERS = False


# Configure OpenAlex
pyalex.config.email = "maximilian.noichl@uni-bamberg.de"

print(f"Imports completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Set up colormaps for the chooser
print("Setting up colormaps...")
colormap_categories = setup_colormaps(
    included_collections=['matplotlib', 'cmocean', 'scientific', 'cmasher'],
    excluded_collections=['colorcet', 'carbonplan', 'sciviz']
)

colormap_chooser = ColormapChooser(
    categories=colormap_categories,
    smooth_steps=10,
    strip_width=200,
    strip_height=50,
    css_height=200,
    # show_search=False,
    # show_category=False,
    # show_preview=False,
    # show_selected_name=True,
    # show_selected_info=False,
    gallery_kwargs=dict(columns=3, allow_preview=False, height="200px")
)


# Create a static directory to store the dynamic HTML files
static_dir = Path("./static")
static_dir.mkdir(parents=True, exist_ok=True)
VIS_STATIC_ROUTE = "/viz_static"

# Tell Gradio which absolute paths are allowed to be served
os.environ["GRADIO_ALLOWED_PATHS"] = str(static_dir.resolve())
print("os.environ['GRADIO_ALLOWED_PATHS'] =", os.environ["GRADIO_ALLOWED_PATHS"])


# Create FastAPI app
app = FastAPI()

# CHANGED (font regression fix): avoid mounting on `/static`, which conflicts with Gradio's
# own static route (including Gradio UI font files).
app.mount(VIS_STATIC_ROUTE, StaticFiles(directory="static"), name="viz_static")





# Resource configuration
REQUIRED_FILES = {
    "100k_filtered_OA_sample_cluster_and_positions_supervised.pkl": 
        "https://huggingface.co/datasets/m7n/intermediate_sci_pickle/resolve/main/100k_filtered_OA_sample_cluster_and_positions_supervised.pkl",
    "umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl":
        "https://huggingface.co/datasets/m7n/intermediate_sci_pickle/resolve/main/umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl"
}
BASEMAP_PATH = "100k_filtered_OA_sample_cluster_and_positions_supervised.pkl"
MAPPER_PARAMS_PATH = "umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl"
MODEL_NAME = "m7n/discipline-tuned_specter_2_024"

# Initialize models and data
start_time = time.time()
print("Initializing resources...")

download_required_files(REQUIRED_FILES)
basedata_df = setup_basemap_data(BASEMAP_PATH)
mapper = setup_mapper(MAPPER_PARAMS_PATH)
model = setup_embedding_model(MODEL_NAME)

print(f"Resources initialized in {time.time() - start_time:.2f} seconds")




# Setting up decorators for embedding on HF-Zero:
def no_op_decorator(func):
    """A no-op (no operation) decorator that simply returns the function."""
    def wrapper(*args, **kwargs):
        # Do nothing special
        return func(*args, **kwargs)
    return wrapper

# # Decide which decorator to use based on environment
# decorator_to_use = spaces.GPU() if is_running_in_hf_space() else no_op_decorator
# #duration=120

@spaces.GPU(duration=1)          # ← forces the detector to see a GPU-aware fn
def _warmup(): 
    print("Warming up...")




# if is_running_in_hf_space():
@spaces.GPU(duration=30)
def create_embeddings_30(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE)

@spaces.GPU(duration=59)
def create_embeddings_59(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE)

@spaces.GPU(duration=120)
def create_embeddings_120(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE)

@spaces.GPU(duration=299)
def create_embeddings_299(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE)
    


# else:
def create_embeddings(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE)



# ADDED (custom CSV coloring support): central plot-type constants.
PLOT_TYPE_NO_SPECIAL = "No special coloring"
PLOT_TYPE_TIME_BASED = "Time-based coloring"
PLOT_TYPE_CATEGORICAL = "Categorical coloring"
BASE_PLOT_TYPE_CHOICES = [
    PLOT_TYPE_NO_SPECIAL,
    PLOT_TYPE_TIME_BASED,
    PLOT_TYPE_CATEGORICAL,
]

# ADDED (local stability): smaller embedding batches for CPU-only local runs.
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_CACHE_PATH = Path("./cache/embedding_cache.pkl")
# ADDED (measure-size scaling): exponential scaling configuration.
MEASURE_BUBBLE_SIZE_BASE = 2.5
MEASURE_BUBBLE_SIZE_MAX = 24.0
MEASURE_BUBBLE_EXP_FACTOR = 5.0
# ADDED (measure semantics): keywords used to detect proportion-like numeric columns.
PROPORTION_KEYWORDS = {
    "proportion",
    "ratio",
    "share",
    "fraction",
    "percent",
    "percentage",
    "pct",
}
# ADDED (measure semantics): columns containing these fragments should always use data-range scaling.
FORCE_DATA_RANGE_KEYWORDS = {
    "banned words",
}
# ADDED (standalone HTML export): cache remote JS/CSS/font assets for offline single-file HTML.
STANDALONE_HTML_ASSET_CACHE_DIR = Path("./cache/standalone_html_assets")
STANDALONE_HTML_FETCH_TIMEOUT_SECONDS = 30


def encode_local_image_to_data_uri(image_path):
    """Encode a local image file as a data URI for standalone HTML export."""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "application/octet-stream")
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


# ADDED (standalone HTML export): fetch and cache remote assets used by datamapplot HTML.
def fetch_remote_asset_with_cache(url):
    STANDALONE_HTML_ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    asset_path = STANDALONE_HTML_ASSET_CACHE_DIR / f"{cache_key}.bin"
    meta_path = STANDALONE_HTML_ASSET_CACHE_DIR / f"{cache_key}.json"

    if asset_path.exists():
        try:
            with open(asset_path, "rb") as asset_file:
                data = asset_file.read()
            content_type = ""
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    content_type = json.load(meta_file).get("content_type", "")
            return data, content_type
        except Exception as e:
            print(f"Warning: Failed reading cached standalone asset for {url}: {e}")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=STANDALONE_HTML_FETCH_TIMEOUT_SECONDS) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()

    try:
        with open(asset_path, "wb") as asset_file:
            asset_file.write(data)
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump({"url": url, "content_type": content_type}, meta_file)
    except Exception as e:
        print(f"Warning: Failed writing cached standalone asset for {url}: {e}")

    return data, content_type


# ADDED (standalone HTML export): normalize remote asset bytes into text.
def decode_remote_text_asset(data):
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ADDED (standalone HTML export): convert URLs inside CSS to data-URIs (fonts/images).
def inline_remote_urls_in_css(css_text):
    css_url_pattern = re.compile(r"url\((['\"]?)(https?://[^)\"']+)\1\)", flags=re.IGNORECASE)
    inlined_count = 0
    failed_count = 0

    def replace_css_url(match):
        nonlocal inlined_count, failed_count
        remote_url = match.group(2)
        try:
            data, content_type = fetch_remote_asset_with_cache(remote_url)
            guessed_type, _ = mimetypes.guess_type(urllib.parse.urlparse(remote_url).path)
            mime_type = content_type or guessed_type or "application/octet-stream"
            encoded = base64.b64encode(data).decode("ascii")
            inlined_count += 1
            return f"url(data:{mime_type};base64,{encoded})"
        except Exception as e:
            print(f"Warning: Failed to inline CSS asset {remote_url}: {e}")
            failed_count += 1
            return match.group(0)

    inlined_css = css_url_pattern.sub(replace_css_url, css_text)
    return inlined_css, inlined_count, failed_count


# ADDED (standalone HTML export): inline remote JS/CSS so the downloaded visualization is one file.
def make_html_fully_standalone(html_path):
    html_path = Path(html_path)
    html_text = html_path.read_text(encoding="utf-8", errors="replace")

    summary = {
        "scripts_inlined": 0,
        "stylesheets_inlined": 0,
        "css_assets_inlined": 0,
        "assets_failed": 0,
        "links_removed": 0,
    }

    # Remove remote preconnect/preload hints. They are unnecessary for offline HTML.
    html_text, removed_preconnect = re.subn(
        r"<link[^>]+rel=['\"]preconnect['\"][^>]*>\s*",
        "",
        html_text,
        flags=re.IGNORECASE,
    )
    html_text, removed_preload = re.subn(
        r"<link[^>]+rel=['\"]preload['\"][^>]+href=['\"]https?://[^'\"]+['\"][^>]*>\s*",
        "",
        html_text,
        flags=re.IGNORECASE,
    )
    summary["links_removed"] += removed_preconnect + removed_preload

    script_pattern = re.compile(
        r"<script(?P<attrs>[^>]*?)\s+src=(?P<quote>['\"])(?P<url>https?://[^'\"]+)(?P=quote)(?P<rest>[^>]*)>\s*</script>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_script(match):
        remote_url = match.group("url")
        try:
            data, _ = fetch_remote_asset_with_cache(remote_url)
            js_text = decode_remote_text_asset(data).replace("</script>", "<\\/script>")
            summary["scripts_inlined"] += 1
            return f"<script data-original-src=\"{html_lib.escape(remote_url, quote=True)}\">\n{js_text}\n</script>"
        except Exception as e:
            print(f"Warning: Failed to inline script asset {remote_url}: {e}")
            summary["assets_failed"] += 1
            return match.group(0)

    html_text = script_pattern.sub(replace_script, html_text)

    stylesheet_pattern = re.compile(
        r"<link(?P<attrs>[^>]*?)\s+href=(?P<quote>['\"])(?P<url>https?://[^'\"]+)(?P=quote)(?P<rest>[^>]*)>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_stylesheet(match):
        full_tag = match.group(0)
        if "stylesheet" not in full_tag.lower():
            return full_tag

        remote_url = match.group("url")
        try:
            data, _ = fetch_remote_asset_with_cache(remote_url)
            css_text = decode_remote_text_asset(data)
            css_text, css_inlined_count, css_failed_count = inline_remote_urls_in_css(css_text)
            css_text = css_text.replace("</style>", "<\\/style>")
            summary["stylesheets_inlined"] += 1
            summary["css_assets_inlined"] += css_inlined_count
            summary["assets_failed"] += css_failed_count
            return (
                f"<style data-original-href=\"{html_lib.escape(remote_url, quote=True)}\">\n"
                f"{css_text}\n"
                "</style>"
            )
        except Exception as e:
            print(f"Warning: Failed to inline stylesheet asset {remote_url}: {e}")
            summary["assets_failed"] += 1
            return full_tag

    html_text = stylesheet_pattern.sub(replace_stylesheet, html_text)

    html_path.write_text(html_text, encoding="utf-8")
    return summary

# ADDED (custom CSV coloring support): columns that should not be offered as extra coloring fields.
EXCLUDED_CUSTOM_COLOR_COLUMNS = {
    "title",
    "abstract",
    "abtract",
    "id",
    "doi",
    "publication_year",
    "abstract_inverted_index",
    "query_index",
    "parsed_publication",
    "parsed_field",
    "color",
    "x",
    "y",
}


# ADDED (custom CSV coloring support): normalize Gradio file object/path handling.
def resolve_uploaded_file_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    return getattr(uploaded_file, "name", None)


# ADDED (custom CSV coloring support): derive extra column names usable for plot coloring.
def extract_custom_coloring_columns(df_columns):
    custom_columns = []
    seen_lower = set()
    for col in df_columns:
        col_name = str(col).strip()
        col_name_lower = col_name.lower()
        if col_name_lower in EXCLUDED_CUSTOM_COLOR_COLUMNS:
            continue
        if col_name_lower in seen_lower:
            continue
        seen_lower.add(col_name_lower)
        custom_columns.append(col_name)
    return custom_columns


# ADDED (custom CSV coloring support): build dynamic plot-color choices from uploaded CSV headers.
def get_plot_coloring_choices_for_file(csv_upload):
    choices = list(BASE_PLOT_TYPE_CHOICES)
    file_path = resolve_uploaded_file_path(csv_upload)
    if not file_path:
        return choices

    try:
        header_df = pd.read_csv(file_path, nrows=0)
        custom_columns = extract_custom_coloring_columns(header_df.columns.tolist())
        choices.extend(custom_columns)
    except Exception as e:
        print(f"Warning: Could not inspect uploaded CSV columns for plot coloring choices: {e}")

    return choices


def format_hover_measure_value(value):
    """Format measure values consistently for tooltip display."""
    if pd.isna(value):
        return "Missing"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


# ADDED (embedding cache): compact key derived from model + project id/text.
def build_embedding_cache_key(model_name, row_id, title, abstract):
    title_text = "" if pd.isna(title) else str(title)
    abstract_text = "" if pd.isna(abstract) else str(abstract)
    text_hash = hashlib.sha256(f"{title_text}\n{abstract_text}".encode("utf-8")).hexdigest()
    id_text = "" if pd.isna(row_id) else str(row_id).strip()
    if id_text and id_text.lower() not in {"nan", "none"}:
        return f"{model_name}::id::{id_text}::text::{text_hash}"
    return f"{model_name}::text::{text_hash}"


def load_embedding_cache():
    """Load cached embeddings from disk."""
    if not EMBEDDING_CACHE_PATH.exists():
        return {}
    try:
        with open(EMBEDDING_CACHE_PATH, "rb") as cache_file:
            cache_obj = pickle.load(cache_file)
        if isinstance(cache_obj, dict):
            return cache_obj
    except Exception as e:
        print(f"Warning: Could not load embedding cache ({EMBEDDING_CACHE_PATH}): {e}")
    return {}


def save_embedding_cache(cache_dict):
    """Persist cached embeddings to disk."""
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDING_CACHE_PATH, "wb") as cache_file:
            pickle.dump(cache_dict, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: Could not save embedding cache ({EMBEDDING_CACHE_PATH}): {e}")


def encode_text_batch(texts_to_embedd, user_type):
    """Encode a list of texts using the same environment-aware strategy as before."""
    if is_running_in_hf_space():
        if len(texts_to_embedd) < 2000:
            return create_embeddings_30(texts_to_embedd)
        if len(texts_to_embedd) < 4000 or user_type == "anonymous":
            return create_embeddings_59(texts_to_embedd)
        if len(texts_to_embedd) < 8000:
            return create_embeddings_120(texts_to_embedd)
        return create_embeddings_299(texts_to_embedd)
    return create_embeddings(texts_to_embedd)


def create_embeddings_with_cache(records_df, user_type):
    """
    Create embeddings and reuse previously cached vectors when possible.
    """
    records_for_cache = records_df.reset_index(drop=True)
    embedding_cache = load_embedding_cache()
    cache_keys = [
        build_embedding_cache_key(MODEL_NAME, row["id"], row["title"], row["abstract"])
        for _, row in records_for_cache.iterrows()
    ]

    cached_embeddings = [embedding_cache.get(key) for key in cache_keys]
    missing_indices = [idx for idx, emb in enumerate(cached_embeddings) if emb is None]

    print(
        f"Embedding cache hit rate: {len(records_for_cache) - len(missing_indices)}/"
        f"{len(records_for_cache)}"
    )

    if missing_indices:
        missing_texts = [
            f"{records_for_cache.at[idx, 'title']} {records_for_cache.at[idx, 'abstract']}"
            for idx in missing_indices
        ]
        missing_embeddings = np.asarray(
            encode_text_batch(missing_texts, user_type),
            dtype=np.float32
        )
        if missing_embeddings.ndim == 1:
            missing_embeddings = np.expand_dims(missing_embeddings, axis=0)

        for array_idx, row_idx in enumerate(missing_indices):
            embedding_vec = missing_embeddings[array_idx]
            cached_embeddings[row_idx] = embedding_vec
            embedding_cache[cache_keys[row_idx]] = embedding_vec

        save_embedding_cache(embedding_cache)
        print(f"Cached {len(missing_indices)} new embeddings")

    embeddings = np.asarray(cached_embeddings, dtype=np.float32)
    return embeddings


def predict(request: gr.Request, text_input, sample_size_slider, reduce_sample_checkbox, 
           sample_reduction_method, plot_type_dropdown, 
           locally_approximate_publication_date_checkbox, 
           download_csv_checkbox, download_png_checkbox, citation_graph_checkbox, 
           standalone_html_checkbox,
           csv_upload, highlight_color, selected_colormap_name, seed_value,
           progress=gr.Progress()):
    """
    Main prediction pipeline that processes OpenAlex queries and creates visualizations.
    
    Args:
        request (gr.Request): Gradio request object
        text_input (str): OpenAlex query URL
        sample_size_slider (int): Maximum number of samples to process
        reduce_sample_checkbox (bool): Whether to reduce sample size
        sample_reduction_method (str): Method for sample reduction ("Random" or "Order of Results")
        plot_type_dropdown (str): Type of plot coloring ("No special coloring", "Time-based coloring", "Categorical coloring")
        locally_approximate_publication_date_checkbox (bool): Whether to approximate publication date locally before plotting.
        download_csv_checkbox (bool): Whether to download CSV data
        download_png_checkbox (bool): Whether to download PNG data
        citation_graph_checkbox (bool): Whether to add citation graph
        standalone_html_checkbox (bool): Whether to export interactive HTML as fully standalone offline file
        csv_upload (str): Path to uploaded CSV file
        highlight_color (str): Color for highlighting points
        selected_colormap_name (str): Name of the selected colormap for time-based coloring
        progress (gr.Progress): Gradio progress tracker
    
    Returns:
        tuple: (link to visualization, iframe HTML)
    """
    # Initialize start_time at the beginning of the function
    start_time = time.time()
    
    # CHANGED (custom CSV coloring support): identify selected coloring mode.
    plot_time_checkbox = plot_type_dropdown == PLOT_TYPE_TIME_BASED
    treat_as_categorical_checkbox = plot_type_dropdown == PLOT_TYPE_CATEGORICAL
    custom_color_column = (
        plot_type_dropdown
        if plot_type_dropdown not in BASE_PLOT_TYPE_CHOICES
        else None
    )
    
    # Initialize variables used later across branches
    urls = []
    query_indices = []
    # Default to anonymous unless we can positively identify the user via a valid token
    user_type = "anonymous"
    # ADDED (custom CSV coloring support): shared legend/color state for later plot steps.
    continuous_color_values = None
    continuous_color_label = None
    continuous_color_cmap = None
    continuous_color_vmin = None
    continuous_color_vmax = None
    continuous_color_ticks = None
    custom_categorical_color_mapping = None
    selected_measure_label_for_hover = None
    selected_measure_values_for_hover = None
    # ADDED (measure-size scaling): optional query bubble scaling driven by numeric measure values.
    measure_scaled_query_point_sizes = None
    measure_scaled_point_radius_max_pixels = 5
    
    # Helper function to generate error responses
    def create_error_response(error_message):
        return [
            error_message,
            gr.DownloadButton(label="Download Interactive Visualization", value='html_file_path', visible=False),
            gr.DownloadButton(label="Download CSV Data", value='csv_file_path', visible=False),
            gr.DownloadButton(label="Download Static Plot", value='png_file_path', visible=False),
            gr.Button(visible=False)
        ]
    
    # Get the authentication token
    if is_running_in_hf_space():
        token = _get_token(request)
        try:
            if token and '.' in token:
                payload_part = token.split('.')[1]
                payload_part = f"{payload_part}{'=' * ((4 - len(payload_part) % 4) % 4)}"
                payload = json.loads(base64.urlsafe_b64decode(payload_part).decode())
                print(payload)
                user = payload.get('user')
                if user is None:
                    user_type = "anonymous"
                elif '[pro]' in user:
                    user_type = "pro"
                else:
                    user_type = "registered"
            else:
                # No token available or malformed; treat as anonymous
                user_type = "anonymous"
        except Exception as e:
            # Any decoding/parsing error → anonymous fallback
            print(f"Warning: failed to parse auth token: {e}")
            user_type = "anonymous"
        print(f"User type: {user_type}")

    # Check if a file has been uploaded or if we need to use OpenAlex query.
    uploaded_file_path = resolve_uploaded_file_path(csv_upload)
    if uploaded_file_path is not None:
        print(f"Using uploaded file instead of OpenAlex query: {uploaded_file_path}")
        try:
            file_extension = os.path.splitext(uploaded_file_path)[1].lower()
            
            if file_extension == '.csv':
                # Read the CSV file
                records_df = pd.read_csv(uploaded_file_path)
                # CHANGED (cache busting): use a unique filename per run to avoid stale browser caching.
                filename_base = os.path.splitext(os.path.basename(uploaded_file_path))[0]
                filename = f"{filename_base}__{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # Check if this is a DOI-list CSV (single column, named 'doi' or similar)
                if (len(records_df.columns) == 1 and records_df.columns[0].lower() in ['doi', 'dois']):
                    from openalex_utils import get_records_from_dois
                    doi_list = records_df.iloc[:,0].dropna().astype(str).tolist()
                    print(f"Detected DOI list with {len(doi_list)} DOIs. Downloading records from OpenAlex...")
                    records_df = get_records_from_dois(doi_list)
                    filename = f"doilist_{len(doi_list)}"
                else:
                    # Convert *every* cell that looks like a serialized list/dict
                    def _try_parse_obj(cell):
                        if isinstance(cell, str):
                            txt = cell.strip()
                            if (txt.startswith('{') and txt.endswith('}')) or (txt.startswith('[') and txt.endswith(']')):
                                # Try JSON first
                                try:
                                    return json.loads(txt)
                                except Exception:
                                    pass
                                # Fallback to Python-repr (single quotes etc.)
                                try:
                                    return ast.literal_eval(txt)
                                except Exception:
                                    pass
                        return cell

                    # CHANGED (robustness): support both newer and older pandas APIs.
                    try:
                        records_df = records_df.map(_try_parse_obj)
                    except AttributeError:
                        records_df = records_df.applymap(_try_parse_obj)
                    print(records_df.head())
                
            else:
                error_message = "Error: Unsupported file type. Please upload a CSV file."
                return create_error_response(error_message)
                
            # CHANGED (custom CSV support): normalize uploaded data into a robust internal schema.
            records_df = process_records_to_df(records_df)
            print(f"Records after normalization/deduplication: {len(records_df)}")
            
            # CHANGED (custom CSV support): only title + abstract are required for embedding.
            required_columns = ['title', 'abstract']
            missing_columns = [col for col in required_columns if col not in records_df.columns]
            
            if missing_columns:
                error_message = f"Error: Uploaded file is missing required columns: {', '.join(missing_columns)}"
                return create_error_response(error_message)
                
            print(f"Successfully loaded {len(records_df)} records from uploaded file")
            progress(0.2, desc="Processing uploaded data...")
            
            # For uploaded files, set all records to query_index 0
            records_df['query_index'] = 0
            
        except Exception as e:
            error_message = f"Error processing uploaded file: {str(e)}"
            return create_error_response(error_message)
    else:
        # Check if input is empty or whitespace
        print(f"Input: {text_input}")
        if not text_input or text_input.isspace():
            error_message = "Error: Please enter a valid OpenAlex URL in the 'OpenAlex-search URL'-field or upload a CSV file"
            return create_error_response(error_message)

        print('Starting data projection pipeline')
        progress(0.1, desc="Starting...")

        # Split input into multiple URLs if present
        urls = [url.strip() for url in text_input.split(';')]
        records = []
        query_indices = []  # Track which query each record comes from
        total_query_length = 0
        expected_download_count = 0  # Track expected number of records to download for progress
        
        # Use first URL for filename
        first_query, first_params = openalex_url_to_pyalex_query(urls[0])
        filename = openalex_url_to_filename(urls[0])
        print(f"Filename: {filename}")

        # Process each URL
        for i, url in enumerate(urls):
            query, params = openalex_url_to_pyalex_query(url)
            query_length = query.count()
            total_query_length += query_length
            
            # Calculate expected download count for this query
            if reduce_sample_checkbox and sample_reduction_method == "First n samples":
                expected_for_this_query = min(sample_size_slider, query_length)
            elif reduce_sample_checkbox and sample_reduction_method == "n random samples":
                expected_for_this_query = min(sample_size_slider, query_length)
            else:  # "All"
                expected_for_this_query = query_length
            
            expected_download_count += expected_for_this_query
            print(f'Requesting {query_length} entries from query {i+1}/{len(urls)} (expecting to download {expected_for_this_query})...')
            
            # Use PyAlex sampling for random samples - much more efficient!
            if reduce_sample_checkbox and sample_reduction_method == "n random samples":
                # Use PyAlex's built-in sample method for efficient server-side sampling
                target_size = min(sample_size_slider, query_length)
                try:
                    seed_int = int(seed_value) if seed_value.strip() else 42
                except ValueError:
                    seed_int = 42
                    print(f"Invalid seed value '{seed_value}', using default: 42")
                
                print(f'Attempting PyAlex sampling: {target_size} from {query_length} (seed={seed_int})')
                
                try:
                    # Check if PyAlex sample method exists and works
                    if hasattr(query, 'sample'):
                        sampled_records = []
                        seen_ids = set()  # Track IDs to avoid duplicates
                        
                        # If target_size > 10k, do batched sampling
                        if target_size > 10000:
                            batch_size = 9998  # Use 9998 to stay safely under 10k limit
                            remaining = target_size
                            batch_num = 1
                            
                            print(f'Target size {target_size} > 10k, using batched sampling with batch size {batch_size}')
                            
                            while remaining > 0 and len(sampled_records) < target_size:
                                current_batch_size = min(batch_size, remaining)
                                batch_seed = seed_int + batch_num  # Different seed for each batch
                                
                                print(f'Batch {batch_num}: requesting {current_batch_size} samples (seed={batch_seed})')
                                
                                # Sample this batch
                                batch_query = query.sample(current_batch_size, seed=batch_seed)
                                
                                batch_records = []
                                batch_count = 0
                                for page in batch_query.paginate(per_page=200, method='page', n_max=None):
                                    for record in page:
                                        # Check for duplicates using OpenAlex ID
                                        record_id = record.get('id', '')
                                        if record_id not in seen_ids:
                                            seen_ids.add(record_id)
                                            batch_records.append(record)
                                            batch_count += 1
                                
                                sampled_records.extend(batch_records)
                                remaining -= len(batch_records)
                                batch_num += 1
                                
                                print(f'Batch {batch_num-1} complete: got {len(batch_records)} unique records ({len(sampled_records)}/{target_size} total)')
                                
                                progress(0.1 + (0.15 * len(sampled_records) / target_size), 
                                        desc=f"Batched sampling from query {i+1}/{len(urls)}... ({len(sampled_records)}/{target_size})")
                                
                                # Safety check to avoid infinite loops
                                if batch_num > 20:  # Max 20 batches (should handle up to ~200k samples)
                                    print("Warning: Maximum batch limit reached, stopping sampling")
                                    break
                        else:
                            # Single batch sampling for <= 10k
                            sampled_query = query.sample(target_size, seed=seed_int)
                            
                            records_count = 0
                            for page in sampled_query.paginate(per_page=200, method='page', n_max=None):
                                for record in page:
                                    sampled_records.append(record)
                                    records_count += 1
                                    progress(0.1 + (0.15 * records_count / target_size), 
                                            desc=f"Getting sampled data from query {i+1}/{len(urls)}... ({records_count}/{target_size})")
                        
                        print(f'PyAlex sampling successful: got {len(sampled_records)} records (requested {target_size})')
                    else:
                        raise AttributeError("sample method not available")
                        
                except Exception as e:
                    print(f"PyAlex sampling failed ({e}), using fallback method...")
                    
                    # Fallback: get all records and sample manually
                    all_records = []
                    records_count = 0
                    
                    # Use page pagination for fallback method
                    for page in query.paginate(per_page=200, method='page', n_max=None):
                        for record in page:
                            all_records.append(record)
                            records_count += 1
                            progress(0.1 + (0.15 * records_count / query_length), 
                                    desc=f"Downloading for sampling from query {i+1}/{len(urls)}...")
                            
                    # Now sample manually
                    if len(all_records) > target_size:
                        import random
                        random.seed(seed_int)
                        sampled_records = random.sample(all_records, target_size)
                    else:
                        sampled_records = all_records
                        
                    print(f'Fallback sampling: got {len(sampled_records)} from {len(all_records)} total')
                
                # Add the sampled records
                for idx, record in enumerate(sampled_records):
                    records.append(record)
                    query_indices.append(i)
                    # Safe progress calculation
                    if expected_download_count > 0:
                        progress_val = 0.1 + (0.2 * len(records) / expected_download_count)
                    else:
                        progress_val = 0.1
                    progress(progress_val, desc=f"Processing sampled data from query {i+1}/{len(urls)}...")
            else:
                # Keep existing logic for "First n samples" and "All"
                target_size = sample_size_slider if reduce_sample_checkbox and sample_reduction_method == "First n samples" else query_length
                records_per_query = 0
                
                print(f"Query {i+1}: target_size={target_size}, query_length={query_length}, method={sample_reduction_method}")
                
                should_break_current_query = False
                # For "First n samples", limit the maximum records fetched to avoid over-downloading
                max_records_to_fetch = target_size if reduce_sample_checkbox and sample_reduction_method == "First n samples" else None
                for page in query.paginate(per_page=200, method='page', n_max=max_records_to_fetch):
                    # Add retry mechanism for processing each page
                    max_retries = 5
                    base_wait_time = 1  # Starting wait time in seconds
                    exponent = 1.5  # Exponential factor
                    
                    for retry_attempt in range(max_retries):
                        try:
                            for record in page:
                                # Safety check: don't process if we've already reached target
                                if reduce_sample_checkbox and sample_reduction_method == "First n samples" and records_per_query >= target_size:
                                    print(f"Reached target size before processing: {records_per_query}/{target_size}, breaking from download")
                                    should_break_current_query = True
                                    break
                                    
                                records.append(record)
                                query_indices.append(i)  # Track which query this record comes from
                                records_per_query += 1
                                # Safe progress calculation
                                if expected_download_count > 0:
                                    progress_val = 0.1 + (0.2 * len(records) / expected_download_count)
                                else:
                                    progress_val = 0.1
                                progress(progress_val, desc=f"Getting data from query {i+1}/{len(urls)}...")
                                
                                if reduce_sample_checkbox and sample_reduction_method == "First n samples" and records_per_query >= target_size:
                                    print(f"Reached target size: {records_per_query}/{target_size}, breaking from download")
                                    should_break_current_query = True
                                    break
                            # If we get here without an exception, break the retry loop
                            break
                        except Exception as e:
                            print(f"Error processing page: {e}")
                            if retry_attempt < max_retries - 1:
                                wait_time = base_wait_time * (exponent ** retry_attempt) + random.random()
                                print(f"Retrying in {wait_time:.2f} seconds (attempt {retry_attempt + 1}/{max_retries})...")
                                time.sleep(wait_time)
                            else:
                                print(f"Maximum retries reached. Continuing with next page.")
                                
                        # Break out of retry loop if we've reached target
                        if should_break_current_query:
                            break
                
                if should_break_current_query:
                    print(f"Successfully downloaded target size for query {i+1}, moving to next query")
                    # Continue to next query instead of breaking the entire query loop
                    continue
            # Continue to next query - don't break out of the main query loop
        print(f"Query completed in {time.time() - start_time:.2f} seconds")
        print(f"Total records collected: {len(records)}")
        print(f"Expected to download: {expected_download_count}")
        print(f"Available from all queries: {total_query_length}")
        print(f"Sample method used: {sample_reduction_method}")
        print(f"Reduce sample enabled: {reduce_sample_checkbox}")
        if sample_reduction_method == "n random samples":
            print(f"Seed value: {seed_value}")

        # Process records
        processing_start = time.time()
        records_df = process_records_to_df(records)
        
        # Add query_index to the dataframe
        records_df['query_index'] = query_indices[:len(records_df)]

        
        if reduce_sample_checkbox and sample_reduction_method != "All" and sample_reduction_method != "n random samples":
            # Note: We skip "n random samples" here because PyAlex sampling is already done above
            sample_size = min(sample_size_slider, len(records_df))
            
            # Check if we have multiple queries for sampling logic
            urls = [url.strip() for url in text_input.split(';')] if text_input else ['']
            has_multiple_queries = len(urls) > 1 and uploaded_file_path is None
            
            # If using categorical coloring with multiple queries, sample each query independently
            if treat_as_categorical_checkbox and has_multiple_queries:
                # Sample the full sample_size from each query independently
                unique_queries = sorted(records_df['query_index'].unique())
                
                sampled_dfs = []
                for query_idx in unique_queries:
                    query_records = records_df[records_df['query_index'] == query_idx]
                    
                    # Apply the full sample size to each query (only for "First n samples")
                    current_sample_size = min(sample_size_slider, len(query_records))
                    
                    if sample_reduction_method == "First n samples":
                        sampled_query = query_records.iloc[:current_sample_size]
                    
                    sampled_dfs.append(sampled_query)
                    print(f"Query {query_idx+1}: sampled {len(sampled_query)} records from {len(query_records)} available")
                
                records_df = pd.concat(sampled_dfs, ignore_index=True)
                print(f"Total after independent sampling: {len(records_df)} records")
                print(f"Query distribution: {records_df['query_index'].value_counts().sort_index()}")
            else:
                # Original sampling logic for single query or non-categorical (only "First n samples" now)
                if sample_reduction_method == "First n samples":
                    records_df = records_df.iloc[:sample_size]
        print(f"Records processed in {time.time() - processing_start:.2f} seconds")
        
    print(query_indices)
    print(records_df)
    # Create embeddings - this happens regardless of data source
    embedding_start = time.time()
    progress(0.3, desc="Embedding Data...")
    embeddings = create_embeddings_with_cache(records_df, user_type)
        
    print(f"Embeddings created in {time.time() - embedding_start:.2f} seconds")

    # Project embeddings
    projection_start = time.time()
    progress(0.5, desc="Project into UMAP-embedding...")
    umap_embeddings = mapper.transform(embeddings)
    records_df[['x','y']] = umap_embeddings
    print(f"Projection completed in {time.time() - projection_start:.2f} seconds")

    # Prepare visualization data
    viz_prep_start = time.time()
    progress(0.6, desc="Preparing visualization data...")
    
    
    # Set up colors:
    
    basedata_df['color'] = '#ced4d211'
    
    # Convert highlight_color to hex if it isn't already
    if not highlight_color.startswith('#'):
        highlight_color = rgba_to_hex(highlight_color)
    highlight_color = rgba_to_hex(highlight_color)
    
    print('Highlight color:', highlight_color)
    
    # Check if we have multiple queries and categorical coloring is enabled.
    # Note: urls was already parsed earlier in the function.
    has_multiple_queries = len(urls) > 1 and uploaded_file_path is None

    # ADDED (custom CSV coloring robustness): resolve selected custom column name
    # case-insensitively and whitespace-normalized to avoid silent fallback.
    resolved_custom_color_column = None
    if custom_color_column:
        normalized_selected_column = re.sub(r"\s+", " ", str(custom_color_column).strip()).casefold()
        normalized_column_lookup = {
            re.sub(r"\s+", " ", str(col).strip()).casefold(): col
            for col in records_df.columns
        }
        resolved_custom_color_column = normalized_column_lookup.get(normalized_selected_column)
        if resolved_custom_color_column is None:
            print(
                f"Warning: Selected coloring column '{custom_color_column}' was not found in "
                f"uploaded data columns: {list(records_df.columns)}"
            )

    custom_color_column_exists = resolved_custom_color_column is not None

    if treat_as_categorical_checkbox and has_multiple_queries:
        # Use categorical coloring for multiple queries
        print("Using categorical coloring for multiple queries")
        
        # Get colors from selected colormap or use default categorical colors
        unique_queries = sorted(records_df['query_index'].unique())
        num_queries = len(unique_queries)
        
        if selected_colormap_name and selected_colormap_name.strip():
            try:
                # Use selected colormap to generate distinct colors
                categorical_cmap = plt.get_cmap(selected_colormap_name)
                # Sample colors evenly spaced across the colormap
                categorical_colors = [mcolors.to_hex(categorical_cmap(i / max(1, num_queries - 1))) 
                                    for i in range(num_queries)]
            except Exception as e:
                print(f"Warning: Could not load colormap '{selected_colormap_name}' for categorical coloring: {e}")
                # Fallback to default categorical colors
                categorical_colors = [
                        "#80418F",  # Plum
                        "#EDA958",  # Earth Yellow
                        "#F35264",  # Crayola Red
                        "#087CA7",  # Cerulean
                        "#FA826B",  # Salmon
                        "#475C8F",  # Navy Blue
                        "#579DA3",  # Moonstone Green
                        "#d61d22",  # Bright Red
                        "#97bb3c",  # Lime Green
                    ]
        else:
            # Use default categorical colors
            categorical_colors = [
                        "#80418F",  # Plum
                        "#EDA958",  # Earth Yellow
                        "#F35264",  # Crayola Red
                        "#087CA7",  # Cerulean
                        "#FA826B",  # Salmon
                        "#475C8F",  # Navy Blue
                        "#579DA3",  # Moonstone Green
                        "#d61d22",  # Bright Red
                        "#97bb3c",  # Lime Green
                    ]
        
        # Assign colors based on query_index
        query_color_map = {query_idx: categorical_colors[i % len(categorical_colors)] 
                          for i, query_idx in enumerate(unique_queries)}
        
        records_df['color'] = records_df['query_index'].map(query_color_map)
        
        # Add query_label for better identification
        records_df['query_label'] = records_df['query_index'].apply(lambda x: f"Query {x+1}")
        selected_measure_label_for_hover = "Query"
        selected_measure_values_for_hover = records_df['query_label']

    # ADDED (custom CSV coloring support): allow any uploaded CSV column to drive coloring.
    elif custom_color_column_exists:
        print(f"Using custom coloring column: '{resolved_custom_color_column}'")
        selected_measure_label_for_hover = resolved_custom_color_column

        # CHANGED (semantic coloring): fixed measure scale
        # 0.0 = green, 0.5 = yellow, 1.0 = red (values > 1 clipped to red).
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "measure_green_yellow_red",
            [(0.0, "#2ca25f"), (0.5, "#ffd92f"), (1.0, "#d73027")]
        )

        custom_values = records_df[resolved_custom_color_column]
        numeric_values = pd.to_numeric(custom_values, errors='coerce')

        # ADDED (custom CSV coloring robustness): parse common numeric string formats
        # like "58,3%" when direct numeric conversion fails.
        normalized_numeric_strings = (
            custom_values.astype(str)
            .str.strip()
            .str.replace('%', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        numeric_values_from_normalized = pd.to_numeric(normalized_numeric_strings, errors='coerce')
        if numeric_values_from_normalized.notna().sum() > numeric_values.notna().sum():
            numeric_values = numeric_values_from_normalized
        numeric_ratio = float(numeric_values.notna().mean()) if len(numeric_values) > 0 else 0.0

        # Heuristic: treat as continuous when enough values can be parsed as numeric.
        if numeric_values.notna().any() and numeric_ratio >= 0.6:
            valid_numeric = numeric_values.dropna()
            numeric_for_colors = numeric_values.fillna(valid_numeric.median())
            numeric_for_colors_float = numeric_for_colors.astype(float)

            # CHANGED (semantic coloring): detect whether the selected numeric column is a proportion.
            normalized_label = str(resolved_custom_color_column).casefold()
            has_proportion_keyword = any(keyword in normalized_label for keyword in PROPORTION_KEYWORDS)
            force_data_range_scaling = any(keyword in normalized_label for keyword in FORCE_DATA_RANGE_KEYWORDS)
            within_zero_one_ratio = float(
                ((valid_numeric >= 0.0) & (valid_numeric <= 1.0)).mean()
            ) if len(valid_numeric) > 0 else 0.0
            is_proportion_measure = (has_proportion_keyword or within_zero_one_ratio >= 0.98) and not force_data_range_scaling

            if is_proportion_measure:
                # Proportion-like: preserve semantic interpretation with fixed 0..1 clipping.
                color_scale_values = np.clip(numeric_for_colors_float, 0.0, 1.0)
                continuous_color_vmin = 0.0
                continuous_color_vmax = 1.0
                continuous_color_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
                print(
                    f"Detected proportion-like measure '{resolved_custom_color_column}' "
                    "(using fixed 0..1 legend scale)."
                )
            else:
                if force_data_range_scaling:
                    print(
                        f"Detected forced data-range measure '{resolved_custom_color_column}' "
                        "(overriding proportion heuristic)."
                    )
                # Non-proportion numeric measure: scale color over the full observed data range.
                data_min = float(valid_numeric.min())
                data_max = float(valid_numeric.max())
                if data_min == data_max:
                    color_scale_values = np.full(len(numeric_for_colors_float), 0.5)
                else:
                    color_scale_values = (numeric_for_colors_float - data_min) / (data_max - data_min)
                continuous_color_vmin = data_min
                continuous_color_vmax = data_max
                continuous_color_ticks = None
                print(
                    f"Detected general numeric measure '{resolved_custom_color_column}' "
                    f"(using data-driven legend scale: {data_min:.4f}..{data_max:.4f})."
                )

            color_scale_values = np.clip(np.asarray(color_scale_values, dtype=float), 0.0, 1.0)
            records_df['color'] = [mcolors.to_hex(custom_cmap(v)) for v in color_scale_values]
            records_df.loc[numeric_values.isna(), 'color'] = '#bcbcbc'

            # CHANGED (measure-size scaling): always scale size from normalized color-scale values.
            # 0 -> default bubble size, 1 -> largest bubble.
            # Use an exponential curve for stronger perceived separation.
            exp_scaled_values = np.expm1(color_scale_values * MEASURE_BUBBLE_EXP_FACTOR) / np.expm1(
                MEASURE_BUBBLE_EXP_FACTOR
            )
            measure_scaled_query_point_sizes = (
                MEASURE_BUBBLE_SIZE_BASE
                + (exp_scaled_values * (MEASURE_BUBBLE_SIZE_MAX - MEASURE_BUBBLE_SIZE_BASE))
            )
            measure_scaled_query_point_sizes = np.asarray(measure_scaled_query_point_sizes, dtype=float)
            measure_scaled_query_point_sizes[np.asarray(numeric_values.isna())] = MEASURE_BUBBLE_SIZE_BASE
            measure_scaled_point_radius_max_pixels = MEASURE_BUBBLE_SIZE_MAX
            print(
                "Measure size scaling active: "
                f"query_size_min={measure_scaled_query_point_sizes.min():.3f}, "
                f"query_size_max={measure_scaled_query_point_sizes.max():.3f}"
            )

            continuous_color_values = numeric_values
            continuous_color_label = resolved_custom_color_column
            continuous_color_cmap = custom_cmap
            selected_measure_values_for_hover = numeric_values
        else:
            categorical_values = custom_values.fillna("Missing").astype(str)
            unique_categories = sorted(categorical_values.unique())
            num_categories = len(unique_categories)

            if selected_colormap_name and selected_colormap_name.strip():
                try:
                    categorical_cmap = plt.get_cmap(selected_colormap_name)
                except Exception as e:
                    print(f"Warning: Could not load colormap '{selected_colormap_name}' for categorical coloring: {e}")
                    categorical_cmap = plt.get_cmap("tab20")
            else:
                categorical_cmap = plt.get_cmap("tab20")

            category_colors = [
                mcolors.to_hex(categorical_cmap(i / max(1, num_categories - 1)))
                for i in range(num_categories)
            ]
            custom_categorical_color_mapping = {
                category: category_colors[i % len(category_colors)]
                for i, category in enumerate(unique_categories)
            }
            records_df['color'] = categorical_values.map(custom_categorical_color_mapping)
            records_df['custom_category_label'] = categorical_values
            selected_measure_values_for_hover = categorical_values

    elif plot_time_checkbox:
        if 'publication_year' not in records_df.columns:
            print("Warning: 'publication_year' missing, falling back to highlight color.")
            records_df['color'] = highlight_color
        else:
            # Use selected colormap if provided, otherwise default to haline.
            if selected_colormap_name and selected_colormap_name.strip():
                try:
                    time_cmap = plt.get_cmap(selected_colormap_name)
                except Exception as e:
                    print(f"Warning: Could not load colormap '{selected_colormap_name}': {e}")
                    time_cmap = colormaps.haline
            else:
                time_cmap = colormaps.haline

            years = pd.to_numeric(records_df['publication_year'], errors='coerce')
            if not years.notna().any():
                print("Warning: 'publication_year' has no numeric values, falling back to highlight color.")
                records_df['color'] = highlight_color
            elif not locally_approximate_publication_date_checkbox:
                years_for_colors = years.fillna(years.dropna().median())
                year_min = float(years_for_colors.min())
                year_max = float(years_for_colors.max())

                if year_min == year_max:
                    normalized_years = np.full(len(years_for_colors), 0.5)
                else:
                    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)
                    normalized_years = norm(years_for_colors)

                records_df['color'] = [mcolors.to_hex(time_cmap(y)) for y in normalized_years]
                records_df.loc[years.isna(), 'color'] = '#bcbcbc'

                continuous_color_values = years
                continuous_color_label = "Publication Year"
                continuous_color_cmap = time_cmap
                selected_measure_label_for_hover = "Publication Year"
                selected_measure_values_for_hover = years
            else:
                years_for_knn = years.fillna(years.dropna().median())
                n_neighbors = min(10, len(records_df))

                if n_neighbors < 2:
                    records_df['color'] = highlight_color
                else:
                    nn = NearestNeighbors(n_neighbors=n_neighbors)
                    nn.fit(umap_embeddings)
                    _, indices = nn.kneighbors(umap_embeddings)

                    # Calculate local average publication year for each point.
                    local_years = np.array([
                        np.mean(years_for_knn.iloc[idx])
                        for idx in indices
                    ])

                    local_year_min = float(local_years.min())
                    local_year_max = float(local_years.max())
                    if local_year_min == local_year_max:
                        normalized_local_years = np.full(len(local_years), 0.5)
                    else:
                        norm = mcolors.Normalize(vmin=local_year_min, vmax=local_year_max)
                        normalized_local_years = norm(local_years)

                    records_df['color'] = [mcolors.to_hex(time_cmap(y)) for y in normalized_local_years]

                    continuous_color_values = pd.Series(local_years, index=records_df.index)
                    continuous_color_label = "Approx. Year"
                    continuous_color_cmap = time_cmap
                    selected_measure_label_for_hover = "Approx. Year"
                    selected_measure_values_for_hover = pd.Series(local_years, index=records_df.index)
    else:
        # No special coloring - use highlight color
        records_df['color'] = highlight_color
        selected_measure_label_for_hover = None
        selected_measure_values_for_hover = None
                        
    # ADDED (hover context): include selected measure under title in query point tooltips.
    if 'title' not in basedata_df.columns:
        basedata_df['title'] = ''
    basedata_df['hover_text_custom'] = basedata_df['title'].fillna('').astype(str).apply(html_lib.escape)

    records_df['hover_text_custom'] = records_df['title'].fillna('').astype(str).apply(html_lib.escape)
    if selected_measure_label_for_hover and selected_measure_values_for_hover is not None:
        hover_measure_series = pd.Series(selected_measure_values_for_hover, index=records_df.index)
        formatted_measure_values = hover_measure_series.apply(format_hover_measure_value).apply(
            lambda x: html_lib.escape(str(x))
        )
        measure_label_html = html_lib.escape(str(selected_measure_label_for_hover))
        records_df['hover_text_custom'] = [
            f"{title}<br>{measure_label_html}: {measure_value}"
            for title, measure_value in zip(records_df['hover_text_custom'], formatted_measure_values)
        ]

    stacked_df = pd.concat([basedata_df, records_df], axis=0, ignore_index=True)
    stacked_df = stacked_df.fillna("Unlabelled")
    stacked_df['parsed_field'] = [get_field(row) for ix, row in stacked_df.iterrows()]
    
    # ADDED (measure-size scaling): default query bubble size unless overridden by numeric measure scaling.
    if measure_scaled_query_point_sizes is None:
        query_point_sizes = np.full(len(records_df), 2.5, dtype=float)
    else:
        query_point_sizes = measure_scaled_query_point_sizes

    # Create marker size array: basemap points fixed, query points optionally measure-scaled.
    marker_sizes = np.concatenate([
        np.full(len(basedata_df), 1.),  # Basemap points
        query_point_sizes
    ])
    
    # CHANGED (tooltip HTML mode): always provide extra_point_data so datamapplot can render
    # hover_text_html_template as HTML (enables actual line breaks instead of raw "<br>").
    extra_data = pd.DataFrame({"_tooltip_index": np.arange(len(stacked_df), dtype=int)})
    click_handler = None
    if 'doi' in stacked_df.columns:
        extra_data['doi'] = stacked_df['doi'].fillna('')
        click_handler = "window.open(`{doi}`)"
    print(f"Visualization data prepared in {time.time() - viz_prep_start:.2f} seconds")
    
    # Prepare file paths
    html_file_name = f"{filename}.html"
    html_file_path = static_dir / html_file_name
    csv_file_path = static_dir / f"{filename}.csv"
    png_file_path = static_dir / f"{filename}.png"
    
    if citation_graph_checkbox:
        citation_graph_start = time.time()
        citation_graph = create_citation_graph(records_df)
        graph_file_name = f"{filename}_citation_graph.jpg"
        graph_file_path = static_dir / graph_file_name
        draw_citation_graph(citation_graph,path=graph_file_path,bundle_edges=True,
                            min_max_coordinates=[np.min(stacked_df['x']),np.max(stacked_df['x']),np.min(stacked_df['y']),np.max(stacked_df['y'])])
        print(f"Citation graph created and saved in {time.time() - citation_graph_start:.2f} seconds")
    
    # ADDED (standalone HTML export): embed citation background image into HTML when offline export is requested.
    background_image_for_plot = None
    if citation_graph_checkbox:
        if standalone_html_checkbox:
            try:
                background_image_for_plot = encode_local_image_to_data_uri(graph_file_path)
            except Exception as e:
                print(f"Warning: Failed to embed citation background image for standalone HTML: {e}")
                background_image_for_plot = str(graph_file_path)
        else:
            background_image_for_plot = graph_file_name
    
    # Create and save plot
    plot_start = time.time()
    progress(0.7, desc="Creating interactive plot...")
    # Create a solid black colormap
    black_cmap = mcolors.LinearSegmentedColormap.from_list('black', ['#000000', '#000000'])
    
    # Generate legends based on plot type
    custom_html = ""
    legend_css = ""
    
    if HAS_LEGEND_BUILDERS:
        # CHANGED (legend behavior): show legend strictly based on selected plot coloring type.
        show_query_categorical_legend = treat_as_categorical_checkbox and has_multiple_queries
        show_custom_categorical_legend = custom_color_column_exists and custom_categorical_color_mapping is not None
        show_continuous_legend = (
            (plot_time_checkbox or custom_color_column_exists)
            and continuous_color_values is not None
            and continuous_color_cmap is not None
        )

        if show_query_categorical_legend:
            # Create categorical legend for multiple OpenAlex queries.
            unique_queries = sorted(records_df['query_index'].unique())
            color_mapping = {}
            
            # Get readable names for each query URL.
            used_names = set()  # Track used names to ensure uniqueness.
            for i, query_idx in enumerate(unique_queries):
                try:
                    if query_idx < len(urls):
                        readable_name = openalex_url_to_readable_name(urls[query_idx])
                        print(f"Query {query_idx}: Original readable name: '{readable_name}'")
                        # Truncate long names for legend display.
                        if len(readable_name) > 40:
                            readable_name = readable_name[:37] + "..."
                            print(f"Query {query_idx}: Truncated to: '{readable_name}'")
                    else:
                        readable_name = f"Query {query_idx + 1}"
                except Exception as e:
                    readable_name = f"Query {query_idx + 1}"
                    print(f"Query {query_idx}: Exception generating name: {e}")
                
                # Ensure uniqueness - if name is already used, append query number.
                original_name = readable_name
                while readable_name in used_names:
                    print(f"Query {query_idx}: Name '{readable_name}' already used, making unique...")
                    readable_name = f"{original_name} ({query_idx + 1})"
                    if len(readable_name) > 40:
                        # Re-truncate if needed after adding query number.
                        base_name = original_name[:32] + "..."
                        readable_name = f"{base_name} ({query_idx + 1})"
                
                used_names.add(readable_name)
                color_mapping[readable_name] = query_color_map[query_idx]
                print(f"Query {query_idx}: Final legend name: '{readable_name}' -> color: {query_color_map[query_idx]}")
            
            print(f"Final color mapping: {color_mapping}")
            
            legend_html, legend_css = categorical_legend_html_css(
                color_mapping,
                title="Queries" if len(color_mapping) > 1 else "Query",
                anchor="top-left",
                container_id="dmp-query-legend"
            )
            custom_html += legend_html

        elif show_custom_categorical_legend:
            # ADDED (custom CSV coloring support): legend for user-provided categorical columns.
            legend_html, legend_css = categorical_legend_html_css(
                custom_categorical_color_mapping,
                title=resolved_custom_color_column,
                anchor="top-left",
                container_id="dmp-query-legend"
            )
            custom_html += legend_html
            
        elif show_continuous_legend:
            # ADDED (custom CSV coloring support): continuous legend for publication year or numeric custom columns.
            valid_values = pd.to_numeric(pd.Series(continuous_color_values), errors='coerce').dropna()
            if len(valid_values) > 0:
                if continuous_color_vmin is not None and continuous_color_vmax is not None:
                    # CHANGED (measure scale): force fixed clipped legend domain.
                    legend_min = float(continuous_color_vmin)
                    legend_max = float(continuous_color_vmax)
                    ticks = (
                        continuous_color_ticks
                        if continuous_color_ticks is not None
                        else np.linspace(legend_min, legend_max, num=5).tolist()
                    )
                else:
                    value_min = float(valid_values.min())
                    value_max = float(valid_values.max())

                    if continuous_color_label in {"Publication Year", "Approx. Year"}:
                        year_min = int(round(value_min))
                        year_max = int(round(value_max))
                        year_range = year_max - year_min

                        first_tick = ((year_min // 5) + 1) * 5
                        ticks = []
                        current_tick = first_tick
                        while current_tick < year_max:
                            ticks.append(current_tick)
                            current_tick += 5

                        if year_range < 15:
                            if not ticks:
                                ticks = [year_min] if year_min == year_max else [year_min, year_max]
                            else:
                                if year_min not in ticks:
                                    ticks.insert(0, year_min)
                                if year_max not in ticks:
                                    ticks.append(year_max)

                        legend_min = year_min
                        legend_max = year_max
                    else:
                        if value_min == value_max:
                            ticks = [value_min]
                        else:
                            ticks = np.linspace(value_min, value_max, num=5).tolist()
                        legend_min = value_min
                        legend_max = value_max

                legend_html, legend_css = continuous_legend_html_css(
                    continuous_color_cmap,
                    legend_min,
                    legend_max,
                    ticks=ticks,
                    label=continuous_color_label or "Value",
                    anchor="top-right",
                    container_id="dmp-year-legend"
                )
                custom_html += legend_html
    
    # Add custom CSS to make legend titles equally large and bold
    legend_title_css = """
/* Make all legend titles equally large and bold */
#dmp-query-legend .legend-title,
#dmp-year-legend .colorbar-label {
    font-size: 16px !important;
    font-weight: bold !important;
    font-family: 'Roboto Condensed', sans-serif !important;
}
"""
    
    # Combine legend CSS with existing custom CSS
    combined_css = DATAMAP_CUSTOM_CSS + "\n" + legend_css + "\n" + legend_title_css
    
    
    
    # CHANGED (custom CSV support): add click behavior only when DOI data exists.
    interactive_plot_kwargs = dict(
        hover_text=[str(value) for value in stacked_df['hover_text_custom']],
        marker_color_array=stacked_df['color'],
        marker_size_array=marker_sizes,
        use_medoids=True,  # Switch back once efficient medoid calculation comes out.
        width=1000,
        height=1000,
        point_radius_min_pixels=1,
        text_outline_width=5,
        point_hover_color=highlight_color,
        point_radius_max_pixels=measure_scaled_point_radius_max_pixels,
        cmap=black_cmap,
        background_image=background_image_for_plot,
        font_family="Roboto Condensed",
        font_weight=600,
        tooltip_font_weight=600,
        tooltip_font_family="Roboto Condensed",
        hover_text_html_template="<div>{hover_text}</div>",
        custom_html=custom_html,
        custom_css=combined_css,
        initial_zoom_fraction=.8,
        enable_search=False,
        # CHANGED (font regression fix): keep preview rendering in normal web mode.
        # Standalone download is produced later from a copied file via post-processing.
        offline_mode=False,
    )
    interactive_plot_kwargs["extra_point_data"] = extra_data
    if click_handler is not None:
        interactive_plot_kwargs["on_click"] = click_handler

    plot = datamapplot.create_interactive_plot(
        stacked_df[['x','y']].values,
        np.array(stacked_df['cluster_2_labels']),
        np.array(['Unlabelled' if pd.isna(x) else x for x in stacked_df['parsed_field']]),
        **interactive_plot_kwargs
    )

    # Save plot
    plot.save(html_file_path)
    # CHANGED (standalone HTML export): keep the iframe HTML untouched and build a separate
    # downloadable standalone file to avoid visual regressions (e.g., font rendering changes).
    html_download_path = html_file_path
    if standalone_html_checkbox:
        standalone_start_time = time.time()
        standalone_html_file_path = static_dir / f"{filename}__standalone.html"
        try:
            with open(html_file_path, "r", encoding="utf-8", errors="replace") as src_file:
                base_html_text = src_file.read()
            with open(standalone_html_file_path, "w", encoding="utf-8") as dst_file:
                dst_file.write(base_html_text)
            standalone_summary = make_html_fully_standalone(standalone_html_file_path)
            html_download_path = standalone_html_file_path
            print(
                "Standalone HTML export summary:",
                standalone_summary,
                f"(completed in {time.time() - standalone_start_time:.2f}s)",
            )
        except Exception as e:
            print(f"Warning: Failed to finalize standalone HTML export: {e}")
    print(f"Plot created and saved in {time.time() - plot_start:.2f} seconds")
    
    # Save additional files if requested
    if download_csv_checkbox:
        # CHANGED (custom CSV support): export all available columns, including custom measure columns.
        export_df = records_df.copy()
        export_df['parsed_field'] = [get_field(row) for ix, row in export_df.iterrows()]
        if 'referenced_works' in export_df.columns:
            export_df['referenced_works'] = [
                ', '.join(x) if isinstance(x, list) else x
                for x in export_df['referenced_works']
            ]
        
        # Add query information if categorical coloring is used
        if treat_as_categorical_checkbox and has_multiple_queries:
            export_df['query_index'] = records_df['query_index']
            export_df['query_label'] = records_df['query_label']
        
        if continuous_color_values is not None:
            export_df['color_value'] = pd.to_numeric(pd.Series(continuous_color_values), errors='coerce')
            export_df['color_value_label'] = continuous_color_label
        export_df.to_csv(csv_file_path, index=False)
        
    if download_png_checkbox:
        png_start_time = time.time()
        print("Starting PNG generation...")

        # Sample and prepare data
        sample_prep_start = time.time()
        sample_to_plot = basedata_df#.sample(20000)
        labels1 = np.array(sample_to_plot['cluster_2_labels'])
        labels2 = np.array(['Unlabelled' if pd.isna(x) else x for x in sample_to_plot['parsed_field']])
        
        ratio = 0.6
        mask = np.random.random(size=len(labels1)) < ratio
        combined_labels = np.where(mask, labels1, labels2)
        
        # Get the 30 most common labels
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        top_30_labels = set(unique_labels[np.argsort(counts)[-80:]])
        
        # Replace less common labels with 'Unlabelled'
        combined_labels = np.array(['Unlabelled' if label not in top_30_labels else label for label in combined_labels])
        colors_base = ['#536878' for _ in range(len(labels1))]
        print(f"Sample preparation completed in {time.time() - sample_prep_start:.2f} seconds")

        # Create main plot
        main_plot_start = time.time()
        fig, ax = datamapplot.create_plot(
            sample_to_plot[['x','y']].values,
            combined_labels,
            label_wrap_width=12,
            label_over_points=True,
            dynamic_label_size=True,
            use_medoids=True, # Switch back once efficient mediod caclulation comes out!
            point_size=2,
            marker_color_array=colors_base,
            force_matplotlib=True,
            max_font_size=12,
            min_font_size=4,
            min_font_weight=100,
            max_font_weight=300,
            font_family="Roboto Condensed",
            color_label_text=False, add_glow=False,
            highlight_labels=list(np.unique(labels1)),
            label_font_size=8,
            highlight_label_keywords={"fontsize": 12, "fontweight": "bold", "bbox":{"boxstyle":"circle", "pad":0.75,'alpha':0.}},
        )
        print(f"Main plot creation completed in {time.time() - main_plot_start:.2f} seconds")

        if citation_graph_checkbox:
            # Read and add the graph image
            graph_img = plt.imread(graph_file_path)
            ax.imshow(graph_img, extent=[np.min(stacked_df['x']),np.max(stacked_df['x']),np.min(stacked_df['y']),np.max(stacked_df['y'])],
                        alpha=0.9, aspect='auto')

        if len(records_df) > 50_000:
            point_size = .5
        elif len(records_df) > 10_000:
            point_size = 1
        else:
            point_size = 5
            
        # CHANGED (custom CSV support): shared PNG logic for continuous and categorical coloring.
        scatter_start = time.time()
        has_continuous_coloring = continuous_color_values is not None and continuous_color_cmap is not None
        if has_continuous_coloring:
            continuous_series = pd.to_numeric(pd.Series(continuous_color_values), errors='coerce')
            if continuous_series.notna().any():
                continuous_series = continuous_series.fillna(continuous_series.dropna().median())
                scatter_kwargs = {}
                if continuous_color_vmin is not None and continuous_color_vmax is not None:
                    # CHANGED (measure scale): keep static plot color scale aligned with interactive plot.
                    continuous_series = continuous_series.clip(lower=continuous_color_vmin, upper=continuous_color_vmax)
                    scatter_kwargs["vmin"] = continuous_color_vmin
                    scatter_kwargs["vmax"] = continuous_color_vmax
                scatter = plt.scatter(
                    umap_embeddings[:, 0],
                    umap_embeddings[:, 1],
                    c=continuous_series,
                    cmap=continuous_color_cmap,
                    alpha=0.8,
                    s=point_size,
                    **scatter_kwargs
                )
                cbar = plt.colorbar(scatter, shrink=0.5)
                if continuous_color_label:
                    cbar.set_label(continuous_color_label)
            else:
                scatter = plt.scatter(
                    umap_embeddings[:, 0],
                    umap_embeddings[:, 1],
                    c=records_df['color'],
                    alpha=0.8,
                    s=point_size
                )
        else:
            scatter = plt.scatter(
                umap_embeddings[:,0],
                umap_embeddings[:,1],
                c=records_df['color'],
                alpha=0.8,
                s=point_size
            )

            legend_handles = []

            # Add legend for OpenAlex categorical query coloring.
            if treat_as_categorical_checkbox and has_multiple_queries:
                unique_categories = records_df['query_index'].unique()
                
                # Create legend handles with larger point size using the color mapping.
                for query_idx in sorted(unique_categories):
                    # Get the readable name for this query.
                    try:
                        if query_idx < len(urls):
                            readable_name = openalex_url_to_readable_name(urls[query_idx])
                            # Truncate long names for legend display.
                            if len(readable_name) > 40:
                                readable_name = readable_name[:37] + "..."
                        else:
                            readable_name = f"Query {query_idx + 1}"
                    except Exception:
                        readable_name = f"Query {query_idx + 1}"
                    
                    color = query_color_map[query_idx]
                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=9, 
                                                label=readable_name, linestyle='None'))

            # Add legend for custom categorical column coloring.
            elif custom_categorical_color_mapping:
                for category, color in custom_categorical_color_mapping.items():
                    legend_handles.append(plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=9,
                        label=str(category), linestyle='None'
                    ))

            if legend_handles:
                plt.legend(
                    handles=legend_handles,
                    loc='upper left',
                    frameon=False,
                    fancybox=False,
                    shadow=False,
                    framealpha=0.9,
                    fontsize=9,
                )
            
        print(f"Scatter plot creation completed in {time.time() - scatter_start:.2f} seconds")

        # Save plot
        save_start = time.time()
        plt.axis('off')
        plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saving completed in {time.time() - save_start:.2f} seconds")
        
        print(f"Total PNG generation completed in {time.time() - png_start_time:.2f} seconds")

    progress(1.0, desc="Done!")
    print(f"Total pipeline completed in {time.time() - start_time:.2f} seconds")
    # CHANGED (cache busting): append a timestamp query parameter to force iframe refresh.
    html_iframe_src = f"{VIS_STATIC_ROUTE}/{html_file_name}"
    iframe = f"""<iframe src="{html_iframe_src}?v={int(time.time())}" width="100%" height="1000px"></iframe>"""
    
    # Return iframe and download buttons with appropriate visibility
    return [
        iframe,
        gr.DownloadButton(label="Download Interactive Visualization", value=html_download_path, visible=True, variant='secondary'),
        gr.DownloadButton(label="Download CSV Data", value=csv_file_path, visible=download_csv_checkbox, variant='secondary'),
        gr.DownloadButton(label="Download Static Plot", value=png_file_path, visible=download_png_checkbox, variant='secondary'),
        gr.Button(visible=False)  # Return hidden state for cancel button
    ]

predict.zerogpu = True



theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Roboto Condensed"), "ui-sans-serif", "system-ui", "sans-serif"],
    text_size="lg",
).set(
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#f3f4f6",
    button_secondary_border_color="black",
    button_secondary_text_color="black",
    button_border_width="2px",
)


# JS to enforce light theme by refreshing the page
js_light = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""




# Gradio interface setup
with gr.Blocks(theme=theme, css=f"""
    .gradio-container a {{
        color: black !important;
        text-decoration: none !important;  /* Force remove default underline */
        font-weight: bold;
        transition: color 0.2s ease-in-out, border-bottom-color 0.2s ease-in-out;
        display: inline-block;  /* Enable proper spacing for descenders */
        line-height: 1.1;  /* Adjust line height */
        padding-bottom: 2px;  /* Add space for descenders */
    }}
    .gradio-container a:hover {{
        color: #b23310 !important;
        border-bottom: 3px solid #b23310;  /* Wider underline, only on hover */
    }}
    
    /* Colormap chooser styles */
    {colormap_chooser.css()}
""", js=js_light) as demo:
    gr.Markdown("""
    <div style="max-width: 100%; margin: 0 auto;">
    <br>
    
    # OpenAlex Mapper
    
    OpenAlex Mapper is a way of projecting search queries from the amazing OpenAlex database on a background map of randomly sampled papers from OpenAlex, which allows you to easily investigate interdisciplinary connections. OpenAlex Mapper was developed by [Maximilian Noichl](https://maxnoichl.eu) and [Andrea Loettgers](https://unige.academia.edu/AndreaLoettgers) at the [Possible Life project](http://www.possiblelife.eu/).

    To use OpenAlex Mapper, first head over to [OpenAlex](https://openalex.org/) and search for something that interests you. For example, you could search for all the papers that make use of the [Kuramoto model](https://openalex.org/works?page=1&filter=default.search%3A%22Kuramoto%20Model%22), for all the papers that were published by researchers at [Utrecht University in 2019](https://openalex.org/works?page=1&filter=authorships.institutions.lineage%3Ai193662353,publication_year%3A2019), or for all the papers that cite Wittgenstein's [Philosophical Investigations](https://openalex.org/works?page=1&filter=cites%3Aw4251395411). Then you copy the URL to that search query into the OpenAlex search URL box below and click "Run Query." It will download all of these records from OpenAlex and embed them on our interactive map. As the embedding step is a little expensive, computationally, it's often a good idea to play around with smaller samples, before running a larger analysis (see below for a note on sample size and gpu-limits). After a little time, that map will appear and be available for you to interact with and download. You can find more explanations in the FAQs below.
    </div>
    
    """)
    

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                run_btn = gr.Button("Run Query", variant='primary')
                cancel_btn = gr.Button("Cancel", visible=False, variant='secondary')
            
            # Create separate download buttons
            html_download = gr.DownloadButton("Download Interactive Visualization", visible=False, variant='secondary')
            csv_download = gr.DownloadButton("Download CSV Data", visible=False, variant='secondary')
            png_download = gr.DownloadButton("Download Static Plot", visible=False, variant='secondary')

            text_input = gr.Textbox(label="OpenAlex-search URL",
                                    info="Enter the URL to an OpenAlex-search.")
            
            # Add the query highlight display
            query_display = gr.HTML(
                value="<div style='padding: 10px; color: #666; font-style: italic;'>Enter OpenAlex URLs separated by semicolons to see query descriptions</div>",
                label="",
                show_label=False
            )
            
            gr.Markdown("### Sample Settings")
            reduce_sample_checkbox = gr.Checkbox(
                label="Reduce Sample Size",
                value=True,
                info="Reduce sample size."
            )
            sample_reduction_method = gr.Dropdown(
                ["All", "First n samples", "n random samples"],
                label="Sample Selection Method",
                value="First n samples",
                info="How to choose the samples to keep.",
                visible=True  # Will be controlled by reduce_sample_checkbox
            )
            
            if is_running_in_hf_zero_gpu():
                max_sample_size = 20000
            else:
                max_sample_size = 250000

            sample_size_slider = gr.Slider(
                label="Sample Size",
                minimum=500,
                maximum=max_sample_size,
                step=10,
                value=1000,
                info="How many samples to keep.",
                visible=True  # Will be controlled by reduce_sample_checkbox
            )

            # Add this new seed field
            seed_textbox = gr.Textbox(
                label="Random Seed",
                value="42",
                info="Seed for random sampling reproducibility.",
                visible=False  # Will be controlled by both reduce_sample_checkbox and sample_reduction_method
            )
            
            gr.Markdown("### Plot Settings")
            # CHANGED (custom CSV support): the dropdown can be extended with uploaded CSV column names.
            plot_type_dropdown = gr.Dropdown(
                BASE_PLOT_TYPE_CHOICES,
                label="Plot Coloring Type",
                value=PLOT_TYPE_NO_SPECIAL,
                info="Choose how to color points. After CSV upload, additional column names appear here."
            )
            locally_approximate_publication_date_checkbox = gr.Checkbox(
                label="Locally Approximate Publication Date",
                value=True,
                info="Colour points by the average publication date in their area.",
                visible=False  # Will be controlled by plot_type_dropdown
            )
            # Remove treat_as_categorical_checkbox since it's now part of the dropdown
            
            gr.Markdown("### Download Options")
            download_csv_checkbox = gr.Checkbox(
                label="Generate CSV Export",
                value=False,
                info="Export the data as CSV file"
            )
            download_png_checkbox = gr.Checkbox(
                label="Generate Static PNG Plot",
                value=False,
                info="Export a static PNG visualization. This will make things slower!"
            )
            standalone_html_checkbox = gr.Checkbox(
                label="Standalone Interactive HTML (Offline)",
                value=True,
                info="Inline JS/CSS/font assets into one HTML file (first export may need internet access)."
            )
            
            
            gr.Markdown("### Citation graph")
            citation_graph_checkbox = gr.Checkbox(
                label="Add Citation Graph",
                value=False,
                info="Adds a citation graph of the sample to the plot."
            )
            
            gr.Markdown("### Upload Your Own Data")
            csv_upload = gr.File(
                file_count="single",
                label="Upload your own CSV file (custom data supported).", 
                file_types=[".csv"],
            )
            
            # --- Aesthetics Accordion ---
            with gr.Accordion("Aesthetics", open=False):
                gr.Markdown("### Color Selection")
                gr.Markdown("*Choose an individual color to highlight your data.*")
                highlight_color_picker = gr.ColorPicker(
                    label="Highlight Color",
                    show_label=False,
                    value="#5e2784",
                    #info="Choose the highlight color for your query points."
                )
                
                # Add colormap chooser
                gr.Markdown("### Colormap Selection")
                gr.Markdown("*Choose a colormap for time-based visualizations (when 'Plot Time' is enabled)*")
                
                # Render the colormap chooser (created earlier)
                colormap_chooser.render_tabs()
            
        with gr.Column(scale=2):
            html = gr.HTML(
                value='<div style="width: 100%; height: 1000px; display: flex; justify-content: center; align-items: center; border: 1px solid #ccc; background-color: #f8f9fa;"><p style="font-size: 1.2em; color: #666;">The visualization map will appear here after running a query</p></div>',
                label="", 
                show_label=False
            )
    gr.Markdown("""
    <div style="max-width: 100%; margin: 0 auto;">
    
    # FAQs
    
    ## Who made this?

    This project was developed by [Maximilian Noichl](https://maxnoichl.eu) (Utrecht University), in cooperation with Andrea Loettgers and Tarja Knuuttila at the [Possible Life project](http://www.possiblelife.eu/), at the University of Vienna. If this project is useful in any way for your research, we would appreciate citation of:
     Noichl, M., Loettgers, A., Knuuttila, T. (2025).[Philosophy at Scale: Introducing OpenAlex Mapper](https://maxnoichl.eu/full/talks/talk_BERLIN_April_2025/working_paper.pdf). *Working Paper*.

    This project received funding from the European Research Council under the European Union's Horizon 2020 research and innovation programme (LIFEMODE project, grant agreement No. 818772). 

    ## How does it work?

    The base map for this project is developed by randomly downloading 250,000 articles from OpenAlex, then embedding their abstracts using our [fine-tuned](https://huggingface.co/m7n/discipline-tuned_specter_2_024) version of the [specter-2](https://huggingface.co/allenai/specter2_aug2023refresh_base) language model, running these embeddings through [UMAP](https://umap-learn.readthedocs.io/en/latest/) to give us a two-dimensional representation, and displaying that in an interactive window using [datamapplot](https://datamapplot.readthedocs.io/en/latest/index.html). After the data for your query is downloaded from OpenAlex, it then undergoes the exact same process, but the pre-trained UMAP model from earlier is used to project your new data points onto this original map, showing where they would show up if they were included in the original sample. For more details, you can take a look at the method section of this [working paper](https://maxnoichl.eu/full/talks/talk_BERLIN_April_2025/working_paper.pdf).
    
    ## I'm getting an "out of GPU credits" error.
    
    Running the embedding process requires an expensive A100 GPU. To provide this, we make use of HuggingFace's ZeroGPU service. As an anonymous user, this entitles you to one minute of GPU runtime, which is enough for several small queries of around a thousand records every day. If you create a free account on HuggingFace, this should increase to five minutes of runtime, allowing you to run successful queries of up to 10,000 records at a time. If you need more, there's always the option to either buy a HuggingFace Pro subscription for roughly ten dollars a month (entitling you to 25 minutes of runtime every day) or get in touch with us to run the pipeline outside of the HuggingFace environment.
    
    ## I want to add multiple queries at once!

    That can be a good idea, e. g. if your interested in a specific paper, as well as all the papers that cite it. Just add the queries to the query box and separate them with a ";" without any spaces in between!

    ## I think I found a mistake in  the map.

    There are various considerations to take into account when working with this map:

    1. The language model we use is fine-tuned to separate disciplines from each other, but of course, disciplines are weird, partially subjective social categories, so what the model has learned might not always correspond perfectly to what you would expect to see.

    2. When pressing down a really high-dimensional space into a low-dimensional one, there will be trade-offs. For example, we see this big ring structure of the sciences on the map, but in the middle of the map there is a overly stretchedstring of bioinformaticsthat stretches from computer science at the bottom up to the life sciences clusters at the top. This is one of the areas where the UMAP algorithm had trouble pressing our high-dimensional dataset into a low-dimensional space. For more information on how to read a UMAP plot, I recommend looking into ["Understanding UMAP"](https://pair-code.github.io/understanding-umap/) by Andy Coenen & Adam Pearce.
    
    3. Finally, the labels we're using for the regions of this plot are created from OpenAlex's own labels of sub-disciplines. They give a rough indication of the papers that could be expected in this broad area of the map, but they are not necessarily the perfect label for the articles that are precisely below them. They are just located at the median point of a usually much larger, much broader, and fuzzier category, so they should always be taken with quite a big grain of salt.
    
    ## I want to use my own data!
    
    Sure! You can upload your own CSV data directly. At minimum, provide `title` plus `abstract` (or `abtract`, which is accepted as an alias). All additional columns are kept and can be used for plot coloring. You can also still upload a CSV containing only `doi` values; those records will then be fetched from OpenAlex before projection.
    
    </div>
    """)

    # Update the visibility control functions
    def update_sample_controls_visibility(reduce_sample_enabled, sample_method):
        """Update visibility of sample reduction controls based on checkbox and method"""
        method_visible = reduce_sample_enabled
        slider_visible = reduce_sample_enabled and sample_method != "All"
        seed_visible = reduce_sample_enabled and sample_method == "n random samples"
        
        return (
            gr.Dropdown(visible=method_visible),
            gr.Slider(visible=slider_visible), 
            gr.Textbox(visible=seed_visible)
        )
    
    def update_plot_controls_visibility(plot_type):
        """Update visibility of plot controls based on plot type"""
        locally_approx_visible = plot_type == PLOT_TYPE_TIME_BASED
        return gr.Checkbox(visible=locally_approx_visible)

    # ADDED (custom CSV coloring support): update plot-coloring choices from uploaded CSV columns.
    def update_plot_coloring_choices_from_upload(csv_file, current_plot_type):
        choices = get_plot_coloring_choices_for_file(csv_file)
        selected_plot_type = current_plot_type if current_plot_type in choices else PLOT_TYPE_NO_SPECIAL
        locally_approx_visible = selected_plot_type == PLOT_TYPE_TIME_BASED
        return (
            gr.Dropdown(choices=choices, value=selected_plot_type),
            gr.Checkbox(visible=locally_approx_visible)
        )

    # Update event handlers
    reduce_sample_checkbox.change(
        fn=update_sample_controls_visibility,
        inputs=[reduce_sample_checkbox, sample_reduction_method],
        outputs=[sample_reduction_method, sample_size_slider, seed_textbox]
    )
    
    sample_reduction_method.change(
        fn=update_sample_controls_visibility,
        inputs=[reduce_sample_checkbox, sample_reduction_method],
        outputs=[sample_reduction_method, sample_size_slider, seed_textbox]
    )
    
    plot_type_dropdown.change(
        fn=update_plot_controls_visibility,
        inputs=[plot_type_dropdown],
        outputs=[locally_approximate_publication_date_checkbox]
    )

    csv_upload.change(
        fn=update_plot_coloring_choices_from_upload,
        inputs=[csv_upload, plot_type_dropdown],
        outputs=[plot_type_dropdown, locally_approximate_publication_date_checkbox]
    )

    def show_cancel_button():
        return gr.Button(visible=True)
    
    def hide_cancel_button():
        return gr.Button(visible=False)
    
    show_cancel_button.zerogpu = True
    hide_cancel_button.zerogpu = True
    predict.zerogpu = True

    # Update the run button click event
    run_event = run_btn.click(
        fn=show_cancel_button,
        outputs=cancel_btn,
        queue=False
    ).then(
        fn=predict,
        inputs=[
            text_input, 
            sample_size_slider, 
            reduce_sample_checkbox, 
            sample_reduction_method, 
            plot_type_dropdown,  # Changed from plot_time_checkbox
            locally_approximate_publication_date_checkbox,
            # Removed treat_as_categorical_checkbox since it's now part of plot_type_dropdown
            download_csv_checkbox, 
            download_png_checkbox,
            citation_graph_checkbox,
            standalone_html_checkbox,
            csv_upload,
            highlight_color_picker,
            colormap_chooser.selected_name,
            seed_textbox
        ],
        outputs=[html, html_download, csv_download, png_download, cancel_btn]
    )

    # Add cancel button click event
    cancel_btn.click(
        fn=hide_cancel_button,
        outputs=cancel_btn,
        cancels=[run_event],
        queue=False  # Important to make the button hide immediately
    )

    # Connect text input changes to query display updates
    text_input.change(
        fn=highlight_queries,
        inputs=text_input,
        outputs=query_display
    )
    
    demo.load(fn=_warmup, inputs=None, outputs=None, queue=False)
    _warmup()
    


# demo.static_dirs = {
#     "static": str(static_dir)
# }


# Mount and run app
# app = gr.mount_gradio_app(app, demo, path="/",ssr_mode=False)

# app.zerogpu = True  # Add this line


# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860, share=True,allowed_paths=["/static"])
    
# Mount Gradio app to FastAPI
if is_running_in_hf_space():
    app = gr.mount_gradio_app(app, demo, path="/",ssr_mode=False) # setting to false for now. 
else:
    app = gr.mount_gradio_app(app, demo, path="/",ssr_mode=False) 

# Run both servers
if __name__ == "__main__":
    if is_running_in_hf_space():
        # For HF Spaces, use SSR mode
        os.environ["GRADIO_SSR_MODE"] = "True"
        uvicorn.run("app:app", host="0.0.0.0", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)
