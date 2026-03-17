import pickle
import requests
#import umap
import umap.umap_ as umap 
from numba.typed import List
import torch
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path


# ADDED (numba pickle compatibility): gracefully handle older/newer Dispatcher._rebuild signatures.
def _apply_numba_impl_kind_compat_patch():
    """
    Patch numba dispatcher rebuild functions to ignore unknown `impl_kind`
    state entries when unpickling mapper objects across numba versions.
    """
    try:
        from numba.core import serialize as numba_serialize
    except Exception as e:
        print(f"Warning: Could not import numba serialize module for compatibility patch: {e}")
        return False

    patch_marker = "_openalex_mapper_custom_rebuild_patch"
    if getattr(numba_serialize, patch_marker, False):
        return True

    original_custom_rebuild = getattr(numba_serialize, "custom_rebuild", None)
    if original_custom_rebuild is None:
        return False

    def _strip_incompatible_numba_keys(obj):
        """
        Recursively remove numba state keys that are known to be incompatible
        across versions.
        """
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key in {"impl_kind", "target_backend"}:
                    continue
                cleaned[key] = _strip_incompatible_numba_keys(value)
            return cleaned
        if isinstance(obj, list):
            return [_strip_incompatible_numba_keys(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_strip_incompatible_numba_keys(v) for v in obj)
        return obj

    def _custom_rebuild_compat(custom_pickled):
        cls, states = custom_pickled.ctor, custom_pickled.states

        # Drop incompatible state keys produced by different numba versions.
        states = _strip_incompatible_numba_keys(states)

        # Map old base dispatcher references to the concrete CPU dispatcher class
        # in newer numba versions.
        if (
            getattr(cls, "__name__", "") == "Dispatcher"
            and getattr(cls, "__module__", "").startswith("numba.core.dispatcher")
        ):
            try:
                from numba.core import registry as numba_registry
                cpu_dispatcher_cls = getattr(numba_registry, "CPUDispatcher", None)
                if cpu_dispatcher_cls is not None:
                    cls = cpu_dispatcher_cls
            except Exception:
                pass

        return cls._rebuild(**states)

    numba_serialize.custom_rebuild = _custom_rebuild_compat
    setattr(numba_serialize, patch_marker, True)
    return True

def check_resources(files_dict, basemap_path, mapper_params_path):
    """
    Check if all required resources are present.
    
    Args:
        files_dict (dict): Dictionary mapping filenames to their download URLs
        basemap_path (str): Path to the basemap pickle file
        mapper_params_path (str): Path to the UMAP mapper parameters pickle file
        
    Returns:
        bool: True if all resources are present, False otherwise
    """
    all_files_present = True
    
    # Check downloaded files
    for filename in files_dict.keys():
        if not Path(filename).exists():
            print(f"Missing file: {filename}")
            all_files_present = False
    
    # Check basemap
    if not Path(basemap_path).exists():
        print(f"Missing basemap file: {basemap_path}")
        all_files_present = False
        
    # Check mapper params
    if not Path(mapper_params_path).exists():
        print(f"Missing mapper params file: {mapper_params_path}")
        all_files_present = False
    
    return all_files_present

def download_required_files(files_dict):
    """
    Download required files from URLs only if they don't exist.
    
    Args:
        files_dict (dict): Dictionary mapping filenames to their download URLs
    """
    print(f"Checking required files: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    files_to_download = {
        filename: url 
        for filename, url in files_dict.items() 
        if not Path(filename).exists()
    }
    
    if not files_to_download:
        print("All files already present, skipping downloads")
        return
        
    print(f"Downloading missing files: {list(files_to_download.keys())}")
    for filename, url in files_to_download.items():
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

def setup_basemap_data(basemap_path):
    """
    Load and setup the base map data.
    
    Args:
        basemap_path (str): Path to the basemap pickle file
    """
    print(f"Getting basemap data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    basedata_df = pickle.load(open(basemap_path, 'rb'))
    return basedata_df

def setup_mapper(mapper_params_path):
    """
    Setup and configure the UMAP mapper.
    
    Args:
        mapper_params_path (str): Path to the UMAP mapper parameters pickle file
    """
    print(f"Getting Mapper: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CHANGED (numba pickle compatibility): retry with compatibility patch when
    # mapper pickle contains an `impl_kind` state from a different numba version.
    try:
        with open(mapper_params_path, 'rb') as mapper_file:
            params_new = pickle.load(mapper_file)
    except TypeError as e:
        if "impl_kind" not in str(e):
            raise

        print("Detected numba pickle compatibility issue (impl_kind). Applying compatibility patch...")
        patch_applied = _apply_numba_impl_kind_compat_patch()
        if not patch_applied:
            raise

        with open(mapper_params_path, 'rb') as mapper_file:
            params_new = pickle.load(mapper_file)

    print("setting up mapper...")
    mapper = umap.UMAP()
    
    umap_params = {k: v for k, v in params_new.get('umap_params', {}).items() 
                  if k != 'target_backend'}
    mapper.set_params(**umap_params)
    
    for attr, value in params_new.get('umap_attributes', {}).items():
        if attr != 'embedding_':
            setattr(mapper, attr, value)
    
    if 'embedding_' in params_new.get('umap_attributes', {}):
        mapper.embedding_ = List(params_new['umap_attributes']['embedding_'])
    
    return mapper

def setup_embedding_model(model_name):
    """
    Setup the SentenceTransformer model.
    
    Args:
        model_name (str): Name or path of the SentenceTransformer model
    """
    print(f"Setting up language model: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # CHANGED (local stability): prefer CPU over MPS to avoid Apple Metal hangs.
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name, device=str(device))
    return model
