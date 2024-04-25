import gzip
import math
import os
import os.path
from pathlib import Path
import pickle
import torch.nn.functional as F
from contextlib import contextmanager
from scipy.ndimage import distance_transform_edt
import sys
import random

# Get the absolute path to the root of the project by navigating up two levels from this file
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset.scene_dataset import SceneDataset
from src.io_utils import read_data
from src.scripts.category_prediction import generate_category_dataset

def ensuredir(dirname):
    """Ensure a directory exists"""
    if not os.path.exists(dirname):
            os.makedirs(dirname)

'''
Turn a number into a string that is zero-padded up to length n
'''
def zeropad(num, n):
    sn = str(num)
    while len(sn) < n:
            sn = '0' + sn
    return sn

def pickle_dump_compressed(object, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """Pickles + compresses an object to file"""
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def pickle_load_compressed(filename):
    """Loads a compressed pickle file and returns reconstituted object"""
    file = gzip.GzipFile(filename, 'rb')
    buffer = b""
    while True:
            data = file.read()
            if data == b"":
                    break
            buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object
        
def get_data_root_dir():
    """
    Get root dir of the data, defaults to /data if env viariable is not set
    """
    env_path = os.environ.get("SCENESYNTH_DATA_PATH")
    if env_path:
        return env_path
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return f"{root_dir}/data"

# stolen from category_prediction.py; returns category dataset or creates it if missing
def get_scene_dataset(dataset_path: Path, type: str) -> SceneDataset:
    scenes_path = dataset_path / "formatted_data" / "parse.pkl"
    metadata_path = dataset_path / "scene_datasets" / "category.pkl"
    if not metadata_path.exists():
        (dataset_path / "scene_datasets").mkdir(parents=True, exist_ok=True)
        scenes = read_data(scenes_path)
        print("Generating category dataset")
        subscenes_meta = read_data(scenes_path.parent / 'subscenes_meta.pkl')
        generate_category_dataset(scenes, subscenes_meta, metadata_path)
    scene_dataset = SceneDataset(scenes_path, metadata_path, type)
    return scene_dataset

# ported from latent_dataset.py in SceneSynth. dims.py requires this each epoch
def prepare_same_category_batches(dataset: SceneDataset, cats_seen, batch_size: int):
    # Build a random list of category indices (grouped by batch_size)
    # This requires than length of dataset is a multiple of batch_size
    assert(len(dataset) % batch_size == 0)
    num_batches = len(dataset) // batch_size
    same_category_batch_indices = []
    for i in range(num_batches):
        # cat_index = random.randint(0, self.n_categories-1)
        cat_index = random.choice(cats_seen)
        for j in range(batch_size):
            same_category_batch_indices.append(cat_index)
    return same_category_batch_indices

def memoize(func):
    """
    Decorator to memoize a function
    https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
    """
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                    cache[key] = func(*args, **kwargs)
            return cache[key]

    return memoized_func

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    From https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    Suppress C warnings
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
