import argparse
from pathlib import Path
import torch

from cat import NextCategory
from cat import latent_dim as cat_latent_dim
from loc import Model as LocModel
from orient import Model as OrientModel
from orient import latent_size as orient_latent_size
from orient import hidden_size as orient_hidden_size
from dims import Model as DimsModel
from dims import latent_size as dims_latent_size
from dims import hidden_size as dims_hidden_size

cat_name = 'nextcat_25.pt'
loc_name = 'location_14.pt'
dims_name = 'model_dims_240.pt'
orient_name = 'model_orient_115.pt'

# These arguments are flexibly I'm not sure what they will need to be quite yet

def sample_category(cat_model, input_img):
    """
    TODO - reference fast_synth.py line 491 - 500
    """

def sample_location(loc_model, input_img, category):
    """
    TODO  - reference fast_synth.py line 609 - 620
    """

def sample_orientation(orient_model, input_img, category):
    """
    TODO - reference fast_synth.py line 520
    I'm pretty sure can just call directly
    """

def sample_dimensions(dims_model, input_img, category):
    """
    TODO  - reference fast_synth.py line 526
    """

def generate_scene(scene, cat_model, loc_model, orient_model, dims_model):
    while True:
        input_img = scene.to_fastsynth_inputs()
        category = sample_category(cat_model, input_img)
        x, y = sample_location(loc_model, input_img, category)
        """
        TODO translate input image - reference fast_synth.py 514 - 519
        generate input_img_orient
        """
        sample_orientation(orient_model, input_img_orient, category)
        """
        TODO rotate input image - reference fast_synth.py 522 - 524
        generate input_img_dims
        """
        sample_dimensions(dims_model, input_img_dims, category)

if __name__ == '__main__':
    from src.config import data_filepath
    from src.object.config import object_types
    from src.io_utils import read_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, type=Path, help="save directory for models")
    parser.add_argument("--out-dir", required=True, type=Path, help="where to output results")
    parser.add_argument("--dataset", type=str, default="grammar")
    args = parser.parse_args()

    scenes_path = data_filepath / args.dataset / "formatted_data" / "parse.pkl"
    scenes = read_data(scenes_path)

    num_categories = len(object_types)
    num_input_channels = num_categories + 6

    print("Loading Category Model")
    cat_model = NextCategory(
        num_input_channels,
        num_categories,
        cat_latent_dim
    )
    cat_model.load_state_dict(torch.load(args.save_dir / cat_name))

    print("Loading Location Model")
    loc_model = LocModel(
        num_classes=num_categories,
        num_input_channels=num_input_channels
    )
    loc_model.load_state_dict(torch.load(args.save_dir / loc_name))

    print("Loading Orientation Model")
    orient_model = OrientModel(
        latent_size=orient_latent_size,
        hidden_size=orient_hidden_size,
        num_input_channels=num_input_channels
    )
    orient_model.load(args.save_dir / orient_name)

    print("Loading Dimensions Model")
    dims_model = DimsModel(
        latent_size=dims_latent_size,
        hidden_size=dims_hidden_size,
        num_input_channels=num_input_channels
    )
    dims_model.load(args.save_dir / dims_name)

    cat_model.eval()
    loc_model.eval()
    orient_model.eval()
    if torch.cuda.is_available():
        cat_model, loc_model, orient_model, dims_model = \
            cat_model.cuda(), loc_model.cuda(), orient_model.cuda(), dims_model.cuda()

    for scene in scenes:
        scene_copy = scene.copy(empty=True)
        generate_scene(scene_copy)