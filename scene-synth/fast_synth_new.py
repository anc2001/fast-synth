import argparse
from pathlib import Path
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import shutil
import math

from cat import NextCategory
from cat import latent_dim as cat_latent_dim
from loc import Model as LocModel
from orient import Model as OrientModel
from orient import latent_size as orient_latent_size
from orient import hidden_size as orient_hidden_size
from dims import Model as DimsModel
from dims import latent_size as dims_latent_size
from dims import hidden_size as dims_hidden_size
from models.utils import inverse_xform_img

from utils import save_input_img_as_png

cat_name = 'nextcat_25.pt'
loc_name = 'location_14.pt'
dims_name = 'model_dims_25.pt'
orient_name = 'model_orient_115.pt'

# These arguments are flexibly I'm not sure what they will need to be quite yet

def sample_category(cat_model, input_img, cats):
    with torch.no_grad():
        category, logits = cat_model.sample(
            input_img, cats, return_logits=True
        )
        category = int(category[0])
    return category

def sample_location(loc_model, input_img, category, debug_dir = None):
    with torch.no_grad():
        outputs = loc_model(input_img)
        outputs = F.softmax(outputs, dim = 1)
        outputs = F.upsample(outputs, mode='bilinear', scale_factor=4).squeeze()[category]
        # Mask out locations occupied by objects and outside room 
        current_room = input_img.squeeze(0)
        outputs[current_room[0] == 1] = 0
        outputs[current_room[1] > 0] = 0
        location_map = outputs.cpu()

    location_map = location_map / location_map.sum()
    loc_idx = int(torch.distributions.Categorical(probs = location_map.view(-1)).sample())
    location_map = location_map.cpu().numpy()
    x, y = divmod(loc_idx, 256)

    if debug_dir is not None:
        Image.fromarray(
            np.uint8((current_room[0] == 1).cpu() * 255)
        ).save(debug_dir / "object_mask.jpg")
        Image.fromarray(
            np.uint8((current_room[1] > 0).cpu() * 255)
        ).save(debug_dir / "room_mask.jpg")

        scene_img = np.array(Image.open(debug_dir / "scene.jpg"))
        color_map = np.zeros((location_map.shape[0], location_map.shape[1], 3)) 
        color_map[..., 0] = (location_map / location_map.max()) * 255
        color_map[x, y] = [255, 255, 255]
        Image.fromarray(np.uint8(color_map)).save(debug_dir / "heatmap.jpg")

        mask = (color_map[..., 0] > 50).nonzero()
        scene_img[mask] = color_map[mask] 
        heatmap_img = Image.fromarray(np.uint8(scene_img))
        heatmap_img.save(debug_dir / "scene_heatmap.jpg")

    x = ((x / 256) - 0.5) * 2
    y = ((y / 256) - 0.5) * 2

    return x, y

def sample_orientation(orient_model, input_img, category):
    """
    TODO - reference fast_synth.py line 520
    I'm pretty sure can just call directly
    """
    noise = torch.randn(1, orient_latent_size).to(input_img.device)
    return orient_model.generate(noise, input_img, category)

def sample_dimensions(dims_model, input_img, category):
    """
    TODO  - reference fast_synth.py line 526
    I'm pretty sure can just call model directly
    """
    noise = torch.randn(1, dims_latent_size).to(input_img.device)
    return dims_model.generate(noise, input_img, category)

def generate_scene(scene, cat_model, loc_model, orient_model, dims_model, device, debug_dir = None):
    iteration = 0
    while True:
        if debug_dir is not None:
            save_dir = debug_dir / str(iteration)
            save_dir.mkdir(exist_ok = True, parents = True)

        input_img = torch.tensor(
            scene.to_fastsynth_inputs(), dtype = torch.float32
        ).unsqueeze(0).to(device)
        cats = torch.tensor(
            scene.get_bag_of_categories(), dtype = torch.float32
        ).unsqueeze(0).to(device)

        category = sample_category(cat_model, input_img, cats)
        if category == 0:
            break

        if debug_dir is not None:
            save_input_img_as_png(input_img.cpu(), save_path=save_dir / "scene.jpg")
        x, y = sample_location(loc_model, input_img, category, debug_dir = save_dir)
        """
        TODO translate input image - reference fast_synth.py 514 - 519
        generate input_img_orient, scene needs to be translated according to prediction of location
        """
        translation = torch.tensor([[x, y]], device=input_img.device)
        orientation = torch.tensor([[math.cos(0), math.sin(0)]], device=input_img.device)

        input_img_orient = inverse_xform_img(
            input_img, translation, orientation, output_size=input_img.shape[-1]
        )

        if debug_dir is not None:
            save_input_img_as_png(input_img_orient.cpu(), save_path=save_dir / "scene_orient.jpg")
        orient = sample_orientation(orient_model, input_img_orient, category)
        """
        TODO rotate input image - reference fast_synth.py 522 - 524
        generate input_img_dims, scene needs to be rotated according to output of orientation
        """
        orientation = torch.tensor([orient], device = input_img.device)
        
        input_img_dims = inverse_xform_img(
            input_img, translation, orientation, output_size=input_img.shape[-1]
        )

        if debug_dir is not None:
            save_input_img_as_png(input_img_dims.cpu(), save_path=save_dir / "scene_dims.jpg")
        dims = sample_dimensions(dims_model, input_img_dims, category)

if __name__ == '__main__':
    from src.config import data_filepath
    from src.object.config import object_types
    from src.io_utils import read_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, type=Path, help="save directory for models")
    parser.add_argument("--out-dir", required=True, type=Path, help="where to output results")
    parser.add_argument("--debug", action="store_true")
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

    debug_dir = args.out_dir / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for scene_idx, scene in enumerate(scenes):
        scene_copy = scene.copy(empty=True)
        if args.debug:
            scene_debug_dir = debug_dir / f"scene_{scene_idx}" 
        else:
            scene_debug_dir = None

        generate_scene(
            scene_copy, cat_model, loc_model, orient_model, dims_model, device,
            debug_dir = scene_debug_dir 
        )
