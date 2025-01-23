import argparse
from pathlib import Path
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import shutil
import math
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import json
import copy

from loc import Model as LocModel
from models.utils import inverse_xform_img

from utils import save_input_img_as_png

from threedf_dataset import ThreedfDataset, ThreedfFurniture, get_categories_list


def sample_location(loc_model, input_img, category, return_map=False, debug_dir=None):
    with torch.no_grad():
        outputs = loc_model(input_img)
        outputs = F.softmax(outputs, dim=1)
        outputs = F.interpolate(outputs, mode="bilinear", scale_factor=4).squeeze()[
            category
        ]
        # Mask out locations occupied by objects and outside room
        current_room = input_img.squeeze(0)
        outputs[current_room[1] == 0] = 0
        location_map = outputs.cpu()

    location_map = location_map / location_map.sum()
    loc_idx = int(torch.distributions.Categorical(probs=location_map.view(-1)).sample())
    location_map = location_map.cpu().numpy()
    x, y = divmod(loc_idx, 256)

    if debug_dir is not None:
        save_input_img_as_png(input_img.cpu(), save_path=debug_dir / "scene_start.jpg")
        scene_img = np.array(Image.open(debug_dir / "scene_start.jpg"))
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

    if return_map:
        return location_map, x, y
    else:
        return x, y

def generate_location_map(
    scene,
    query_id,
    loc_model,
    device,
    debug_dir=None,
):
    if debug_dir is not None:
        debug_dir.mkdir(parents=True)

    fastsynth_input = scene.to_fastsynth_inputs()
    input_img = (
        torch.tensor(fastsynth_input, dtype=torch.float32).unsqueeze(0).to(device)
    )

    location_map, _, _ = sample_location(
        loc_model, input_img, query_id, return_map=True, debug_dir=debug_dir
    )

    location_map = location_map / location_map.max()
    return location_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="save directory for models"
    )
    parser.add_argument(
        "--scene-json", required=True, type=Path, help="annotation info folder"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--room-type", type=str, required=True)
    parser.add_argument("--bounds-file", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)

    args = parser.parse_args()

    categories = get_categories_list(args.room_type)
    num_categories = len(categories)
    num_input_channels = num_categories + 6

    cat_dataset = ThreedfDataset(
        args.input_dir, "cat", args.room_type, args.bounds_file, args.grid_size
    )
    scenes = cat_dataset.scenes
    scene_id_to_scene = {scene.scene_id : scene for scene in scenes}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(args.scene_json, "r") as f:
        subscene_info = json.load(f)

    weight_file = Path(args.run_dir) / "location_100.pt"

    loc_model = LocModel(
        num_classes=num_categories, num_input_channels=num_input_channels
    )
    loc_model.load_state_dict(torch.load(weight_file, weights_only=True))
    loc_model = loc_model.to(device)
    loc_model.eval()

    output_directory = Path(args.scene_json).parent / "fastsynth_mask"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True) 
    debug_dir = output_directory / "debug"

    save_dir = output_directory

    scene = copy.deepcopy(scene_id_to_scene[subscene_info['scene_id']])

    scene.furniture = []
    for object_info in subscene_info["objects"]:
        category_id = scene.categories.index(object_info['category'])
        furniture_piece = ThreedfFurniture(
            category_id,
            np.array(object_info['rotation']),
            np.array(object_info['size']),
            np.array(object_info['translation']),
        )
        scene.furniture.append(furniture_piece)
    
    query_id = scene.categories.index(subscene_info["query_object"]["category"])
    location_map = generate_location_map(scene, query_id, loc_model, device)

    img = scene.convert_to_image()
    Image.fromarray(np.uint8(img * 255)).save(save_dir / 'scene.png')

    img[location_map > 0.5] = [1.0, 0.0, 0.0]
    Image.fromarray(np.uint8(img * 255)).save(save_dir / 'scene_dist.png')

    mask = location_map > 0.5
    np.savez(save_dir / 'masks.npz', masks = [mask])
