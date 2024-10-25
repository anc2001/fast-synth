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

from threedf_dataset import ThreedfDataset, ThreedfFurniture, get_categories_list


def sample_category(cat_model, input_img, cats):
    with torch.no_grad():
        category, logits = cat_model.sample(input_img, cats, return_logits=True)
        category = int(category[0])
    return category


def sample_location(loc_model, input_img, category, return_map=False, debug_dir=None):
    with torch.no_grad():
        outputs = loc_model(input_img)
        outputs = F.softmax(outputs, dim=1)
        outputs = F.interpolate(outputs, mode="bilinear", scale_factor=4).squeeze()[
            category
        ]
        # Mask out locations occupied by objects and outside room
        current_room = input_img.squeeze(0)
        outputs[current_room[0] == 1] = 0
        outputs[current_room[1] > 0] = 0
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


def sample_orientation(orient_model, input_img, category):
    noise = torch.randn(1, orient_latent_size).to(input_img.device)
    return orient_model.generate(noise, input_img, category)


def sample_dimensions(dims_model, input_img, category):
    noise = torch.randn(1, dims_latent_size).to(input_img.device)
    return dims_model.generate(noise, input_img, category)


def generate_scene(
    scene,
    categories,
    room_largest_dim,
    cat_model,
    loc_model,
    orient_model,
    dims_model,
    device,
    debug_dir=None,
):
    iteration = 0
    while True:
        if debug_dir is not None:
            save_dir = debug_dir / str(iteration)
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = None

        input_img = (
            torch.tensor(scene.to_fastsynth_inputs(), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        cats = (
            torch.tensor(scene.get_bag_of_categories(), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        category = sample_category(cat_model, input_img, cats)
        if categories[category] == "stop":
            break

        x, y = sample_location(loc_model, input_img, category, debug_dir=save_dir)

        translation = torch.tensor([[x, y]], device=input_img.device)
        orientation = torch.tensor(
            [[math.cos(0), math.sin(0)]], device=input_img.device
        )

        input_img_orient = inverse_xform_img(
            input_img, translation, orientation, output_size=input_img.shape[-1]
        )

        if debug_dir is not None:
            save_input_img_as_png(
                input_img_orient.cpu(), save_path=save_dir / "scene_orient.jpg"
            )
        orientation = sample_orientation(orient_model, input_img_orient, category)

        input_img_dims = inverse_xform_img(
            input_img, translation, orientation, output_size=input_img.shape[-1]
        )

        if debug_dir is not None:
            save_input_img_as_png(
                input_img_dims.cpu(), save_path=save_dir / "scene_dims.jpg"
            )
        dims = sample_dimensions(dims_model, input_img_dims, category)

        # Convert all relevant values back
        multiplier = room_largest_dim / 2
        x_dims = dims[0, 1].item() * multiplier
        y_dims = dims[0, 0].item() * multiplier

        size = np.array([x_dims, 0, y_dims]) / 2

        # Given rotation is ccw, ours is cw
        cos, sin = (orientation[0, 0].item(), orientation[0, 1].item())
        angle = math.atan2(sin, cos)
        rotation = np.array([-angle])

        # x and y in normalized image space
        translation = np.array([x, 0, y]) * multiplier

        query_object = ThreedfFurniture(category, rotation, size, translation)
        scene.furniture.append(query_object)
        if debug_dir is not None:
            img = scene.convert_to_image()
            Image.fromarray(np.uint8(img * 255)).save(save_dir / "scene_final.jpg")

        iteration += 1

        if debug_dir is not None:
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(4 * 4, 6))
            # heatmap
            axs[0].imshow(Image.open(save_dir / "heatmap.jpg"))
            axs[0].set_title("heatmap")
            # Location heatmap
            axs[1].imshow(Image.open(save_dir / "scene_heatmap.jpg"))
            axs[1].set_title(f"scene heatmap : {categories[category]}")
            # Image given to orient
            axs[2].imshow(Image.open(save_dir / "scene_orient.jpg"))
            axs[2].set_title("orient image")
            # Image given to dims
            axs[3].imshow(Image.open(save_dir / "scene_dims.jpg"))
            axs[3].set_title(f"dims image")
            # Final image
            axs[4].imshow(Image.open(save_dir / "scene_final.jpg"))
            axs[4].set_title("final image")

            for ax in axs.flat:
                ax.axis("off")

            fig.set_tight_layout(True)
            fig.savefig(save_dir / "collate.jpg")
            plt.close(fig)

    return scene


def load_cat_model(checkpoint_path, num_input_channels, num_categories, device):
    print("Loading Category Model")
    cat_model = NextCategory(num_input_channels, num_categories, cat_latent_dim)
    cat_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    cat_model = cat_model.to(device)
    cat_model.eval()

    return cat_model


def load_loc_model(checkpoint_path, num_input_channels, num_categories, device):
    print("Loading Location Model")
    loc_model = LocModel(
        num_classes=num_categories, num_input_channels=num_input_channels
    )
    loc_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    loc_model = loc_model.to(device)
    loc_model.eval()

    return loc_model


def load_orient_model(checkpoint_path, num_input_channels, device):
    print("Loading Orientation Model")
    orient_model = OrientModel(
        latent_size=orient_latent_size,
        hidden_size=orient_hidden_size,
        num_input_channels=num_input_channels,
    )
    orient_model.load(checkpoint_path)
    orient_model.testing = True
    orient_model = orient_model.to(device)
    orient_model.eval()

    return orient_model


def load_dims_model(checkpoint_path, num_input_channels, device):
    print("Loading Dimensions Model")
    dims_model = DimsModel(
        latent_size=dims_latent_size,
        hidden_size=dims_hidden_size,
        num_input_channels=num_input_channels,
    )
    dims_model.load(checkpoint_path)
    dims_model = dims_model.to(device)
    dims_model.eval()

    return dims_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir", required=True, type=Path, help="save directory for models"
    )
    parser.add_argument("--num-scenes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cat-name", type=str, required=True)
    parser.add_argument("--dims-name", type=str, required=True)
    parser.add_argument("--loc-name", type=str, required=True)
    parser.add_argument("--orient-name", type=str, required=True)

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    debug_dir = args.save_dir / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    cat_model = load_cat_model(
        args.save_dir / args.cat_name, num_input_channels, num_categories, device
    )
    loc_model = load_loc_model(
        args.save_dir / args.loc_name, num_input_channels, num_categories, device
    )
    orient_model = load_orient_model(
        args.save_dir / args.orient_name, num_input_channels, device
    )
    dims_model = load_dims_model(
        args.save_dir / args.dims_name, num_input_channels, device
    )

    np.random.seed(seed=args.seed)
    scenes = cat_dataset.scenes
    np.random.shuffle(scenes)
    generated_scenes = []
    for scene_idx, scene in enumerate(tqdm(scenes[: args.num_scenes])):
        if args.debug:
            scene_debug_dir = debug_dir / f"scene_{scene_idx:03d}"
        else:
            scene_debug_dir = None

        scene.furniture = []
        generated_scene = generate_scene(
            scene,
            categories,
            cat_dataset.room_largest_dim,
            cat_model,
            loc_model,
            orient_model,
            dims_model,
            device,
            debug_dir=scene_debug_dir,
        )
        generated_scenes.append(generated_scene)
