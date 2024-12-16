import pickle
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from numba import jit
import cv2
import torch
import random
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from threedftoolbox.render.render_depth import render
from threedftoolbox.atiss_furniture_config import (
    THREED_FRONT_BEDROOM_FURNITURE,
    THREED_FRONT_LIVINGROOM_FURNITURE,
    THREED_FRONT_LIBRARY_FURNITURE,
)


def get_rot_matrix(theta):
    costheta = float(np.cos(theta))
    sintheta = float(np.sin(theta))

    rotation_m = np.asarray(
        [
            [costheta, 0, sintheta],
            [0, 1, 0],
            [-sintheta, 0, costheta],
        ]
    )
    return rotation_m


@jit(nopython=True)
def get_triangles(verts, faces):
    result = np.zeros((len(faces), 3, 3), dtype=np.float64)
    for i, face in enumerate(faces):
        result[i] = np.stack(
            (verts[face[0]][:3], verts[face[1]][:3], verts[face[2]][:3])
        )
    return result


def render_orthographic(verts, faces, corner_pos, cell_size, grid_size, flat=True):
    new_verts = np.clip(
        (verts - corner_pos) / cell_size,
        np.array([0, 0, 0], dtype=np.float64),
        np.array([grid_size, grid_size, grid_size], dtype=np.float64),
    )
    new_verts = new_verts[:, np.array([0, 2, 1], dtype=np.int64)]
    triangles = get_triangles(new_verts, faces)
    img = render(triangles, grid_size, flat=flat)
    return img


def get_threedf_to_atiss_category(room_type):
    if room_type == "bedroom":
        return THREED_FRONT_BEDROOM_FURNITURE
    elif room_type == "livingroom":
        return THREED_FRONT_LIVINGROOM_FURNITURE
    elif room_type == "diningroom":
        return THREED_FRONT_LIVINGROOM_FURNITURE
    elif room_type == "library":
        return THREED_FRONT_LIBRARY_FURNITURE
    else:
        raise NotImplementedError(f"{room_type} not yet implemented")


def get_categories_list(room_type):
    if room_type == "bedroom":
        return ["stop"] + np.unique(
            list(THREED_FRONT_BEDROOM_FURNITURE.values())
        ).tolist()
    elif room_type == "livingroom":
        return ["stop"] + np.unique(
            list(THREED_FRONT_LIVINGROOM_FURNITURE.values())
        ).tolist()
    elif room_type == "diningroom":
        return ["stop"] + np.unique(
            list(THREED_FRONT_LIVINGROOM_FURNITURE.values())
        ).tolist()
    elif room_type == "library":
        return ["stop"] + np.unique(
            list(THREED_FRONT_LIBRARY_FURNITURE.values())
        ).tolist()
    else:
        raise NotImplementedError(f"{room_type} not yet implemented")


class ThreedfFurniture:
    def __init__(self, category_id, rotation, size, translation):
        self.id = category_id

        self.size = size
        self.extent = 2 * size
        max_bound = size
        min_bound = -size
        self.vertices = np.array(
            [
                [min_bound[0], 0, max_bound[2]],
                [max_bound[0], 0, max_bound[2]],
                [min_bound[0], 0, min_bound[2]],
                [max_bound[0], 0, min_bound[2]],
            ]
        )
        self.faces = np.array([[0, 3, 1], [0, 2, 3]])
        self.center = np.array([0.0, 0.0, 0.0])
        self.rot = 0

        self.rotate(-rotation)
        self.translate(translation)

        self.mask = None

    def rasterize_to_mask(self, corner_pos, cell_size, grid_size):
        if self.mask is None:
            self.mask = render_orthographic(
                self.vertices,
                self.faces,
                corner_pos,
                cell_size,
                grid_size,
            )
        return self.mask

    def rotate(self, theta):
        """
        theta : float of rotation given in radians (in list)
        """
        self.rot += theta[0]
        if self.rot > 2 * np.pi:
            self.rot = 2 * np.pi - self.rot
        elif self.rot < 0:
            self.rot += 2 * np.pi

        rot_matrix = get_rot_matrix(theta)
        self.vertices = np.matmul(self.vertices, rot_matrix)

    def translate(self, translation):
        translation[1] = 0
        self.vertices += translation
        self.center += translation


class ThreedfScene:
    def __init__(
        self,
        pickle_path,
        threedf_to_atiss_category,
        categories,
        room_largest_dim,
        grid_size,
    ):
        with open(pickle_path, "rb") as f:
            all_info = pickle.load(f)

        self.categories = categories
        self.num_categories = len(categories)
        self.scene_id = all_info["scene_id"]
        self.floor_verts = all_info["floor_verts"]
        self.floor_fs = all_info["floor_fs"]

        self.corner_pos = -np.array(
            [room_largest_dim / 2.0, 0.0, room_largest_dim / 2.0]
        )
        self.cell_size = room_largest_dim / grid_size
        self.grid_size = grid_size

        self.floor_mask = render_orthographic(
            self.floor_verts,
            self.floor_fs,
            self.corner_pos,
            self.cell_size,
            self.grid_size,
        )
        self.floor_mask = np.asarray(self.floor_mask, dtype=bool)

        self.furniture = []
        for instance, bbox in zip(all_info["furnitures"], all_info["bboxes"]):
            threedf_category = instance.info.category.lower().replace(" / ", "/")
            if threedf_category not in threedf_to_atiss_category:
                continue

            atiss_category = threedf_to_atiss_category[threedf_category]
            category_id = categories.index(atiss_category)
            if any(abs(v) > room_largest_dim / 2.0 for v in bbox["translation"]):
                continue

            furniture_piece = ThreedfFurniture(
                category_id,
                bbox["rotation"],
                bbox["size"],
                bbox["translation"],
            )
            self.furniture.append(furniture_piece)

    # Convert scene object into a format that fastsynth can load
    # Returns a multi channel image (from page 4; https://dritchie.github.io/pdf/deepsynth.pdf):
    # 0: depth: depth from camera (1 if object occupies location, 0 otherwise)
    # 1: room mask: 1 if inside room boundaries, 0 otherwise
    # 2: wall mask: 0.5 for wall, 1 for door, 0 otherwise
    # 3: object mask: number of objects present at pixel (+0.5 per object)
    # 4: orientation (sin): rotation of object present around world up vector
    # 5: orientation (cos): rotation of object present around world up vector
    # 6+: category channels (1 for each category): number of objects of category x at pixel
    def to_fastsynth_inputs(self, object_indices=None):
        num_non_cat_channels = 6
        num_channels = num_non_cat_channels + self.num_categories

        fastsynth_input = np.zeros((num_channels, self.grid_size, self.grid_size))

        # start with floor and wall
        floor_mask = np.array(self.floor_mask, dtype=np.float32)
        fastsynth_input[1] = floor_mask

        # wall
        # https://stackoverflow.com/questions/72215748/how-to-extend-mask-region-true-by-1-or-2-pixels
        wall_mask = np.array(floor_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
        wall_mask[floor_mask.astype(bool)] = 0
        wall_mask[wall_mask.astype(bool)] = 0.5

        fastsynth_input[2] = wall_mask

        if object_indices is None:
            furniture_in_scene = self.furniture
        else:
            furniture_in_scene = np.array(self.furniture)[object_indices].tolist()

        # object specific channels (uses helper method)
        for furniture_piece in furniture_in_scene:
            # figure out which channel this object belongs to
            obj_channel_index = num_non_cat_channels + furniture_piece.id - 1

            # start with mask of proper dimensions with 1s where object is
            furniture_mask = furniture_piece.rasterize_to_mask(
                self.corner_pos, self.cell_size, self.grid_size
            ).astype(bool)

            # loop: depth +1 (max 1), object +0.5, orientation, proper cat
            # update depth mask (set to 1)
            fastsynth_input[0][furniture_mask] = 1
            # update object mask (+0.5)
            fastsynth_input[3][furniture_mask] += 0.5
            # update orientation masks
            sin, cos = np.sin(-furniture_piece.rot), np.cos(-furniture_piece.rot)
            fastsynth_input[4][furniture_mask] = sin
            fastsynth_input[5][furniture_mask] = cos

            # update category
            fastsynth_input[obj_channel_index][furniture_mask] += 1

        return fastsynth_input

    def get_bag_of_categories(self, object_indices=None):
        bag = np.zeros(self.num_categories)
        if object_indices is None:
            furniture_in_scene = self.furniture
        else:
            furniture_in_scene = np.array(self.furniture)[object_indices].tolist()

        for furniture in furniture_in_scene:
            bag[furniture.id] += 1
        return bag

    def convert_to_image(self):
        rgb_image = np.zeros((self.grid_size, self.grid_size, 3))
        floor_mask = render_orthographic(
            self.floor_verts,
            self.floor_fs,
            self.corner_pos,
            self.cell_size,
            self.grid_size,
        ).astype(bool)
        rgb_image[floor_mask] = np.array([169, 169, 169]) / 256
        rgb_image[~floor_mask] = np.array([222, 222, 222]) / 256
        colors = plt.cm.get_cmap("tab20", self.num_categories)
        for furniture_piece in self.furniture:
            furniture_mask = furniture_piece.rasterize_to_mask(
                self.corner_pos, self.cell_size, self.grid_size
            ).astype(bool)
            rgb_image[furniture_mask] = colors(furniture_piece.id)[:3]
            furniture_piece.rot

            front_centroid = (furniture_piece.vertices[0] + furniture_piece.vertices[1]) / 2
            coords = (front_centroid - self.corner_pos) / self.cell_size
            x_coord = int(coords[0])
            y_coord = int(coords[2])
            rgb_image[x_coord-1 : x_coord+1, y_coord-1 : y_coord+1] = [1.0, 0, 0]

        return rgb_image

    def to_json(self, output_path):
        output_dict = {}
        output_dict["vertices"] = self.floor_verts.tolist()
        output_dict["faces"] = self.floor_fs.tolist()
        output_dict["scene_id"] = self.scene_id

        objects = []
        for furniture_piece in self.furniture:
            object_info = {
                "translation": furniture_piece.center.tolist(),
                "size": furniture_piece.size.tolist(),
                "rotation": [furniture_piece.rot],
                "category": self.categories[furniture_piece.id],
            }
            objects.append(object_info)

        output_dict["objects"] = objects

        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)


class ThreedfDataset(Dataset):
    def __init__(
            self, 
            input_dir, 
            dataset_type, 
            room_type, 
            bounds_file_path, 
            grid_size,
            use_ordering=False,
            scene_ids=None,
        ):
        threedf_to_atiss_category = get_threedf_to_atiss_category(room_type)
        self.categories = get_categories_list(room_type)
        self.dataset_type = dataset_type
        self.cat_to_idx_list = None
        self.grid_size = grid_size

        with open(bounds_file_path, "rb") as f:
            bounds = yaml.safe_load(f)
        self.room_largest_dim = bounds[room_type]["largest_allowed_dim"]

        scenes = []
        folders = list(Path(input_dir).iterdir())
        for folder in tqdm(folders):
            if not (folder / "all_info.pkl").exists():
                continue
            pickle_path = folder / "all_info.pkl"
            scene = ThreedfScene(
                pickle_path,
                threedf_to_atiss_category,
                self.categories,
                self.room_largest_dim,
                grid_size,
            )
            if len(scene.furniture) > 0:
                scenes.append(scene)

        if scene_ids is None:
            self.scenes = scenes
        else:
            self.scenes = [scene for scene in scenes if scene.scene_id in scene_ids]

        if use_ordering:
            # Extract basic dataset statistics to generate object ordering
            # score = average size of category * frequency of occurence
            category_id_to_sizes = defaultdict(list)
            total_num_objects = 0
            for scene in scenes:
                for object in scene.objects:
                    size = object.size
                    category_id_to_sizes[object.id].append((size[0] + size[2]) / 2)
                    total_num_objects += 1

            category_id_to_score = dict()
            for category_id, sizes in category_id_to_sizes.items():
                category_id_to_score[category_id] = np.mean(sizes) * (
                    len(sizes) / total_num_objects
                )
            self.category_id_to_score = category_id_to_score
        else:
            self.category_id_to_score = None

    # From LatentDataset of Fastsynth
    def prepare_same_category_batches(self, batch_size):
        # Build a random list of category indices (grouped by batch_size)
        # This requires than length of dataset is a multiple of batch_size
        if len(self) % batch_size != 0:
            num_batches = len(self) // batch_size
            self.scenes = self.scenes[: num_batches * batch_size]

        if self.cat_to_idx_list is None:
            # Just build the list such that the desired category is in the scene
            self.cat_to_idx_list = defaultdict(list)
            for idx, scene in enumerate(self.scenes):
                for furniture_piece in scene.furniture:
                    self.cat_to_idx_list[furniture_piece.id].append(idx)
            self.available_cats = []
            for cat_id, indices in self.cat_to_idx_list.items():
                if len(indices) > batch_size:
                    self.available_cats.append(cat_id)

        assert len(self) % batch_size == 0
        num_batches = len(self) // batch_size
        self.same_category_batch_indices = []
        for _ in range(num_batches):
            cat_index = random.choice(self.available_cats)
            for _ in range(batch_size):
                self.same_category_batch_indices.append(cat_index)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.cat_to_idx_list is not None:
            cat = self.same_category_batch_indices[idx]
            same_cat_idx = np.random.choice(self.cat_to_idx_list[cat])
            idx = same_cat_idx

        if self.dataset_type == "cat":
            scene = self.scenes[idx]
            indices = np.arange(len(scene.furniture))
            np.random.shuffle(indices)

            if self.category_id_to_score is None:
                # Want to also include choosing entire scene and predicting stop
                num_objects = np.random.randint(low=0, high=len(indices) + 1)
                object_indices = indices[:num_objects]

                if num_objects == len(indices):
                    t_cat_raw = self.categories.index("stop")
                else:
                    query_index = indices[num_objects]
                    t_cat_raw = scene.furniture[query_index].id
            else:
                while True:
                    num_objects = np.random.randint(low=0, high=len(indices) + 1)
                    object_indices = indices[:num_objects]

                    if num_objects == len(indices):
                        t_cat_raw = self.categories.index("stop")
                        break
                    else:
                        query_index = indices[num_objects]
                        t_cat_raw = scene.furniture[query_index].id
                        query_score = self.category_id_to_score[t_cat_raw]
                        valid = True
                        for obj_idx in object_indices:
                            obj_id = scene.furniture[obj_idx].id
                            if query_score > self.category_id_to_score[obj_id]:
                                valid = False
                        if valid:
                            break

            input_img_raw = scene.to_fastsynth_inputs(object_indices=object_indices)
            catcount_raw = scene.get_bag_of_categories(object_indices=object_indices)

            input_img = torch.tensor(input_img_raw, dtype=torch.float32)
            t_cat = torch.tensor(t_cat_raw, dtype=torch.long)
            catcount = torch.tensor(catcount_raw, dtype=torch.float32)

            return input_img, t_cat, catcount
        elif self.dataset_type == "loc":
            scene = self.scenes[idx]
            indices = np.arange(len(scene.furniture))
            np.random.shuffle(indices)
            num_objects = np.random.randint(low=0, high=len(indices))
            object_indices = indices[:num_objects]

            input_img_raw = scene.to_fastsynth_inputs(object_indices=object_indices)

            inputs = torch.tensor(input_img_raw, dtype=torch.float32)
            output = torch.zeros(
                (int(self.grid_size / 4), int(self.grid_size / 4))
            ).long()

            for object_idx in indices[num_objects:]:
                furniture_piece = scene.furniture[object_idx]
                coords = (furniture_piece.center - scene.corner_pos) / (
                    scene.cell_size * 4
                )
                x_coord = int(coords[0])
                y_coord = int(coords[2])

                output[x_coord, y_coord] = furniture_piece.id

            return inputs, output
        elif self.dataset_type == "orient_dims":
            scene = self.scenes[idx]

            # The desired category is guaranteed to be in the scene
            indices = np.arange(len(scene.furniture))
            np.random.shuffle(indices)

            query_index = None
            for idx in indices:
                if scene.furniture[idx].id == cat:
                    query_index = idx
                    break
            assert query_index is not None

            # Remove query index from list of options
            indices = indices[indices != query_index]

            num_objects = np.random.randint(low=0, high=len(indices) + 1)
            object_indices = indices[:num_objects]
            query_object = scene.furniture[query_index]

            assert query_object.id == cat

            cat = np.array([cat])
            catcount = scene.get_bag_of_categories(object_indices=object_indices)
            input_img = scene.to_fastsynth_inputs(object_indices=object_indices)

            # Select just the object mask channel from the output image
            output_img = query_object.rasterize_to_mask(
                scene.corner_pos, scene.cell_size, scene.grid_size
            )

            # Normalize the coordinates to [-1, 1], with (0,0) being the image center
            loc = query_object.center
            x_loc = loc[0] / (self.room_largest_dim / 2)
            y_loc = loc[2] / (self.room_largest_dim / 2)
            loc = np.array([x_loc, y_loc])

            # Get the orientation of the object
            sin, cos = np.sin(-query_object.rot), np.cos(-query_object.rot)
            orient = np.array([cos, sin])

            # Get the object-space dimensions of the output object (in pixel space)
            # (Normalize to [0, 1])
            xsize, _, ysize = query_object.extent / (self.room_largest_dim / 2)
            dims = np.array([ysize, xsize])

            return input_img, output_img, cat, loc, orient, dims, catcount
        else:
            raise NotImplementedError(
                f"{self.dataset_type} not a recognized scene dataset"
            )
