import pickle 
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from numba import jit
import torch

from threedftoolbox.render.render_depth import render
from threedftoolbox.atiss_furniture_config import \
        THREED_FRONT_BEDROOM_FURNITURE, \
        THREED_FRONT_LIVINGROOM_FURNITURE, \
        THREED_FRONT_LIBRARY_FURNITURE

def get_rot_matrix(theta):
    costheta = float(np.cos(theta))
    sintheta = float(np.sin(theta))

    rotation_m = np.asarray([
            [costheta,0,sintheta],
            [0,1,0],
            [-sintheta,0,costheta],
            ])
    return rotation_m

@jit(nopython=True)
def get_triangles(verts, faces):
    result = np.zeros((len(faces),3,3),dtype=np.float64)
    for i,face in enumerate(faces):
        result[i] = np.stack((verts[face[0]][:3],verts[face[1]][:3],verts[face[2]][:3]))
    return result

def render_orthographic(verts, faces, corner_pos, cell_size, grid_size, flat = True):
    new_verts = np.clip(
        (verts - corner_pos) / cell_size, 
        np.array([0, 0, 0],dtype=np.float64), 
        np.array([grid_size, grid_size, grid_size],dtype=np.float64
        )
    )
    new_verts = new_verts[:,np.array([0,2,1],dtype=np.int64)]
    triangles = get_triangles(new_verts, faces)
    img = render(triangles, grid_size, flat = flat)
    return img

def get_threedf_to_atiss_category(room_type):
    if room_type == "bedroom":
        return THREED_FRONT_BEDROOM_FURNITURE
    elif room_type == "living_room":
        return THREED_FRONT_LIVINGROOM_FURNITURE
    elif room_type == "library":
        return THREED_FRONT_LIBRARY_FURNITURE
    else:
        raise NotImplementedError(f"{room_type} not yet implemented")

def get_categories_list(room_type):
    if room_type == "bedroom":
        return np.unique(list(THREED_FRONT_BEDROOM_FURNITURE.values())).tolist()
    elif room_type == "living_room":
        return np.unique(list(THREED_FRONT_LIVINGROOM_FURNITURE.values())).tolist()
    elif room_type == "library":
        return np.unique(list(THREED_FRONT_LIBRARY_FURNITURE.values())).tolist()
    else:
        raise NotImplementedError(f"{room_type} not yet implemented")

class ThreedfFurniture():
    def __init__(self, category_id, rotation, size, translation):
        self.id = category_id

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
    
    def rasterize_to_mask(self, corner_pos, cell_size, grid_size):
        return render_orthographic(
            self.vertices,
            self.faces,
            corner_pos,
            cell_size,
            grid_size,
        )

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


class ThreedfScene():
    def __init__(
        self, 
        pickle_path, 
        threedf_to_atiss_category,
        categories,
        room_largest_dim, 
        grid_size
    ):
        with open(pickle_path, 'rb') as f:
            all_info = pickle.load(f)

        self.num_categories = len(categories) 
        self.scene_id = all_info["scene_id"]
        self.floor_verts = all_info["floor_verts"]
        self.floor_fs = all_info["floor_fs"]

        min_bound = np.amin(self.floor_verts, axis = 0)
        max_bound = np.amax(self.floor_verts, axis = 0)
        floor_centroid = np.mean([min_bound, max_bound], axis=0)

        self.corner_pos = -np.array(
            [room_largest_dim / 2.0, 0.0, room_largest_dim / 2.0]
        )
        self.cell_size = room_largest_dim / grid_size
        self.grid_size = grid_size

        self.furniture = []
        for instance, bbox in zip(all_info["furnitures"], all_info["bboxes"]):
            threedf_category = instance.info.category.lower().replace(' / ', '/')
            if threedf_category not in threedf_to_atiss_category:
                continue

            atiss_category = threedf_to_atiss_category[threedf_category] 
            category_id = categories.index(atiss_category)
            furniture_piece = ThreedfFurniture(
                category_id,
                bbox["rotation"],
                bbox["size"],
                bbox["translation"] - floor_centroid,
            )
            self.furniture.append(furniture_piece)

    # Given a floor mask, creates a binary mask of the edges of the floor
    def get_wall_mask(self, floor_mask, wall_value=0.5):
        # Initialize the wall mask with zeros
        wall_mask = np.zeros_like(floor_mask)
        # for wall in self.walls:
        #     wall.write_to_mask(self.corner_pos, bedroom_cell_size, wall_mask)
        # wall_mask[wall_mask.astype(bool)] = wall_value

        # Get the shape of the mask
        rows, cols = floor_mask.shape
        # Iterate through the array, including edges
        for row in range(rows):
            for col in range(cols):
                # Check if current cell is floor space
                if floor_mask[row, col] == 1:
                    # Check the neighbors, including edge cases
                    if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                        wall_mask[row, col] = wall_value
                    elif (
                        floor_mask[row - 1, col] == 0
                        or floor_mask[row + 1, col] == 0
                        or floor_mask[row, col - 1] == 0
                        or floor_mask[row, col + 1] == 0
                    ):
                        wall_mask[row, col] = wall_value
        return wall_mask

    # Convert scene object into a format that fastsynth can load
    # Returns a multi channel image (from page 4; https://dritchie.github.io/pdf/deepsynth.pdf):
    # 0: depth: depth from camera (1 if object occupies location, 0 otherwise)
    # 1: room mask: 1 if inside room boundaries, 0 otherwise
    # 2: wall mask: 0.5 for wall, 1 for door, 0 otherwise
    # 3: object mask: number of objects present at pixel (+0.5 per object)
    # 4: orientation (sin): rotation of object present around world up vector
    # 5: orientation (cos): rotation of object present around world up vector
    # 6+: category channels (1 for each category): number of objects of category x at pixel
    def to_fastsynth_inputs(self, object_indices = None):
        num_non_cat_channels = 6
        num_channels = num_non_cat_channels + self.num_categories 

        fastsynth_input = np.zeros((num_channels, self.grid_size, self.grid_size))

        # start with floor and wall
        floor_mask = render_orthographic(
            self.floor_verts, self.floor_fs, self.corner_pos, self.cell_size, self.grid_size
        )
        floor_mask = ~np.asarray(floor_mask, dtype=bool)
        floor_mask = np.asarray(floor_mask, dtype=np.float32)
        fastsynth_input[1] = floor_mask

        # wall
        wall = self.get_wall_mask(floor_mask)
        fastsynth_input[2] = wall

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

    def get_bag_of_categories(self, object_indices = None):
        bag = np.zeros(self.num_categories)
        if object_indices is not None:
            furniture_in_scene = self.furniture
        else:
            furniture_in_scene = np.array(self.furniture)[object_indices].tolist()

        for furniture in furniture_in_scene:
            bag[furniture.id] += 1
        return bag

class ThreedfDataset():
    def __init__(
            self, 
            input_dir, 
            dataset_type,
            room_type, 
            bounds_file_path, 
            grid_size
        ):
        threedf_to_atiss_category = get_threedf_to_atiss_category(room_type)
        self.categories = get_categories_list(room_type)

        with open(bounds_file_path, 'rb') as f:
            bounds = yaml.safe_load(f)
        room_largest_dim = bounds[room_type]["largest_allowed_dim"] 

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
                room_largest_dim, 
                grid_size
            )
            scenes.append(scene)

        self.dataset_type = dataset_type
        self.cat_to_idx_list = None
        print(f"Caching data for {self.dataset_type} scene dataset")
        if self.dataset_type in ['cat', 'loc']: 
            self.data = scenes 
        elif self.dataset_type == 'orient_dims':
            assert indices is not None
            for meta_idx in tqdm(indices):
                item = meta[meta_idx]
                scene = original_scenes[item['scene_idx']]
                object_indices = item['object_indices']
                query_index = item['query_idx']
                new_scene, query_object = scene.subsample(object_indices, query_index)
                self.data.append((new_scene, query_object))
        else:
            raise ValueError(f"{self.dataset_type} not recognized")

    # From LatentDataset of Fastsynth
    def prepare_same_category_batches(self, batch_size):
        # Build a random list of category indices (grouped by batch_size)
        # This requires than length of dataset is a multiple of batch_size
        if len(self) % batch_size != 0:
            num_batches = len(self) // batch_size
            self.data = self.data[:num_batches * batch_size]

        if self.cat_to_idx_list is None:
            self.cat_to_idx_list = defaultdict(list)
            for idx, vals in enumerate(self.data):
                if self.dataset_type == 'orient_dims':
                    _, query_object = vals
                self.cat_to_idx_list[query_object.id].append(idx)

        assert(len(self) % batch_size == 0)
        num_batches = len(self) // batch_size
        self.same_category_batch_indices = []
        for _ in range(num_batches):
            cat_index = np.random.randint(1, len(object_types))
            for _ in range(batch_size):
                self.same_category_batch_indices.append(cat_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cat_to_idx_list is not None:
            cat = self.same_category_batch_indices[idx]
            same_cat_idx = np.random.choice(self.cat_to_idx_list[cat])
            idx = same_cat_idx
            
        if self.dataset_type == "cat":
            scene = self.data[idx]

            indices = np.arange(len(scene.furniture))
            if len(indices) == 1:
                object_indices = []
                query_index = 0
            else:
                np.random.shuffle(indices)
                num_objects = np.random.randint(low = 0, high = len(indices) - 1)
                object_indices = indices[:num_objects]
                query_index = indices[num_objects]

            input_img_raw = scene.to_fastsynth_inputs(object_indices = object_indices)
            catcount_raw = scene.get_bag_of_categories(object_indices = object_indices)

            t_cat_raw = scene.furniture[query_index].id

            input_img = torch.tensor(input_img_raw, dtype=torch.float32)
            t_cat = torch.tensor(t_cat_raw, dtype=torch.long)
            catcount = torch.tensor(catcount_raw, dtype=torch.float32)

            return input_img, t_cat, catcount
        elif self.type == 'loc':
            scene = self.data[idx]

            indices = np.arange(len(scene.objects))
            np.random.shuffle(indices)
            num_objects = np.random.randint(low = 0, high = len(indices))

            input_img_raw = scene.to_fastsynth_inputs(object_indices = object_indices)

            inputs = torch.tensor(input_img_raw, dtype=torch.float32)
            output = torch.zeros((int(self.grid_size / 4), int(self.grid_size / 4))).long()
            for object_idx in indices[num_objects:]:
                furniture_piece = scene.furniture[object_idx]
                coords = (furniture_piece.center - scene.corner_pos) / (scene.cell_size * 4)
                x_coord = int(coords[0])
                y_coord = int(coords[2])

                output[x_coord, y_coord] = furniture_piece.id

            return inputs, output
        elif self.type == 'orient_dims':
            scene, query_object = self.data[idx]
            if self.cat_to_idx_list is not None:
                assert query_object.id == cat

            cat = np.array([cat])

            catcount = scene.get_bag_of_categories()
            input_img = scene.to_fastsynth_inputs()

            # Select just the object mask channel from the output image
            output_img = np.zeros((grid_size, grid_size))
            query_object.write_to_mask(scene.corner_pos, bedroom_cell_size, output_img)

            # Normalize the coordinates to [-1, 1], with (0,0) being the image center
            loc = query_object.center
            x_loc = loc[0] / (bedroom_largest_dim / 2)
            y_loc = loc[2] / (bedroom_largest_dim / 2)
            loc = np.array([x_loc, y_loc])

            # Get the orientation of the object
            sin, cos = query_object.get_rotation()
            orient = np.array([cos, sin])

            # Get the object-space dimensions of the output object (in pixel space)
            # (Normalize to [0, 1])
            xsize, _, ysize = query_object.extent / (bedroom_largest_dim / 2)
            dims = np.array([ysize, xsize])

            return input_img, output_img, cat, loc, orient, dims, catcount
        else:
            raise NotImplementedError(f"{self.type} not a recognized scene dataset")
