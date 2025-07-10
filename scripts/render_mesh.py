import argparse
import math
import os
from pathlib import Path
import shutil
from typing import Dict, Literal, Tuple

import bpy
from mathutils import Vector
from PIL import Image



# --- Blender Setup Functions ---
def global_settings():
    """Configures global Blender rendering settings."""
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    return scene


def add_lighting() -> None:
    """Adds area lights to the scene."""
    # Delete the default light
    if "Light" in bpy.data.objects:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()

    # Add a new large area light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100

    # Add a fill light
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 2))
    fill_obj = bpy.context.active_object
    fill_obj.data.energy = 2000
    fill_obj.scale = (10, 10, 10)


def reset_scene() -> None:
    """Resets the scene to a clean state by deleting all objects and data."""
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Delete all meshes
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)

    # Delete all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # Delete all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # Delete all images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

    # Delete all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)

    # Delete all cameras
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam, do_unlink=True)

    # Delete all empties and curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve, do_unlink=True)

    # Reset world
    if bpy.data.worlds:
        for world in bpy.data.worlds:
            bpy.data.worlds.remove(world, do_unlink=True)

    # Create a new default world
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.view_layer.update()


def load_object(object_path: str) -> None:
    """Loads a 3D model into the scene based on its file extension."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


# --- Scene Normalization and Utility Functions ---
def scene_bbox(single_obj=None, ignore_matrix=False):
    """Calculates the bounding box of the scene or a single object."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("No objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    """Generator for all root objects in the scene."""
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    """Generator for all mesh objects in the scene."""
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(target_scale=1.0):
    """Normalizes the scene: scales to fit target size and centers at the origin."""
    bbox_min, bbox_max = scene_bbox()
    size = bbox_max - bbox_min
    max_dim = max(size.x, size.y, size.z)
    if max_dim == 0:
        raise ValueError("Model has zero size. Cannot normalize.")

    scale = target_scale / max_dim
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    center = (bbox_min + bbox_max) * 0.5
    for obj in scene_root_objects():
        obj.location -= center

    bpy.context.view_layer.update()


# --- Camera and Lighting Setup ---
def setup_camera(scene):
    """Configures the camera and adds a tracking constraint."""
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
) -> bpy.types.Object:
    """Creates and returns a configured light object."""
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)

    light_object.location = location
    light_object.rotation_euler = rotation

    light_data.energy = energy
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor

    return light_object


def create_lighting() -> Dict[str, bpy.types.Object]:
    """Creates a deterministic multi-directional sun lighting setup."""
    # Remove existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Add 4 deterministic sun lights
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),  # 45°, -45° in radians
        energy=0.5,
    )
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),  # 45°, 135°
        energy=0.3,
    )
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),  # -45°, -225°
        energy=0.5,
    )
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),  # 180° (from below)
        energy=0.2,
    )
    return {
        "key_light": key_light,
        "fill_light": fill_light,
        "rim_light": rim_light,
        "bottom_light": bottom_light,
    }


# --- Main Rendering and Image Processing Functions ---
def render_object(
    object_file: str,
    output_dir: str,
    camera_views=[(30, 30, 1.5), (90, -20, 1.5), (150, 30, 1.5), (210, -20, 1.5), (270, 30, 1.5), (330, -20, 1.5)],
    background_color=(255, 255, 255)
) -> None:
    """Renders images of an object from multiple camera views."""
    scene = global_settings()
    os.makedirs(output_dir, exist_ok=True)
    reset_scene()

    # Create and set up a new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    scene.collection.objects.link(camera)
    scene.camera = camera

    scene.view_settings.view_transform = 'Standard'

    # Set background color
    world = bpy.data.worlds["World"]
    world.use_nodes = False
    world.color = tuple(channel / 255 for channel in background_color)
    scene.render.film_transparent = False
    scene.world = world

    # Load, normalize, and light the object
    load_object(object_file)
    normalize_scene()
    create_lighting()
    cam, cam_constraint = setup_camera(scene)

    # Create an empty object for the camera to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    for i, (azim, elev, camera_dist) in enumerate(camera_views):
        # Set camera position
        theta = math.radians(azim)
        phi = math.radians(elev)
        point = (
            camera_dist * math.cos(phi) * math.cos(theta),
            camera_dist * math.cos(phi) * math.sin(theta),
            camera_dist * math.sin(phi),
        )
        cam.location = point

        # Render the image
        render_path = os.path.join(output_dir, f"{i:02d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)


def create_tiled_grid(
    image_paths=["00.png", "01.png", "02.png", "03.png", "04.png", "05.png"],
    output_path="tiled_grid.png",
    tile_width=320,
    tile_height=320,
    background_color=(255, 255, 255),
):
    """Creates a 2x3 tiled grid image from a list of six image paths."""
    if len(image_paths) != 6:
        print("Error: Exactly 6 image paths are required.")
        return

    grid_width = tile_width * 2
    grid_height = tile_height * 3
    grid_image = Image.new("RGB", (grid_width, grid_height), background_color)

    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        img = img.resize((tile_width, tile_height))
        # Handle transparency by pasting onto a solid background
        if img.mode == "RGBA":
            background = Image.new("RGB", (tile_width, tile_height), background_color)
            background.paste(img, (0, 0), img)
            img = background

        x = (i % 2) * tile_width
        y = (i // 2) * tile_height
        grid_image.paste(img, (x, y))

    grid_image.save(output_path)
    print(f"Tiled grid image saved to: {output_path}")



# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 3D object into a multi-view  and source image format for EditP23.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input .glb or .fbx file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output src.png and src_mv.png.")
    parser.add_argument("--camera_dist", type=float, default=1.35, help="Camera distance from the object.")
    parser.add_argument("--azim_offset", type=float, default=0, help="Azimuthal offset for camera views in degrees.")
    args = parser.parse_args()

    RENDERS_SUBDIR = "all_renders"
    BACKGROUND_COLOR = (255, 255, 255)
    
    output_dir = Path(args.output_dir)
    renders_path = output_dir / RENDERS_SUBDIR


    ELEV_1 = 20
    ELEV_2 = -10
    elevs = [ELEV_1, ELEV_2] * 3
    azims = [(30 + 60 * i + args.azim_offset) % 360 for i in range(6)]
    camera_views = [(azim, elev, args.camera_dist) for azim, elev in zip(azims, elevs)] + [
        ((0 + args.azim_offset) % 360, ELEV_1, args.camera_dist)
    ]
    
    
    # Render the object from different views
    render_object(
        args.mesh_path,
        output_dir=str(renders_path),
        camera_views=camera_views,
        background_color=BACKGROUND_COLOR,
    )
    
    # --- Create Final Outputs ---
    image_paths_for_grid = [renders_path / f"{i:02d}.png" for i in range(6)]
    
    create_tiled_grid(
        image_paths=image_paths_for_grid,
        output_path=str(output_dir/"src_mv.png"),
        background_color=BACKGROUND_COLOR,
    )

    shutil.copy(renders_path / "06.png", output_dir / "src.png")

    print(f"Saved conditioning view and multi-view grid to {renders_path}.")
