# # generate_3d.py
# import os
# import sys
# import glob
# import gc
# import time
# import torch
# import numpy as np
# import trimesh
# import open3d as o3d
# from PIL import Image
# from torchvision import transforms

# # Custom imports
# from Image_generator import mymodel
# from fathomnet.api import images
# from utils import get_best_crop_image
# from GenAI_image_generator import text_to_image

# # HY3D imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from hy3dgen.texgen import Hunyuan3DPaintPipeline
# from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# from hy3dgen.rembg import BackgroundRemover


# # ------------------ MEMORY CLEANUP ------------------
# def clean_memory():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# # ------------------ DEVICE SELECTION ------------------
# clean_memory()
# DEVICE_2 = torch.device(
#     "cuda:1" if torch.cuda.device_count() > 1
#     else ("cuda:0" if torch.cuda.is_available() else "cpu")
# )

# # ------------------ LOAD PIPELINES (ONCE) ------------------
# print("Loading models...")

# mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
#     "models", device=DEVICE_2
# )

# paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
#     "models", subfolder="hunyuan3d-paint-v2-0-turbo"
# )

# print("Models loaded successfully.")


# # ------------------ MAIN FUNCTION ------------------
# def generate_3d(concept: str, method: str = "genai", output_dir="outputs"):
#     """
#     Generate a 3D asset from a text prompt or fetched image.

#     Args:
#         concept (str): Name of the object/creature (e.g., "Grimpoteuthis").
#         method (str): "genai" for text-to-image generation OR "fathomnet" for fetching real images.
#         output_dir (str): Directory to save results.

#     Returns:
#         dict: Paths to exported meshes {"raw_mesh": ..., "painted_mesh": ...}
#     """

#     os.makedirs(output_dir, exist_ok=True)

#     # ------------------ IMAGE GENERATION ------------------
#     if method == "genai":
#         positive_prompt = (
#             f"photograph of a {concept}, anatomically correct, centered, full body, "
#             "plain white background, bright studio lighting, 4k, high detail, sharp focus."
#         )

#         negative_prompt = (
#             "blurry, deformed, mutated, disfigured, extra limbs, cartoon, painting, artistic, "
#             "dark, shadows, text, watermark, underwater scene, noisy background."
#         )

#         best_image = text_to_image(
#             prompt=positive_prompt,
#             negative_prompt=negative_prompt
#         )  # returns PIL.Image directly

#     elif method == "fathomnet":
#         fathomnet_image_list = images.find_by_concept(concept)
#         if not fathomnet_image_list:
#             raise ValueError(f"No images found on FathomNet for concept {concept}")

#         # Load SRGAN model
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = mymodel()
#         model.load_state_dict(torch.load("image_models/SR_GAN_best.pth", map_location=device))
#         model.to(device)
#         model.eval()

#         sr_transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#         best_image = get_best_crop_image(concept, model, sr_transform, device)
#         if best_image is None:
#             raise ValueError("No suitable image found in FathomNet pipeline.")

#     else:
#         raise ValueError("Invalid method. Choose 'genai' or 'fathomnet'.")

#     # ------------------ BACKGROUND REMOVAL ------------------
#     image = best_image.convert("RGBA")
#     rembg = BackgroundRemover()
#     image = rembg(image)

#     # ------------------ MESH GENERATION ------------------
#     img_mesh = mesh_pipeline(image=image)[0]

#     if isinstance(img_mesh, trimesh.Scene):
#         meshes = [g for g in img_mesh.geometry.values()]
#         mesh = trimesh.util.concatenate(meshes)
#     else:
#         mesh = img_mesh

#     # ------------------ MESH SIMPLIFICATION ------------------
#     o3d_mesh = o3d.geometry.TriangleMesh()
#     o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
#     o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

#     target_faces = int(len(mesh.faces) * 0.1)  # 90% reduction
#     simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

#     decimated_mesh = trimesh.Trimesh(
#         vertices=np.asarray(simplified.vertices),
#         faces=np.asarray(simplified.triangles),
#         process=False
#     )

#     # ------------------ PAINTING ------------------
#     painted_mesh = paint_pipeline(decimated_mesh, image=image)

#     # ------------------ EXPORT ------------------
#     raw_mesh_path = os.path.join(output_dir, f"mesh_{concept}.glb")
#     painted_mesh_path = os.path.join(output_dir, f"painted_{concept}.glb")

#     mesh.export(raw_mesh_path)
#     painted_mesh.export(painted_mesh_path)

#     clean_memory()

#     return {
#         "raw_mesh": raw_mesh_path,
#         "painted_mesh": painted_mesh_path
#     }


# if __name__ == "__main__":
#     # Example local run
#     results = generate_3d("Grimpoteuthis", method="genai")
#     print("Exported:", results)

# generate_3d.py
import os
import sys
import gc
import torch
import numpy as np
import trimesh
import open3d as o3d
from torchvision import transforms

# Custom imports
from Image_generator import mymodel
from fathomnet.api import images
from utils import get_best_crop_image
from GenAI_image_generator import text_to_image
from hy3dgen.rembg import BackgroundRemover

# ------------------ MEMORY CLEANUP ------------------
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ------------------ MAIN FUNCTION ------------------
def generate_3d(
    concept: str,
    method: str = "genai",
    output_dir="outputs",
    mesh_pipeline=None,
    paint_pipeline=None,
):
    """
    Generate a 3D asset from a text prompt or fetched image.

    Args:
        concept (str): Name of the object/creature (e.g., "Grimpoteuthis").
        method (str): "genai" for text-to-image generation OR "fathomnet" for fetching real images.
        output_dir (str): Directory to save results.
        mesh_pipeline: Preloaded Hunyuan3DDiTFlowMatchingPipeline
        paint_pipeline: Preloaded Hunyuan3DPaintPipeline

    Returns:
        dict: Paths to exported meshes {"raw_mesh": ..., "painted_mesh": ...}
    """
    if mesh_pipeline is None or paint_pipeline is None:
        raise ValueError("mesh_pipeline and paint_pipeline must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------ IMAGE GENERATION ------------------
    if method == "genai":
        positive_prompt = (
            f"photograph of a {concept}, anatomically correct, centered, full body, "
            "plain white background, bright studio lighting, 4k, high detail, sharp focus."
        )
        negative_prompt = (
            "blurry, deformed, mutated, disfigured, extra limbs, cartoon, painting, artistic, "
            "dark, shadows, text, watermark, underwater scene, noisy background."
        )
        best_image = text_to_image(
            prompt=positive_prompt,
            negative_prompt=negative_prompt
        )  # returns PIL.Image directly

    elif method == "fathomnet":
        fathomnet_image_list = images.find_by_concept(concept)
        if not fathomnet_image_list:
            raise ValueError(f"No images found on FathomNet for concept {concept}")

        # Load SRGAN model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = mymodel()
        model.load_state_dict(torch.load("image_models/SR_GAN_best.pth", map_location=device))
        model.to(device)
        model.eval()

        sr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        best_image = get_best_crop_image(concept, model, sr_transform, device)
        if best_image is None:
            raise ValueError("No suitable image found in FathomNet pipeline.")

    else:
        raise ValueError("Invalid method. Choose 'genai' or 'fathomnet'.")

    # ------------------ BACKGROUND REMOVAL ------------------
    image = best_image.convert("RGBA")
    rembg = BackgroundRemover()
    image = rembg(image)

    # ------------------ MESH GENERATION ------------------
    img_mesh = mesh_pipeline(image=image)[0]

    if isinstance(img_mesh, trimesh.Scene):
        meshes = [g for g in img_mesh.geometry.values()]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = img_mesh

    # ------------------ MESH SIMPLIFICATION ------------------
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    target_faces = int(len(mesh.faces) * 0.1)  # 90% reduction
    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

    decimated_mesh = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False
    )

    # ------------------ PAINTING ------------------
    painted_mesh = paint_pipeline(decimated_mesh, image=image)

    # ------------------ EXPORT ------------------
    raw_mesh_path = os.path.join(output_dir, f"mesh_{concept}.glb")
    painted_mesh_path = os.path.join(output_dir, f"painted_{concept}.glb")

    mesh.export(raw_mesh_path)
    painted_mesh.export(painted_mesh_path)

    clean_memory()

    return {
        "raw_mesh": raw_mesh_path,
        "painted_mesh": painted_mesh_path
    }
