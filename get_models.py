# from huggingface_hub import snapshot_download

# # Replace with the model repo ID you want
# repo_id = "tencent/Hunyuan3D-2"

# # Destination folder to save the model
# local_dir = "models"

# # Download the model
# snapshot_download(
#     repo_id="tencent/Hunyuan3D-2",
#     local_dir="models",
#     allow_patterns=[
#         "hunyuan3d-dit-v2-0/model.fp16.safetensors",
#         "hunyuan3d-dit-v2-0/config.yaml"
#     ],
# )

# snapshot_download(
#     repo_id="tencent/Hunyuan3D-2",
#     local_dir="models",
#     allow_patterns=["hunyuan3d-paint-v2-0-turbo/**"],  # <== download everything inside the subfolder
# )

# snapshot_download(
#     repo_id="tencent/Hunyuan3D-2",
#     local_dir="models",
#     allow_patterns=["hunyuan3d-delight-v2-0/**"],  # <== download everything inside the subfolder
# )

# print("Model downloaded to:", local_dir)
from huggingface_hub import snapshot_download

# Repo ID for the NF4-quantized T5 encoder
repo_id = "diffusers/t5-nf4"

# Destination folder
local_dir = "models/t5-nf4"

# Download the full repo into models/t5-nf4
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # ensures actual files instead of symlinks
)

print("T5 NF4 model downloaded to:", local_dir)
