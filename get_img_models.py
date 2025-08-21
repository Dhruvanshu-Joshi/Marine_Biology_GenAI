from huggingface_hub import snapshot_download

from huggingface_hub import login

# To be prompted for your token in the terminal or a notebook widget:
login(token="") # enter your token


# Folder where models will be saved
save_dir = "models/stable-diffusion-3.5-large-turbo"

# Download *all* files from the repo into save_dir
snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # Make actual copies instead of symlinks
)

print(f"Model fully downloaded to: {save_dir}")
