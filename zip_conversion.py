import os
import zipfile

# Root folder (adjust if needed)
root_dir = os.path.dirname(os.path.abspath(__file__))
zip_filename = os.path.join(root_dir, "project_no_models.zip")

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for foldername, subfolders, filenames in os.walk(root_dir):
        # Skip the 'models' folder
        if "models" in foldername:
            continue

        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            # Ensure zip paths are relative to root
            arcname = os.path.relpath(file_path, root_dir)
            zipf.write(file_path, arcname)

print(f"[INFO] Zipped folder created at: {zip_filename}")
