import requests
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from fathomnet.api import images

# Assumes `model`, `sr_transform`, and `device` are defined globally
# You will need to ensure they are initialized before calling these functions.

# --- Scoring functions ---
def score_crop_quality(crop, weights=(0.2, 0.3, 0.5)):
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    b_score = min(max((brightness - 30) / (255 - 30), 0), 1)
    s_score = min(sharpness / 300.0, 1.0)
    c_score = min(contrast / 60.0, 1.0)

    return weights[0] * b_score + weights[1] * s_score + weights[2] * c_score


# --- Process and score images ---
def get_best_crop_image(concept, model, sr_transform, device):
    results = []
    for img in images.find_by_concept(concept):
        try:
            img_url = img.url
            # print(img_url)
            response = requests.get(img_url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if original_img is None:
                continue

            # Super-resolve image
            pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            input_tensor = sr_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                sr_tensor = model(input_tensor)[0].detach().cpu()

            sr_img = transforms.ToPILImage()(sr_tensor.clamp(-1, 1) * 0.5 + 0.5)
            sr_img = np.array(sr_img)
            sr_img = cv2.resize(sr_img, (original_img.shape[1], original_img.shape[0]))
            sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

            for box in img.boundingBoxes:
                if box.concept != concept:
                    continue
                x1, y1 = max(0, box.x), max(0, box.y)
                x2 = min(original_img.shape[1], box.x + box.width)
                y2 = min(original_img.shape[0], box.y + box.height)

                crop_orig = original_img[y1:y2, x1:x2]
                crop_sr = sr_img[y1:y2, x1:x2]

                score_orig = score_crop_quality(crop_orig)
                score_sr = score_crop_quality(crop_sr)

                best_crop = crop_sr if score_sr > score_orig else crop_orig
                best_crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(best_crop_rgb)
                results.append((max(score_sr, score_orig), pil_crop))
        except Exception:
            continue
    # print(results)
    return sorted(results, key=lambda x: x[0], reverse=True)[0][1] if results else None


def get_all_cropped_images(concept):
    results = []
    for img in images.find_by_concept(concept):
        try:
            img_url = img.url
            response = requests.get(img_url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if original_img is None:
                continue

            # Super-resolve image
            pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            input_tensor = sr_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                sr_tensor = model(input_tensor)[0].detach().cpu()

            sr_img = transforms.ToPILImage()(sr_tensor.clamp(-1, 1) * 0.5 + 0.5)
            sr_img = np.array(sr_img)
            sr_img = cv2.resize(sr_img, (original_img.shape[1], original_img.shape[0]))
            sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

            for box in img.boundingBoxes:
                if box.concept != concept:
                    continue
                x1, y1 = max(0, box.x), max(0, box.y)
                x2 = min(original_img.shape[1], box.x + box.width)
                y2 = min(original_img.shape[0], box.y + box.height)

                crop_orig = original_img[y1:y2, x1:x2]
                crop_sr = sr_img[y1:y2, x1:x2]

                score_orig = score_crop_quality(crop_orig)
                score_sr = score_crop_quality(crop_sr)

                best_crop = crop_sr if score_sr > score_orig else crop_orig
                best_crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(best_crop_rgb)
                results.append((max(score_sr, score_orig), pil_crop))
        except Exception:
            continue

    return sorted(results, key=lambda x: x[0], reverse=True)
