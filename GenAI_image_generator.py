import torch
import gc
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel

# ------------------ MEMORY CLEANUP ------------------
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory cleared.")
    else:
        print("No GPU available.")


# ------------------ LOAD PIPELINE ------------------
def load_pipeline(model_path="./models/stable-diffusion-3.5-large-turbo"):
    """
    Loads Stable Diffusion 3.5 pipeline from a local folder
    with NF4 quantization and CPU offloading.
    """
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load transformer with 4-bit quantization
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    # Load T5 text encoder in 4-bit
    t5_nf4 = T5EncoderModel.from_pretrained(
        "models/t5-nf4",   # <-- also local version (downloaded & placed in models folder)
        # "diffusers/t5-nf4",
        torch_dtype=torch.bfloat16
    )

    # Final pipeline with offloading
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()
    return pipeline


# ------------------ TEXT TO IMAGE FUNCTION ------------------
def text_to_image(prompt, output_path="generated.png", 
                  negative_prompt=None, steps=28, guidance=7.0,
                  model_path="models/stable-diffusion-3.5-large-turbo"):
    """
    Generate a 2D image from text prompt using Stable Diffusion 3.5 (local model).
    """
    clean_memory()
    
    pipe = load_pipeline(model_path)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    image.save(output_path)
    print(f" Image saved at {output_path}")


# ------------------ EXAMPLE USAGE ------------------
if __name__ == "__main__":
    positive_prompt = (
        "photograph of a Dumbo Octopus (Grimpoteuthis), anatomically correct, "
        "8 webbed arms, 2 large ear-like fins on its mantle. Symmetrical, centered, "
        "neutral pose. Full body shot, front view. Shot on a plain white background, "
        "bright studio lighting, no shadows. 4k, high detail, sharp focus."
    )

    negative_prompt = (
        "blurry, deformed, mutated, disfigured, extra fins, extra arms, missing limbs, "
        "tentacles, cartoon, painting, artistic, dark, shadows, text, watermark, "
        "underwater scene, ocean background, noisy background."
    )

    text_to_image(
        positive_prompt, 
        "dumbo_octopus_reference.png", 
        negative_prompt,
        model_path="./models/stable-diffusion-3.5-large-turbo"  # local model path
    )
