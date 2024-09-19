import os
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

# Define the negative prompt
neg_prompt = 'deformity, blur faces, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'

# Check if CUDA (GPU) is available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the SDXL model with appropriate settings for CPU and GPU
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

if device == "cuda":
    # Use float16 for GPU to leverage faster mixed-precision calculations
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False,
        force_download=True  # Force redownload in case of cache issues
    ).to(device)
else:
    # Use float32 for CPU as float16 is not efficient on CPU
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float32, use_safetensors=False, add_watermarker=False,
        force_download=True  # Force redownload in case of cache issues
    ).to(device)

# Initialize image-to-image pipeline using the loaded text-to-image model
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)

def generate_paraphrases(image, description, save_folder="Para_Images"):
    """
    Generate 5 paraphrased images using the given image and description.
    The generated images are saved in the specified folder.
    
    Args:
        image (PIL.Image): The base image to be used for image-to-image generation.
        description (str): The text description for the paraphrasing process.
        save_folder (str): The folder where generated images will be saved.
        
    Returns:
        str: A message indicating where the images were saved.
    """
    
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate 5 paraphrased images with the negative prompt
    for i in range(5):
        gen_image = pipeline(
            description,
            image=image,
            strength=0.25,  # Controls how much the input image is transformed
            guidance_scale=7.5,  # Controls adherence to the prompt
            negative_prompt=neg_prompt
        ).images[0]
        
        # Save each image to the specified folder
        image_path = os.path.join(save_folder, f"paraphrased_image_{i+1}.png")
        gen_image.save(image_path)
        print(f"Saved: {image_path}")
    
    return f"Paraphrased images saved in folder '{save_folder}'"
