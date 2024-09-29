import os
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

# Define the negative prompt
neg_prompt = 'deformity, blur faces, bad anatomy, cloned face, amputee, extra limbs, missing arms/legs, text, low quality'

# Check if CUDA (GPU) is available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Stable Diffusion XL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    use_safetensors=True if device == "cuda" else False,
    add_watermarker=False,
    force_download=True
).to(device)

# Initialize image-to-image pipeline
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)

def generate_paraphrases(image, description, save_folder="Para_Images"):
    """
    Generate paraphrased images based on input image and description.
    Saves paraphrased images to the specified folder.
    
    Args:
        image (PIL.Image): Input image for image-to-image generation.
        description (str): Text description for the paraphrasing.
        save_folder (str): Folder to save paraphrased images.
        
    Returns:
        list: Paths of saved paraphrased images.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_paths = []
    for i in range(5):
        gen_image = pipeline(
            description,
            image=image,
            strength=0.25,
            guidance_scale=7.5,
            negative_prompt=neg_prompt
        ).images[0]
        
        image_path = os.path.join(save_folder, f"paraphrased_image_{i+1}.png")
        gen_image.save(image_path)
        image_paths.append(image_path)
        print(f"Saved: {image_path}")
    
    return image_paths
