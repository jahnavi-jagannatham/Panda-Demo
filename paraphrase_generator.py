import os
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

# Define the negative prompt
neg_prompt = 'deformity, blur faces, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'

# Load the SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
).to("cuda")

pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

def generate_paraphrases(image, description, save_folder="Para_Images"):
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate 5 paraphrased images with the negative prompt
    for i in range(5):
        gen_image = pipeline(
            description,
            image=image,
            strength=0.25,
            guidance_scale=7.5,
            negative_prompt=neg_prompt
        ).images[0]
        
        # Save each image to the specified folder
        image_path = os.path.join(save_folder, f"paraphrased_image_{i+1}.png")
        gen_image.save(image_path)
        print(f"Saved: {image_path}")
    
    return f"Paraphrased images saved in folder '{save_folder}'"
