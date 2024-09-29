# Install necessary packages from requirements.txt
!pip install -r requirements.txt

import gradio as gr
from PIL import Image
import os
from paraphrase_generator import generate_paraphrases  # Import the paraphrase generation function

# Gradio function to handle input and display output
def paraphrase_images(image, description):
    # Generate paraphrased images and save them to 'Para_Images' folder
    save_folder = "Para_Images"
    result_message = generate_paraphrases(image, description, save_folder)
    
    # Collect the generated images to display in Gradio
    paraphrased_images = []
    for i in range(1, 6):
        img_path = os.path.join(save_folder, f"paraphrased_image_{i}.png")
        paraphrased_images.append(Image.open(img_path))
    
    return paraphrased_images

# Define Gradio inputs and outputs
image_input = gr.Image(type="pil")
text_input = gr.Textbox(label="Description", placeholder="Enter a description for the image", lines=2)

# Output: Display 5 images
outputs = [gr.Image(label=f"Paraphrased Image {i+1}") for i in range(5)]

# Launch Gradio interface
gr.Interface(
    fn=paraphrase_images,
    inputs=[image_input, text_input],
    outputs=outputs,
    title="Paraphrase Image Generator",
    description="Upload an image and provide a description to generate 5 paraphrased images."
).launch()
