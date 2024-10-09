import os
import numpy as np
import PIL.Image
import torch
from torchvision import models, transforms
import saliency.core as saliency

class XRAIHeatmapGenerator:
    def __init__(self):
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the pretrained InceptionV3 model
        self.model = models.inception_v3(pretrained=True, init_weights=False).to(self.device)
        self.model.eval()

        # Register hooks for Grad-CAM
        self.conv_layer = self.model.Mixed_7c
        self.conv_layer_outputs = {}

        self.conv_layer.register_forward_hook(self.conv_layer_forward)
        self.conv_layer.register_full_backward_hook(self.conv_layer_backward)

        # Transformer for input preprocessing
        self.transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # String for the class index key used in XRAI
        self.class_idx_str = 'class_idx_str'

    def conv_layer_forward(self, m, i, o):
        self.conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

    def conv_layer_backward(self, m, i, o):
        self.conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()

    # Function to load and preprocess the image
    def load_image(self, file_path):
        im = PIL.Image.open(file_path)
        im = im.resize((299, 299))  # Resizing to 299x299 for InceptionV3 input
        im = np.asarray(im)
        return im

    # Preprocess a list of images for input to the model
    def preprocess_images(self, images):
        images = np.array(images) / 255  # Normalize image data
        images = np.transpose(images, (0, 3, 1, 2))  # Move channel dimension to (batch, channels, height, width)
        images = torch.tensor(images, dtype=torch.float32, device=self.device)
        images = self.transformer.forward(images)
        return images.requires_grad_(True)

    # Call model function for XRAI
    def call_model_function(self, images, call_model_args=None, expected_keys=None):
        images = self.preprocess_images(images)  # Preprocess the input images
        target_class_idx = call_model_args[self.class_idx_str]
        output = self.model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)

        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            outputs = output[:, target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.movedim(grads[0], 1, 3)  # Move channel dimension
            gradients = grads.detach().cpu().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class_idx] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return self.conv_layer_outputs

    # Generate XRAI heatmap for a single image
    def generate_xrai_heatmap(self, image):
        call_model_args = {self.class_idx_str: np.argmax(self.model(self.preprocess_images([image]))[0].detach().cpu().numpy())}
        xrai_object = saliency.XRAI()
        xrai_attributions = xrai_object.GetMask(image, self.call_model_function, call_model_args, batch_size=20)
        return xrai_attributions

    # Function to process a list of image files and return heatmaps
    def process_images(self, image_files):
        heatmaps = []

        # Iterate through each image in the provided list
        for image_file in image_files:
            im_orig = self.load_image(image_file)  # Load the image
            im = im_orig.astype(np.float32)

            # Generate XRAI heatmap
            xrai_heatmap = self.generate_xrai_heatmap(im)
            heatmaps.append(xrai_heatmap)

        return heatmaps


# Example usage of the XRAIHeatmapGenerator class
if __name__ == "__main__":
    # Example: providing a list of image file paths directly
    image_files = ['/content/Image_620-4.png']  # Add image paths here

    # Initialize the XRAIHeatmapGenerator class
    xrai_generator = XRAIHeatmapGenerator()

    # Process the images and get the XRAI heatmaps
    heatmaps = xrai_generator.process_images(image_files)

    # Loop through each heatmap and display it using PIL
    for i, heatmap in enumerate(heatmaps):
        print(f"Displaying heatmap for image {i + 1}...")

        # Normalize the heatmap for display (optional)
        heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        # Display the heatmap using Matplotlib
        plt.imshow(heatmap_normalized, cmap='inferno')  # Use a colormap like 'inferno'
        plt.title(f'Heatmap {i + 1}')
        plt.axis('off')  # Hide axes
        plt.show()
