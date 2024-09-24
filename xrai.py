import os
import saliency.core as saliency
import numpy as np
def XRAI(self, Para_Images):
        """
        Processes each paraphrased image in the Para_Images directory,
        generates and saves XRAI saliency maps, and analyzes the heatmap.

        Parameters:
        -----------
        Para_Images : str
            The path to the directory containing folders with paraphrased images.
        """
        # for folder in os.listdir(Para_Images):
        #     print(f' --- folder : {Para_Images}/{folder} ---- ')
        #     folder_path = os.path.join(Para_Images, folder)
        #     if os.path.isdir(folder_path):
        #         print('ok ok ok ok ok')
        #         # Assumes only one image per folder
        for img_filename in os.listdir(Para_Images):
            # Check if the filename starts with 'Original' or 'gen_'
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "Heatmap" in img_filename or "Top" in img_filename : 
                    print(f'Leaving out {img_filename}')
                    continue 

                img_path = os.path.join(Para_Images, img_filename)
                print(f"Processing image: {img_path}")

                img_result_dir = os.path.join(Para_Images, os.path.splitext(img_filename)[0])
                os.makedirs(img_result_dir, exist_ok=True)

                im_orig = self.load_image(img_path)
                im_tensor = self.preprocess_images([im_orig])
                predictions = self.model(im_tensor)
                predictions = predictions.detach().numpy()
                prediction_class = np.argmax(predictions[0])
                call_model_args = {'class_idx_str': prediction_class}

                print("Prediction class: " + str(prediction_class))

                xrai_object = saliency.XRAI()
                xrai_params = saliency.XRAIParameters()
                xrai_params.algorithm = 'fast'
                xrai_attributions_fast = xrai_object.GetMask(
                    im_orig, self.call_model_function, call_model_args, 
                    extra_parameters=xrai_params, batch_size=20
                )

                # Save original image
                self.save_image(im_orig, img_result_dir, title='Original Image')
                
                # Save heatmap of full saliency
                self.save_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI Heatmap')

                # Generate masks for the top 15% and bottom 15% salient regions
                top_15_mask = xrai_attributions_fast >= np.percentile(xrai_attributions_fast, 85)
                bottom_15_mask = xrai_attributions_fast <= np.percentile(xrai_attributions_fast, 15)

                # Create a combined image with top 15% in red and bottom 15% in blue
                im_combined = np.array(im_orig).copy()

                # Highlight top 15% in red (R channel)
                im_combined[top_15_mask] = [255, 0, 0]

                # Highlight bottom 15% in blue (B channel), ensuring top 15% isn't overwritten
                im_combined[bottom_15_mask & ~top_15_mask] = [0, 0, 255]

                # Save the combined image with both top 15% and bottom 15% highlighted
                self.save_image(im_combined, img_result_dir, title='Top15_Bottom15_Combined')

                # Analyze and save the heatmap (optional step)
                self.analyze_heatmap(xrai_attributions_fast, img_result_dir, title='XRAI_Analysis')
            else:
                print(f"Skipping file: {img_filename}'")
        
        self.process_and_score_boxes(base_path=Para_Images)
