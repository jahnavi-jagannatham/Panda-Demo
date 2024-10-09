import cv2
import numpy as np
import matplotlib.pyplot as plt

class BoundingBoxFinder:
    """
    A class for detecting common regions in multiple images and finding bounding boxes
    for the common regions.
    """

    def __init__(self, image_list):
        """
        Initializes the class with the list of image paths.

        Parameters:
        -----------
        image_list : list
            A list of image file paths.
        """
        self.image_list = image_list
        self.images = self.load_images()

    def load_images(self):
        """Loads images from the file paths and returns a list of image arrays."""
        images = []
        for img_path in self.image_list:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Error loading image: {img_path}")
        return images

    def find_common_regions(self):
        """
        Finds the common regions between the loaded images and returns a list of bounding box coordinates.

        Returns:
        --------
        list of tuples:
            A list of tuples, each containing (x, y, w, h) as the bounding box coordinates.
        """
        if len(self.images) < 2:
            raise ValueError("At least two images are required for finding common regions.")

        # Convert images to grayscale for comparison
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]

        # Initialize common_region with the first image
        common_region = gray_images[0]

        # Find the common region by performing bitwise AND with subsequent images
        for gray_img in gray_images[1:]:
            common_region = cv2.bitwise_and(common_region, gray_img)

        # Find contours in the common region to identify all bounding boxes
        contours, _ = cv2.findContours(common_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes

    def non_max_suppression(self, boxes, threshold=0.3):
        """
        Applies Non-Maximum Suppression to eliminate overlapping bounding boxes.

        Parameters:
        -----------
        boxes : list of tuples
            A list of bounding boxes (x, y, w, h).
        threshold : float
            The threshold for IoU to consider boxes as overlapping.

        Returns:
        --------
        list of tuples:
            The remaining bounding boxes after applying NMS.
        """
        if len(boxes) == 0:
            return []

        # Convert bounding boxes to a numpy array
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]

        # Compute the area of the bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            # Compute the intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Compute the width and height of the intersection boxes
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # Compute the ratio of overlap (IoU)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than the threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return boxes_array[keep].tolist()

    def get_top_bounding_boxes(self, bounding_boxes, max_boxes=5):
        """
        Returns the top bounding boxes based on their area after applying NMS.

        Parameters:
        -----------
        bounding_boxes : list
            A list of bounding box tuples (x, y, w, h).
        max_boxes : int
            The maximum number of bounding boxes to return.

        Returns:
        --------
        list of tuples:
            A list of the top bounding boxes sorted by area.
        """
        # Apply Non-Maximum Suppression to remove overlapping boxes
        non_overlapping_boxes = self.non_max_suppression(bounding_boxes)

        # Sort bounding boxes by area (width * height)
        non_overlapping_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        # Return the top N bounding boxes
        return non_overlapping_boxes[:max_boxes]

    def get_bounding_box_on_image(self, image, bbox):
        """
        Draws a bounding box on the provided image.

        Parameters:
        -----------
        image : numpy.ndarray
            The image on which to draw the bounding box.
        bbox : tuple
            A tuple containing the bounding box coordinates (x, y, w, h).

        Returns:
        --------
        numpy.ndarray:
            The image with the bounding box drawn on it.
        """
        x, y, w, h = bbox
        img_with_box = image.copy()

        # Draw the bounding box on the image
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_with_box

    def display_image(self, image):
        """
        Displays the provided image using matplotlib.

        Parameters:
        -----------
        image : numpy.ndarray
            The image to display.
        """
        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide axes
        plt.show()


# Example usage:
if __name__ == "__main__":
    # List of image file paths
    image_list = ['/content/4428_gen_0.png', '/content/4428_gen_1.png',
                  '/content/4428_gen_2.png', '/content/4428_gen_3.png', 
                  '/content/4428_gen_4.png']

    # Create an instance of the BoundingBoxFinder
    bbox_finder = BoundingBoxFinder(image_list)

    # Find the common bounding boxes
    common_bboxes = bbox_finder.find_common_regions()

    # Limit to the top bounding boxes based on area
    max_boxes_to_display = 5  # Set the desired maximum number of boxes
    top_bboxes = bbox_finder.get_top_bounding_boxes(common_bboxes, max_boxes_to_display)

    if top_bboxes:
        print(f"Top bounding boxes: {top_bboxes}")

        # Get the first image
        first_image = bbox_finder.images[0]

        # Draw the bounding boxes on the first image
        img_with_bboxes = first_image.copy()
        for bbox in top_bboxes:
            img_with_bboxes = bbox_finder.get_bounding_box_on_image(img_with_bboxes, bbox)

        # Display the image with the bounding boxes
        bbox_finder.display_image(img_with_bboxes)
    else:
        print("No common region found between the images.")
