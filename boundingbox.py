"""Official Bounding Box file"""

import numpy as np
import matplotlib.pyplot as P
import matplotlib.patches as patches

class Boundingbox:
    """
    A class for analyzing heatmaps by dividing them into grids, identifying the highest points within these grids,
    and merging overlapping regions. The class also provides methods to plot the heatmap with the identified grids.

    Attributes:
    -----------
    heatmap : numpy.ndarray
        The input heatmap to be analyzed.
    grid_fraction : float
        The fraction of the heatmap's size used to determine the grid size.
    grid_size : int
        The size of each grid based on the heatmap's dimensions and grid fraction.
    half_grid_size : int
        Half the size of the grid, used for calculating grid boundaries.
    grid_centers : list
        A list to store the centers of identified grids with the highest heatmap values.
    merged_boxes : list
        A list to store merged bounding boxes that encompass overlapping grid regions.

    Methods:
    --------
    find_highest_points_with_grids(num_grids=10):
        Identifies and returns the centers of grids with the highest values in the heatmap.
    
    merge_overlapping_boxes():
        Merges overlapping grid boxes and returns the merged boxes as a list of dictionaries.

    non_max_suppression(threshold=0.6):
        Applies Non-Maximal Suppression (NMS) to merge overlapping boxes based on a specified IoU threshold.

    plot_heatmap_with_grids():
        Plots the heatmap and overlays rectangles representing the merged grid boxes.
    """

    def __init__(self, heatmap, grid_fraction=1/8):
        """
        Initializes the HeatmapGridAnalyzer class by setting up the heatmap and grid parameters.

        Parameters:
        -----------
        heatmap : numpy.ndarray
            The input heatmap to be analyzed.
        grid_fraction : float, optional
            The fraction of the heatmap's size used to determine the grid size (default is 1/8).
        """
        self.heatmap = heatmap
        self.grid_fraction = grid_fraction
        self.grid_size = int(heatmap.shape[0] * grid_fraction)
        self.half_grid_size = self.grid_size // 2
        self.grid_centers = []
        self.merged_boxes = []

    def find_highest_points_with_grids(self, num_grids):
        """
        Finds the highest points in the heatmap using grids and marks the regions as -inf.
        
        Parameters:
        -----------
        num_grids : int
            The number of grids to divide the heatmap into.
        
        Returns:
        --------
        grid_centers : list
            A list of grid centers with the highest values.
        """
        processed_heatmap = np.array(self.heatmap, dtype=np.float32)  # Ensure heatmap is float32
        grid_centers = []
    
        for i in range(num_grids):
            # Find the highest point in the current heatmap
            max_idx = np.unravel_index(np.argmax(processed_heatmap), processed_heatmap.shape)
            grid_centers.append({'x': max_idx[1], 'y': max_idx[0]})
    
            # Mask the area around the found highest point with -inf to avoid selecting it again
            x, y = max_idx
            x_start = max(0, x - self.half_grid_size)
            x_end = min(processed_heatmap.shape[0], x + self.half_grid_size + 1)
            y_start = max(0, y - self.half_grid_size)
            y_end = min(processed_heatmap.shape[1], y + self.half_grid_size + 1)
            processed_heatmap[x_start:x_end, y_start:y_end] = -np.inf  # Now safe to assign -inf
    
        # Convert grid centers to a list of dictionaries with 'x' and 'y' keys
        return grid_centers


    def merge_overlapping_boxes(self):
        """
        Merges overlapping grid boxes based on their bounding coordinates. Each grid box is defined by
        its center and half-grid size. Overlapping boxes are merged into a single bounding box.

        Returns:
        --------
        list of dict
            A list of dictionaries representing the merged bounding boxes with keys 'x1', 'y1', 'x2', and 'y2'.
        """
        self.merged_boxes = []
        for center in self.grid_centers:
            x, y = center
            # Define the bounding box for the current grid
            new_box = [y - self.half_grid_size, x - self.half_grid_size, y + self.half_grid_size, x + self.half_grid_size]
            merged = False

            # Check for overlaps with existing boxes and merge if necessary
            for box in self.merged_boxes:
                if not (new_box[2] < box[0] or new_box[0] > box[2] or new_box[3] < box[1] or new_box[1] > box[3]):
                    box[0] = min(box[0], new_box[0])
                    box[1] = min(box[1], new_box[1])
                    box[2] = max(box[2], new_box[2])
                    box[3] = max(box[3], new_box[3])
                    merged = True
                    break

            # If no overlap, add the new box to the list
            if not merged:
                self.merged_boxes.append(new_box)

        # Convert merged boxes to a list of dictionaries with 'x1', 'y1', 'x2', and 'y2' keys
        merged_boxes_json = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in self.merged_boxes]
        return merged_boxes_json

    def non_max_suppression(self, threshold=0.25):
        """
        Applies Non-Maximal Suppression (NMS) to merge overlapping boxes based on a specified Intersection over Union (IoU) threshold.

        Parameters:
        -----------
        threshold : float, optional
            The IoU threshold for merging boxes (default is 0.25).

        Returns:
        --------
        list of dict
            A list of dictionaries representing the merged bounding boxes after applying NMS with keys 'x1', 'y1', 'x2', and 'y2'.
        """
        if not self.grid_centers:
            return []

        # First, find and merge overlapping boxes
        self.merge_overlapping_boxes()

        boxes = np.array(self.merged_boxes)

        # Coordinates of the boxes
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]

        # Compute the area of the boxes and sort by the bottom-right y-coordinate of the box
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute the coordinates of the intersection boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Compute the width and height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap (IoU)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than the threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        self.merged_boxes = boxes[keep].tolist()

        # Convert merged boxes to a list of dictionaries with 'x1', 'y1', 'x2', and 'y2' keys
        merged_boxes_json = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in self.merged_boxes]
        return merged_boxes_json

    def plot_heatmap_with_grids(self):
        """
        Plots the heatmap and overlays rectangles representing the merged grid boxes. Each box is
        drawn with a blue border and no fill, highlighting the areas of interest within the heatmap.

        The plot is displayed using matplotlib.

        Returns:
        --------
        None
        """
        fig, ax = P.subplots()
        cax = ax.imshow(self.heatmap, cmap='hot', interpolation='nearest')
        fig.colorbar(cax)

        # Draw rectangles for each merged box
        for box in self.merged_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # P.show()
