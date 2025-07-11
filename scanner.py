import os
import math
import argparse
import numpy as np
import cv2
from PIL import Image
import glob
from datetime import datetime
import random
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Human detection will be disabled.")

# New imports - for advanced edge smoothing and distance calculation
try:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Advanced edge smoothing and distance-based sampling will use fallback methods.")


def parse_image_size(size_str):
    """Parse image size string"""
    if not size_str or size_str.lower() == 'auto':
        return None
    
    # Remove brackets and spaces
    size_str = size_str.strip('[]').replace(' ', '')
    
    # Predefined sizes
    predefined_sizes = {'A4': (2480, 3508),      # A4 300 DPI
                        'A4_150': (1240, 1754),  # A4 150 DPI  
                        'A4_200': (1654, 2339),  # A4 200 DPI
                        'A3': (3508, 4961),      # A3 300 DPI
                        'LETTER': (2550, 3300),  # Letter 300 DPI
                        'LEGAL': (2550, 4200),}  # Legal 300 DPI
    
    # Check if it's a predefined size
    if size_str.upper() in predefined_sizes:
        return predefined_sizes[size_str.upper()]
    
    # Parse custom size
    try:
        if ',' in size_str:
            width, height = map(int, size_str.split(','))
        elif 'x' in size_str.lower():
            width, height = map(int, size_str.lower().split('x'))
        else:
            raise ValueError("Invalid size format")
        
        if width <= 0 or height <= 0:
            raise ValueError("Size must be positive")
        
        return (width, height)
    except ValueError as e:
        raise ValueError(f"Cannot parse size '{size_str}': {e}")


class HumanDetector:
    """Human detection and removal processor - using random sampling fill method"""
    
    def __init__(self):
        self.model = None
        if YOLO_AVAILABLE:
            try:
                # Try to load YOLO11 model
                self.model = YOLO('yolo11n.pt')  # Use nano version for faster speed
                print("YOLO11 model loaded successfully")
            except Exception as e:
                print(f"YOLO11 model loading failed: {e}")
                self.model = None
        else:
            print("YOLO library not installed, human detection feature unavailable")
    
    def detect_humans(self, image):
        """Detect human parts in image"""
        if self.model is None:
            return None, None
        
        try:
            # Use YOLO for detection
            results = self.model(image, verbose=False)
            
            # Create human mask
            human_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            detected_humans = False
            
            for result in results:
                if result.masks is not None:
                    for i, class_id in enumerate(result.boxes.cls):
                        # In COCO dataset, person class ID is 0
                        if int(class_id) == 0:  # person class
                            mask = result.masks.data[i].cpu().numpy()
                            # Resize mask to original image size
                            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            # Convert to binary mask
                            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                            human_mask = cv2.bitwise_or(human_mask, mask_binary)
                            detected_humans = True
            
            if not detected_humans:
                # If no segmentation results, try using bounding box detection
                for result in results:
                    if result.boxes is not None:
                        for i, class_id in enumerate(result.boxes.cls):
                            if int(class_id) == 0:  # person class
                                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = box
                                # Create mask in detection box area
                                human_mask[y1:y2, x1:x2] = 255
                                detected_humans = True
            
            return human_mask if detected_humans else None, detected_humans
            
        except Exception as e:
            print(f"Error during human detection: {e}")
            return None, False
    
    def create_smooth_mask(self, mask, method='gaussian', smooth_radius=15):
        """Create smooth mask for edge feathering"""
        if mask is None:
            return None
        
        # Ensure mask is in float format
        smooth_mask = mask.astype(np.float32) / 255.0
        
        if method == 'gaussian' and SCIPY_AVAILABLE:
            # Gaussian blur feathering - prioritize scipy
            smooth_mask = gaussian_filter(smooth_mask, sigma=smooth_radius/3)
        elif method == 'gaussian':
            # Fallback: use OpenCV Gaussian blur
            kernel_size = max(3, int(smooth_radius * 2) | 1)  # Ensure odd number
            smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), smooth_radius/3)
            
        elif method == 'distance':
            # Distance transform based feathering
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                normalized_dist = dist_transform / max_dist
                # Use sigmoid function to create smooth edges
                smooth_mask = 1 / (1 + np.exp(-10 * (normalized_dist - 0.3)))
                # Ensure original mask area is still 1
                smooth_mask[mask > 0] = 1.0
            
        elif method == 'morphological':
            # Morphological smoothing
            kernel_size = max(3, smooth_radius // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Dilate then erode, then Gaussian blur
            dilated = cv2.dilate(mask, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            kernel_size = max(3, int(smooth_radius / 2) | 1)
            smooth_mask = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, 
                                          (kernel_size, kernel_size), smooth_radius/4)
        
        # Ensure values are in [0,1] range
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
        
        return smooth_mask
    
    def random_sampling_fill(self, image, human_mask):
        """
        Use random sampling method to fill human regions
        Fill range: mask expanded outward by 10 pixels
        Sampling range: area within 20 pixels outside the new region (unified sampling pool)
        """
        if human_mask is None:
            return image
        
        print("Using random sampling method for human removal...")
        result_image = image.copy()
        
        # 1. Create expanded fill area (original mask expanded outward by 10 pixels)
        kernel_expand = np.ones((21, 21), np.uint8)  # 10*2+1 = 21
        expanded_mask = cv2.dilate(human_mask, kernel_expand, iterations=1)
        
        # 2. Create sampling area (20 pixels outside expanded area)
        kernel_sample = np.ones((41, 41), np.uint8)  # 20*2+1 = 41  
        sampling_outer = cv2.dilate(expanded_mask, kernel_sample, iterations=1)
        
        # Sampling area = outer area - expanded area
        sampling_mask = cv2.bitwise_and(sampling_outer, cv2.bitwise_not(expanded_mask))
        
        # 3. Pre-store pixel values from entire sampling area (unified sampling pool)
        sampling_coords = np.where(sampling_mask > 0)
        
        if len(sampling_coords[0]) == 0:
            print("Warning: No valid sampling area found, using fallback method")
            return self.fallback_fill(image, human_mask)
        
        # Store all pixel values from sampling area
        if len(image.shape) == 3:
            # Color image: store RGB values
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # shape: (N, 3)
        else:
            # Grayscale image: store grayscale values
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # shape: (N,)
        
        # 4. Get coordinates of areas to fill
        fill_coords = np.where(expanded_mask > 0)
        fill_points = list(zip(fill_coords[0], fill_coords[1]))
        
        print(f"Sampling pool pixels: {len(sampling_pixels)}")
        print(f"Fill area pixels: {len(fill_points)}")
        
        # 5. For each pixel to fill, randomly select from unified sampling pool
        for fill_y, fill_x in fill_points:
            # Randomly select a pixel value from sampling pool
            random_idx = random.randint(0, len(sampling_pixels) - 1)
            result_image[fill_y, fill_x] = sampling_pixels[random_idx]
        
        print("Random sampling fill completed")
        return result_image
    
    def improved_random_sampling_fill(self, image, human_mask):
        """
        Improved random sampling fill considering distance weighting and texture consistency
        Use unified sampling pool but weighted selection based on distance
        """
        if human_mask is None:
            return image
        
        print("Using improved random sampling method for human removal...")
        result_image = image.copy()
        
        # 1. Create expanded fill area
        kernel_expand = np.ones((21, 21), np.uint8)
        expanded_mask = cv2.dilate(human_mask, kernel_expand, iterations=1)
        
        # 2. Create sampling area
        kernel_sample = np.ones((41, 41), np.uint8)
        sampling_outer = cv2.dilate(expanded_mask, kernel_sample, iterations=1)
        sampling_mask = cv2.bitwise_and(sampling_outer, cv2.bitwise_not(expanded_mask))
        
        # 3. Pre-store coordinates and pixel values from entire sampling area (unified sampling pool)
        sampling_coords = np.where(sampling_mask > 0)
        fill_coords = np.where(expanded_mask > 0)
        
        if len(sampling_coords[0]) == 0:
            return self.fallback_fill(image, human_mask)
        
        # Store sampling point coordinates and pixel values
        sampling_points = np.column_stack((sampling_coords[0], sampling_coords[1]))  # coordinates
        if len(image.shape) == 3:
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # RGB values
        else:
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # grayscale values
            
        fill_points = np.column_stack((fill_coords[0], fill_coords[1]))
        
        print(f"Sampling pool pixels: {len(sampling_points)}")
        print(f"Fill area pixels: {len(fill_points)}")
        
        # 4. For efficiency, if too many sampling points, randomly select a subset as sampling pool
        if len(sampling_points) > 1000:
            indices = random.sample(range(len(sampling_points)), 1000)
            sampling_points = sampling_points[indices]
            sampling_pixels = sampling_pixels[indices]
        
        # 5. For each fill point, weighted random selection from sampling pool based on distance
        if SCIPY_AVAILABLE:
            for i, fill_point in enumerate(fill_points):
                # Calculate distance from fill point to all sampling points
                distances = cdist([fill_point], sampling_points, metric='euclidean')[0]
                
                # Select nearest 10 sampling points for weighted random selection
                nearest_indices = np.argsort(distances)[:min(200, len(distances))]
                nearest_distances = distances[nearest_indices]
                
                # Calculate weights (closer distance = higher weight)
                weights = 1.0 / (nearest_distances + 1.0)  # Add 1 to avoid division by zero
                weights = weights / np.sum(weights)  # Normalize
                
                # Randomly select a sampling point based on weights
                chosen_local_idx = np.random.choice(len(nearest_indices), p=weights)
                chosen_global_idx = nearest_indices[chosen_local_idx]
                
                # Get corresponding pixel value from sampling pool
                fill_y, fill_x = fill_point
                result_image[fill_y, fill_x] = sampling_pixels[chosen_global_idx]
        else:
            # Fallback: use basic random sampling (from unified sampling pool)
            print("Using basic random sampling (scipy not available)")
            for i, fill_point in enumerate(fill_points):
                # Randomly select a pixel value from sampling pool
                random_idx = random.randint(0, len(sampling_pixels) - 1)
                fill_y, fill_x = fill_point
                result_image[fill_y, fill_x] = sampling_pixels[random_idx]
        
        print("Improved random sampling fill completed")
        return result_image
    
    def fallback_fill(self, image, human_mask):
        """Fallback fill method: use image inpainting"""
        print("Using fallback fill method...")
        if len(image.shape) == 3:
            return cv2.inpaint(image, human_mask, 3, cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(image, human_mask, 3, cv2.INPAINT_TELEA)
    
    def remove_humans_from_document(self, image, human_mask):
        """Remove human parts from document image - using random sampling method"""
        if human_mask is None:
            return image
        
        # Prioritize improved random sampling fill
        if SCIPY_AVAILABLE:
            result_image = self.improved_random_sampling_fill(image, human_mask)
        else:
            result_image = self.random_sampling_fill(image, human_mask)
        
        # Optional: apply slight edge feathering for further smoothing
        smooth_mask = self.create_smooth_mask(human_mask, 'gaussian', 8)
        if smooth_mask is not None:
            if len(image.shape) == 3:
                for c in range(3):
                    result_image[:, :, c] = (
                        smooth_mask * result_image[:, :, c] + 
                        (1 - smooth_mask) * image[:, :, c]
                    ).astype(np.uint8)
            else:
                result_image = (
                    smooth_mask * result_image + 
                    (1 - smooth_mask) * image
                ).astype(np.uint8)
        
        print("Random sampling human removal processing completed")
        return result_image


class IlluminationCorrector:
    """Illumination correction processor - simplified version, only keeping hybrid method"""
    
    def background_subtraction(self, image, blur_size=101):
        """Background subtraction method"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if blur_size % 2 == 0:
            blur_size += 1
            
        background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        if len(image.shape) == 3:
            corrected_color = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i].astype(np.float32)
                bg_channel = background.astype(np.float32)
                mask = bg_channel > 1
                corrected_channel = np.zeros_like(channel)
                corrected_channel[mask] = (channel[mask] / bg_channel[mask]) * 255
                corrected_channel[~mask] = channel[~mask]
                corrected_color[:, :, i] = np.clip(corrected_channel, 0, 255).astype(np.uint8)
            return corrected_color
        else:
            corrected = np.zeros_like(gray, dtype=np.float32)
            mask = background > 1
            corrected[mask] = (gray[mask].astype(np.float32) / background[mask].astype(np.float32)) * 255
            corrected[~mask] = gray[~mask].astype(np.float32)
            return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def adaptive_histogram_equalization(self, image, clip_limit=1.5):
        """Adaptive histogram equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)
    
    def enhance_document(self, image):
        """Document enhancement main method - only using hybrid method"""
        # Hybrid method: background subtraction first, then moderate CLAHE
        corrected = self.background_subtraction(image)
        corrected = self.adaptive_histogram_equalization(corrected)
        
        # Light denoising
        if len(corrected.shape) == 3:
            corrected = cv2.fastNlMeansDenoisingColored(corrected, None, 10, 10, 7, 21)
        else:
            corrected = cv2.fastNlMeansDenoising(corrected, None, 10, 7, 21)
        
        return corrected


class ClickDocumentScanner:
    def __init__(self, image_path, output_size=None, batch_mode=True):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.points = []
        self.auto_detected_points = []
        self.auto_detection_status = "failed"
        self.auto_detection_message = ""
        self.image_path = image_path
        self.output_size = output_size
        self.batch_mode = batch_mode
        self.use_auto_points = False
        self.corrector = IlluminationCorrector()  # Illumination corrector
        self.human_detector = HumanDetector()     # Human detector
        
        # Adjust display size
        self.scale = 1.0
        h, w = self.original_image.shape[:2]
        if w > 1200 or h > 800:
            self.scale = min(1200/w, 800/h)
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.display_image = cv2.resize(self.original_image, (new_w, new_h))
        
        # Auto-detect corners
        self.auto_detect_corners()
        
        if not batch_mode:
            print("Instructions:")
            if self.auto_detection_status == "success":
                print("  ✓ Auto-detection successful!")
                print("  1. 'a' to accept auto-detected points (yellow display)")
            else:
                print(f"  ✗ Auto-detection failed: {self.auto_detection_message}")
                print("  1. Need to manually select four corner points")
            print("  2. Click four corner points manually if needed")
            print("  3. Press 't' to apply perspective transform after selecting 4 points")
            print("  4. Press 'r' to reselect points")
            print("  5. Press 's' to save result")
            print("  6. Press 'q' to quit")
            # Updated feature description
            print("  7. Press 'h' for random sampling human detection and removal")
            if self.output_size:
                print(f"  Output size: {self.output_size[0]} x {self.output_size[1]} pixels")
            else:
                print("  Output size: auto-calculated")
            print("=" * 60)
    
    def auto_detect_corners(self):
        """Automatically find document corners through contour detection"""
        self.auto_detection_status = "failed"
        self.auto_detection_message = ""
        
        try:
            height = 800
            image_height, image_width = self.original_image.shape[:2]
            ratio = height / image_height
            new_width = int(image_width * ratio)
            resized_image = cv2.resize(self.original_image, (new_width, height))
            
            blurred = cv2.medianBlur(resized_image, 9)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 50, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area > 1000 and perimeter > 100:
                    filtered_contours.append((area, contour))
            
            if not filtered_contours:
                self.auto_detection_status = "no_contours"
                self.auto_detection_message = "No suitable document contours found"
                if not self.batch_mode:
                    print(f"Auto-detection failed: {self.auto_detection_message}")
                return
            
            filtered_contours.sort(key=lambda x: x[0], reverse=True)
            largest_contour = filtered_contours[0][1]
            
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) != 4:
                for eps_factor in [0.01, 0.03, 0.04, 0.05]:
                    epsilon = eps_factor * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    if len(approx) == 4:
                        break
            
            if len(approx) == 4:
                detected_points = []
                for point in approx:
                    x, y = point[0]
                    orig_x = int(x / ratio)
                    orig_y = int(y / ratio)
                    detected_points.append([orig_x, orig_y])
                
                self.auto_detected_points = detected_points
                self.auto_detection_status = "success"
                self.auto_detection_message = f"Successfully detected 4 corner points"
                if not self.batch_mode:
                    print(f"Auto-detection: {self.auto_detection_message}")
            else:
                self.auto_detection_status = "insufficient_points"
                self.auto_detection_message = f"Detected {len(approx)} points, manual selection needed"
                if not self.batch_mode:
                    print(f"Auto-detection failed: {self.auto_detection_message}")
                
        except Exception as e:
            self.auto_detection_status = "error"
            self.auto_detection_message = f"Detection process error: {str(e)}"
            if not self.batch_mode:
                print(f"Auto-detection failed: {self.auto_detection_message}")
    
    def get_current_points(self):
        """Get currently used points (auto-detected or manually selected)"""
        if self.use_auto_points and len(self.auto_detected_points) == 4:
            return self.auto_detected_points
        else:
            return self.points
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                actual_x = int(x / self.scale)
                actual_y = int(y / self.scale)
                self.points.append([actual_x, actual_y])
                self.use_auto_points = False
                
                if not self.batch_mode:
                    print(f"Manual point {len(self.points)}: ({actual_x}, {actual_y})")
                
                self.update_display()
                
                if len(self.points) == 4 and not self.batch_mode:
                    print("Selected 4 points, press 't' for perspective transform")
    
    def update_display(self):
        """Update display image"""
        self.display_image = cv2.resize(self.original_image, 
                                        (int(self.original_image.shape[1] * self.scale),
                                         int(self.original_image.shape[0] * self.scale)))
        
        display_points = self.get_current_points()
        
        for i, point in enumerate(display_points):
            display_point = (int(point[0] * self.scale), int(point[1] * self.scale))
            
            if self.use_auto_points and len(self.auto_detected_points) == 4:
                color = (0, 255, 255)
                text_color = (0, 255, 255)
            else:
                color = (0, 255, 0)
                text_color = (0, 255, 0)
            
            cv2.circle(self.display_image, display_point, 8, color, -1)
            cv2.putText(self.display_image, 
                        str(i+1), 
                        (display_point[0] + 10, display_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        text_color, 
                        2)
        
        if len(display_points) > 1:
            line_color = (255, 255, 0) if (self.use_auto_points and len(self.auto_detected_points) == 4) else (255, 0, 0)
            
            for i in range(len(display_points)):
                if i < len(display_points) - 1:
                    pt1 = (int(display_points[i][0] * self.scale), int(display_points[i][1] * self.scale))
                    pt2 = (int(display_points[i+1][0] * self.scale), int(display_points[i+1][1] * self.scale))
                    cv2.line(self.display_image, pt1, pt2, line_color, 2)
            
            if len(display_points) == 4:
                pt1 = (int(display_points[3][0] * self.scale), int(display_points[3][1] * self.scale))
                pt2 = (int(display_points[0][0] * self.scale), int(display_points[0][1] * self.scale))
                cv2.line(self.display_image, pt1, pt2, line_color, 2)
    
    def sort_points(self, points):
        """Sort points: top-left, top-right, bottom-right, bottom-left"""
        if len(points) != 4:
            return np.array(points, dtype=np.float32)
            
        points = np.array(points, dtype=np.float32)
        
        sorted_by_y = points[np.argsort(points[:, 1])]
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]
        
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        top_left, top_right = top_sorted[0], top_sorted[1]
        
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bottom_left, bottom_right = bottom_sorted[0], bottom_sorted[1]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def apply_perspective_transform(self):
        """Apply perspective transform (without illumination correction)"""
        current_points = self.get_current_points()
        
        if len(current_points) != 4:
            if not self.batch_mode:
                print("Please select four points or accept auto-detection results first")
            return None
        
        sorted_points = self.sort_points(current_points)
        
        if self.output_size:
            max_width, max_height = self.output_size
            if not self.batch_mode:
                print(f"Using specified size: {max_width} x {max_height}")
        else:
            tl, tr, br, bl = sorted_points
            
            width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
            width_bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
            max_width = max(int(width_top), int(width_bottom))
            
            height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
            height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
            max_height = max(int(height_left), int(height_right))
            
            if not self.batch_mode:
                print(f"Auto-calculated size: {max_width} x {max_height}")
        
        dst_points = np.array([[0, 0],
                               [max_width - 1, 0],
                               [max_width - 1, max_height - 1],
                               [0, max_height - 1]
                               ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
        warped = cv2.warpPerspective(self.original_image, matrix, (max_width, max_height))
        
        return warped
    
    def apply_illumination_correction(self, image):
        """Apply illumination correction - simplified version, only using hybrid method"""
        return self.corrector.enhance_document(image)
    
    def apply_human_detection_and_removal(self, image):
        """Apply human detection and removal - using random sampling method"""
        if self.human_detector.model is None:
            print("Human detection feature unavailable")
            return image, False
        
        print("Detecting humans...")
        human_mask, detected = self.human_detector.detect_humans(image)
        
        if detected and human_mask is not None:
            print("Humans detected, removing using random sampling method...")
            cleaned_image = self.human_detector.remove_humans_from_document(
                image, human_mask
            )
            return cleaned_image, True
        else:
            print("No humans detected")
            return image, False
    
    def save_image(self, image):
        """Save image"""
        base_name = os.path.splitext(self.image_path)[0]
        save_path = f"{base_name}_warped.jpg"
        
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base_name}_warped_{counter}.jpg"
            counter += 1
        
        cv2.imwrite(save_path, image)
        if not self.batch_mode:
            print(f"Image saved to: {save_path}")
        return save_path
    
    def run_batch_image(self):
        """Process single image in batch mode - using random sampling"""
        cv2.namedWindow('Document Scanner - Random Sampling Mode', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Document Scanner - Random Sampling Mode', self.mouse_callback)
        
        if self.auto_detection_status == "success":
            self.use_auto_points = True
        
        self.update_display()
        
        # State management
        state = "selecting_points"  # selecting_points, transformed, human_detected, corrected
        warped_image = None
        human_cleaned_image = None
        corrected_image = None
        final_image = None
        status_color = (0, 0, 0)
        human_detection_applied = False
        
        filename = os.path.basename(self.image_path)
        
        while True:
            if state == "selecting_points":
                # Display original image and corner selection interface
                display_img = self.display_image.copy()
                current_points = self.get_current_points()
                
                # Build processing result status
                if self.auto_detection_status == "success":
                    status_text = "Processing result: Auto detection OK"
                    status_color = (0, 255, 0)  # Green
                elif self.auto_detection_status == "insufficient_points":
                    status_text = "Processing result: Auto failed - Not enough points"
                    status_color = (0, 0, 255)  # Red
                elif self.auto_detection_status == "no_contours":
                    status_text = "Processing result: Auto failed - No contours"
                    status_color = (0, 0, 255)  # Red
                elif self.auto_detection_status == "error":
                    status_text = "Processing result: Auto failed - Error"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = "Processing result: Auto failed"
                    status_color = (0, 0, 255)  # Red
                
                if len(current_points) == 4:
                    point_source = "Auto-detected" if (self.use_auto_points and self.auto_detection_status == "success") else "Manual"
                    instructions = f"{filename}\n{status_text}\nSelected 4 points ({point_source})\na: Accept auto-detection\nt: Transform\nr: Reselect\ns: Skip\nESC: Quit"
                else:
                    manual_count = len(self.points)
                    instructions = f"{filename}\n{status_text}\nClick 4 corner points ({manual_count}/4)\na: Accept auto-detection\nr: Reselect\ns: Skip\nESC: Quit"
                    
            elif state == "transformed":
                # Display perspective transform result, ask if human detection is needed
                display_img = warped_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Transform completed"
                status_color = (0, 255, 0)  # Green
                if YOLO_AVAILABLE and self.human_detector.model is not None:
                    instructions = f"{filename}\n{status_text}\nh: Random sampling human detection\ni: Apply illumination correction\nt: Accept current result\nr: Reselect points\ns: Skip\nESC: Quit"
                else:
                    instructions = f"{filename}\n{status_text}\ni: Apply illumination correction\nt: Accept current result\nr: Reselect points\ns: Skip\nESC: Quit"
                
            elif state == "human_detected":
                # Display human detection result
                display_img = human_cleaned_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Random sampling human removal applied"
                status_color = (0, 255, 0)  # Green
                instructions = f"{filename}\n{status_text}\ni: Apply illumination correction\nt: Accept current result\nb: Back to original transform\nr: Reselect points\ns: Skip\nESC: Quit"
                
            elif state == "corrected":
                # Display illumination correction result
                display_img = corrected_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Illumination correction applied"
                status_color = (0, 255, 0)  # Green
                back_option = "b: Back to human-cleaned" if human_detection_applied else "b: Back to original transform"
                instructions = f"{filename}\n{status_text}\nt: Accept corrected result\n{back_option}\nr: Reselect points\ns: Skip\nESC: Quit"
            
            # Display instructions on image
            lines = instructions.split('\n')
            y_offset = 30
            
            for i, line in enumerate(lines):
                # Second line (Processing result) uses specific color, other lines use default color
                if i == 1 and line.startswith("Processing result:"):
                    # Use status corresponding color
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White background
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)  # Status color
                else:
                    # Other lines use default color (white background, black foreground)
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imshow('Document Scanner - Random Sampling Mode', display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('a') and state == "selecting_points":
                # Accept auto-detected points
                if self.auto_detection_status == "success" and len(self.auto_detected_points) == 4:
                    self.use_auto_points = True
                    self.points = []
                    self.update_display()
                    print("Accepted auto-detected corner points")
                else:
                    print(f"Cannot accept auto-detection: {self.auto_detection_message}")
                    
            elif key == ord('t'):
                if state == "selecting_points":
                    # Perspective transform from corner selection state
                    if len(self.get_current_points()) == 4:
                        warped_image = self.apply_perspective_transform()
                        if warped_image is not None:
                            state = "transformed"
                            final_image = warped_image
                            print("Perspective transform completed")
                        else:
                            print("Perspective transform failed")
                elif state == "transformed":
                    # Accept current transform result
                    final_image = warped_image
                    cv2.destroyAllWindows()
                    return final_image
                elif state == "human_detected":
                    # Accept human detection result
                    final_image = human_cleaned_image
                    cv2.destroyAllWindows()
                    return final_image
                elif state == "corrected":
                    # Accept illumination correction result
                    final_image = corrected_image
                    cv2.destroyAllWindows()
                    return final_image
                    
            elif key == ord('h') and state == "transformed":
                # Apply random sampling human detection
                if YOLO_AVAILABLE and self.human_detector.model is not None:
                    cleaned_image, detected = self.apply_human_detection_and_removal(warped_image)
                    if detected:
                        human_cleaned_image = cleaned_image
                        final_image = human_cleaned_image
                        state = "human_detected"
                        human_detection_applied = True
                        print("Random sampling human detection and removal completed")
                    else:
                        print("No humans detected, keeping original image")
                else:
                    print("Human detection feature unavailable")
                    
            elif key == ord('i'):
                # Apply illumination correction
                if state == "transformed":
                    print("Applying illumination correction...")
                    corrected_image = self.apply_illumination_correction(warped_image)
                    final_image = corrected_image
                    state = "corrected"
                    print("Illumination correction completed")
                elif state == "human_detected":
                    print("Applying illumination correction...")
                    corrected_image = self.apply_illumination_correction(human_cleaned_image)
                    final_image = corrected_image
                    state = "corrected"
                    print("Illumination correction completed")
                
            elif key == ord('b'):
                # Go back to previous step
                if state == "corrected":
                    if human_detection_applied:
                        final_image = human_cleaned_image
                        state = "human_detected"
                        print("Cancelled illumination correction, returned to human detection result")
                    else:
                        final_image = warped_image
                        state = "transformed"
                        print("Cancelled illumination correction, returned to perspective transform result")
                elif state == "human_detected":
                    final_image = warped_image
                    state = "transformed"
                    human_detection_applied = False
                    print("Cancelled human detection, returned to perspective transform result")
                
            elif key == ord('r'):
                # Reselect points
                self.points = []
                self.use_auto_points = False
                warped_image = None
                human_cleaned_image = None
                corrected_image = None
                final_image = None
                state = "selecting_points"
                human_detection_applied = False
                self.update_display()
                print("Reselecting corner points")
                
            elif key == ord('s'):
                # Skip this image
                cv2.destroyAllWindows()
                return None
                
            elif key == 27:  # ESC key, exit entire batch processing
                cv2.destroyAllWindows()
                return "quit"


def get_image_files(folder_path):
    """Get all image files in folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    return sorted(list(set(image_files)))


def images_to_pdf(image_paths, output_pdf_path):
    """Convert image list to PDF"""
    if not image_paths:
        print("No images to convert to PDF")
        return False
    
    try:
        first_image = Image.open(image_paths[0])
        if first_image.mode != 'RGB':
            first_image = first_image.convert('RGB')
        
        other_images = []
        for img_path in image_paths[1:]:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            other_images.append(img)
        
        first_image.save(output_pdf_path, save_all=True, append_images=other_images)
        print(f"PDF saved to: {output_pdf_path}")
        return True
    
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return False


def batch_process_folder(folder_path, output_size=None, output_pdf=None):
    """Batch process images in folder"""
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"No image files found in folder '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} images")
    print("=" * 60)
    
    processed_images = []
    temp_files = []
    auto_success_count = 0
    human_detection_count = 0
    
    try:
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                test_image = cv2.imread(image_path)
                if test_image is None:
                    print(f"✗ Cannot read image: {os.path.basename(image_path)}")
                    continue
                
                scanner = ClickDocumentScanner(image_path, output_size, batch_mode=True)
                
                if scanner.auto_detection_status == "success":
                    auto_success_count += 1
                    print(f"  └─ Auto-detection: ✓ Success")
                else:
                    print(f"  └─ Auto-detection: ✗ {scanner.auto_detection_message}")
                
                result = scanner.run_batch_image()
                
                if isinstance(result, str) and result == "quit":
                    print("User exited batch processing")
                    break
                elif result is not None and not (isinstance(result, str) and result == "quit"):
                    temp_filename = f"temp_warped_{i:03d}.jpg"
                    temp_path = os.path.join(folder_path, temp_filename)
                    cv2.imwrite(temp_path, result)
                    processed_images.append(temp_path)
                    temp_files.append(temp_path)
                    print(f"✓ Processed: {os.path.basename(image_path)}")
                else:
                    print(f"✗ Skipped: {os.path.basename(image_path)}")
            
            except Exception as e:
                import traceback
                print(f"✗ Processing failed {os.path.basename(image_path)}: {e}")
                print(f"Detailed error: {traceback.format_exc()}")
                continue
        
        if processed_images:
            if output_pdf is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = os.path.basename(os.path.abspath(folder_path))
                output_pdf = os.path.join(folder_path, f"scanned_documents_{folder_name}_{timestamp}.pdf")
            
            print(f"\nGenerating PDF: {len(processed_images)} images")
            success = images_to_pdf(processed_images, output_pdf)
            
            if success:
                print(f"Batch processing completed!")
                print(f"Processed {len(processed_images)} images")
                print(f"Auto-detection success: {auto_success_count}/{len(image_files)} images")
                print(f"PDF file: {output_pdf}")
                if YOLO_AVAILABLE:
                    print(f"Human detection feature: Available (random sampling method)")
                else:
                    print(f"Human detection feature: Unavailable (requires ultralytics installation)")
                if SCIPY_AVAILABLE:
                    print(f"Advanced edge smoothing and distance calculation: Available")
                else:
                    print(f"Advanced edge smoothing and distance calculation: Partially available (recommend installing scipy)")
            
        else:
            print("No images were successfully processed")
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Enhanced Document Scanner - Random Sampling Human Detection and Illumination Correction',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''
Enhanced Document Scanner Features:
  1. Automatic corner detection
  2. Perspective transform correction
  3. Random sampling human detection and removal (requires ultralytics installation)
  4. Illumination correction (hybrid method)
  
Install dependencies:
  pip install ultralytics scipy
  
  Keyboard controls:
  - Corner selection phase:
    a: Accept auto-detected corner points
    t: Execute perspective transform
    r: Reselect corner points
    s: Skip current image
    ESC: Exit program
    
  - After perspective transform completion:
    h: Random sampling human detection and removal
    i: Apply illumination correction (hybrid method)
    t: Accept current result
    r: Reselect corner points
    s: Skip current image
    ESC: Exit program
    
  - After random sampling human detection completion:
    i: Apply illumination correction (hybrid method)
    t: Accept current result
    b: Return to transform result
    r: Reselect corner points
    s: Skip current image
    ESC: Exit program
    
  - After illumination correction completion:
    t: Accept corrected result
    b: Return to previous step
    r: Reselect corner points
    s: Skip current image
    ESC: Exit program

Random Sampling Human Removal Algorithm:
  1. Expand human mask outward by 10 pixels as fill area
  2. Sample within 20 pixels outside the fill area
  3. For each pixel to fill, randomly select pixels from sampling area for filling
  4. Prioritize distance-weighted improved algorithm (requires scipy)

Output size formats:
  Predefined sizes:
    A4        - A4 size (2480x3508, 300 DPI)
    A4_150    - A4 size (1240x1754, 150 DPI)  
    A4_200    - A4 size (1654x2339, 200 DPI)
    A3        - A3 size (3508x4961, 300 DPI)
    LETTER    - Letter size (2550x3300, 300 DPI)
    LEGAL     - Legal size (2550x4200, 300 DPI)

Usage examples:
  1. Batch process folder: 
     python enhanced_scanner_random_sampling.py --folder=./images --image_size=A4
  
  2. Specify output PDF: 
     python enhanced_scanner_random_sampling.py --folder=./images --output_pdf=./result.pdf
  
  3. Auto-calculate size: 
     python enhanced_scanner_random_sampling.py --folder=./images --image_size=auto
        ''')
    
    parser.add_argument('--folder', type=str, required=True, help='Folder path containing images')
    parser.add_argument('--image_size', type=str, default='A4', help='Output image size (default: A4)')
    parser.add_argument('--output_pdf', type=str, help='Output PDF file path (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Enhanced Document Scanner with Random Sampling Human Detection")
    print("Enhanced Document Scanner - Random Sampling Human Detection and Illumination Correction")
    print("=" * 60)
    
    # Check dependency library availability
    if YOLO_AVAILABLE:
        print("✓ YOLO11 available - Random sampling human detection feature enabled")
    else:
        print("✗ YOLO11 unavailable - Human detection feature disabled")
        print("  Installation: pip install ultralytics")
    
    if SCIPY_AVAILABLE:
        print("✓ SciPy available - Advanced edge smoothing and distance calculation enabled")
    else:
        print("✗ SciPy unavailable - Using basic edge smoothing and random sampling")
        print("  Installation: pip install scipy")
    
    try:
        parsed_size = parse_image_size(args.image_size)
        if parsed_size:
            print(f"Set output size: {parsed_size[0]} x {parsed_size[1]} pixels")
        else:
            print("Output size: Auto-calculated")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Input folder: {args.folder}")
    print("Processing workflow: Corner detection → Perspective transform → [Random sampling human detection] → [Illumination correction(hybrid)]")
    if args.output_pdf:
        print(f"Output PDF: {args.output_pdf}")
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder does not exist - {args.folder}")
        return
    
    if not os.path.isdir(args.folder):
        print(f"Error: Not a folder - {args.folder}")
        return
    
    try:
        batch_process_folder(args.folder, parsed_size, args.output_pdf)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
