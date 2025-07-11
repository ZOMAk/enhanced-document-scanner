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

# 新增导入 - 用于高级边缘平滑和距离计算
try:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Advanced edge smoothing and distance-based sampling will use fallback methods.")


def parse_image_size(size_str):
    """解析图像尺寸字符串"""
    if not size_str or size_str.lower() == 'auto':
        return None
    
    # 移除方括号和空格
    size_str = size_str.strip('[]').replace(' ', '')
    
    # 预定义的尺寸
    predefined_sizes = {'A4': (2480, 3508),      # A4 300 DPI
                        'A4_150': (1240, 1754),  # A4 150 DPI  
                        'A4_200': (1654, 2339),  # A4 200 DPI
                        'A3': (3508, 4961),      # A3 300 DPI
                        'LETTER': (2550, 3300),  # Letter 300 DPI
                        'LEGAL': (2550, 4200),}  # Legal 300 DPI
    
    # 检查是否是预定义尺寸
    if size_str.upper() in predefined_sizes:
        return predefined_sizes[size_str.upper()]
    
    # 解析自定义尺寸
    try:
        if ',' in size_str:
            width, height = map(int, size_str.split(','))
        elif 'x' in size_str.lower():
            width, height = map(int, size_str.lower().split('x'))
        else:
            raise ValueError("无效的尺寸格式")
        
        if width <= 0 or height <= 0:
            raise ValueError("尺寸必须为正数")
        
        return (width, height)
    except ValueError as e:
        raise ValueError(f"无法解析尺寸 '{size_str}': {e}")


class HumanDetector:
    """人体检测和去除处理器 - 使用随机采样填充方法"""
    
    def __init__(self):
        self.model = None
        if YOLO_AVAILABLE:
            try:
                # 尝试加载YOLO11模型
                self.model = YOLO('yolo11n.pt')  # 使用nano版本，速度更快
                print("YOLO11模型加载成功")
            except Exception as e:
                print(f"YOLO11模型加载失败: {e}")
                self.model = None
        else:
            print("YOLO库未安装，人体检测功能不可用")
    
    def detect_humans(self, image):
        """检测图像中的人体部分"""
        if self.model is None:
            return None, None
        
        try:
            # 使用YOLO进行检测
            results = self.model(image, verbose=False)
            
            # 创建人体mask
            human_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            detected_humans = False
            
            for result in results:
                if result.masks is not None:
                    for i, class_id in enumerate(result.boxes.cls):
                        # COCO数据集中，person的类别ID是0
                        if int(class_id) == 0:  # person类别
                            mask = result.masks.data[i].cpu().numpy()
                            # 将mask resize到原图尺寸
                            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            # 转换为二值mask
                            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                            human_mask = cv2.bitwise_or(human_mask, mask_binary)
                            detected_humans = True
            
            if not detected_humans:
                # 如果没有分割结果，尝试使用边界框检测
                for result in results:
                    if result.boxes is not None:
                        for i, class_id in enumerate(result.boxes.cls):
                            if int(class_id) == 0:  # person类别
                                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = box
                                # 在检测框区域创建mask
                                human_mask[y1:y2, x1:x2] = 255
                                detected_humans = True
            
            return human_mask if detected_humans else None, detected_humans
            
        except Exception as e:
            print(f"人体检测过程中出错: {e}")
            return None, False
    
    def create_smooth_mask(self, mask, method='gaussian', smooth_radius=15):
        """创建平滑的mask，用于边缘羽化"""
        if mask is None:
            return None
        
        # 确保mask是浮点数格式
        smooth_mask = mask.astype(np.float32) / 255.0
        
        if method == 'gaussian' and SCIPY_AVAILABLE:
            # 高斯模糊羽化 - 优先使用scipy
            smooth_mask = gaussian_filter(smooth_mask, sigma=smooth_radius/3)
        elif method == 'gaussian':
            # 备用方案：使用OpenCV的高斯模糊
            kernel_size = max(3, int(smooth_radius * 2) | 1)  # 确保是奇数
            smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), smooth_radius/3)
            
        elif method == 'distance':
            # 基于距离变换的羽化
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                normalized_dist = dist_transform / max_dist
                # 使用sigmoid函数创建平滑边缘
                smooth_mask = 1 / (1 + np.exp(-10 * (normalized_dist - 0.3)))
                # 确保原始mask区域仍然是1
                smooth_mask[mask > 0] = 1.0
            
        elif method == 'morphological':
            # 形态学平滑
            kernel_size = max(3, smooth_radius // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 先膨胀再腐蚀，然后高斯模糊
            dilated = cv2.dilate(mask, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            kernel_size = max(3, int(smooth_radius / 2) | 1)
            smooth_mask = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, 
                                          (kernel_size, kernel_size), smooth_radius/4)
        
        # 确保值在[0,1]范围内
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
        
        return smooth_mask
    
    def random_sampling_fill(self, image, human_mask):
        """
        使用随机采样的方法填充人体区域
        填充范围：mask向外扩大10个像素的新区域
        采样范围：新区域外围20个像素内的区域（统一采样池）
        """
        if human_mask is None:
            return image
        
        print("使用随机采样方法进行人体去除...")
        result_image = image.copy()
        
        # 1. 创建扩展的填充区域（原mask向外扩展10像素）
        kernel_expand = np.ones((21, 21), np.uint8)  # 10*2+1 = 21
        expanded_mask = cv2.dilate(human_mask, kernel_expand, iterations=1)
        
        # 2. 创建采样区域（扩展区域外围20像素）
        kernel_sample = np.ones((41, 41), np.uint8)  # 20*2+1 = 41  
        sampling_outer = cv2.dilate(expanded_mask, kernel_sample, iterations=1)
        
        # 采样区域 = 外围区域 - 扩展区域
        sampling_mask = cv2.bitwise_and(sampling_outer, cv2.bitwise_not(expanded_mask))
        
        # 3. 预先存储整个采样区域的像素值（统一采样池）
        sampling_coords = np.where(sampling_mask > 0)
        
        if len(sampling_coords[0]) == 0:
            print("警告：没有找到有效的采样区域，使用备用方法")
            return self.fallback_fill(image, human_mask)
        
        # 存储采样区域的所有像素值
        if len(image.shape) == 3:
            # 彩色图像：存储RGB值
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # shape: (N, 3)
        else:
            # 灰度图像：存储灰度值
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # shape: (N,)
        
        # 4. 获取需要填充的区域坐标
        fill_coords = np.where(expanded_mask > 0)
        fill_points = list(zip(fill_coords[0], fill_coords[1]))
        
        print(f"采样池像素数: {len(sampling_pixels)}")
        print(f"填充区域像素数: {len(fill_points)}")
        
        # 5. 对每个需要填充的像素从统一采样池中随机选择
        for fill_y, fill_x in fill_points:
            # 从采样池中随机选择一个像素值
            random_idx = random.randint(0, len(sampling_pixels) - 1)
            result_image[fill_y, fill_x] = sampling_pixels[random_idx]
        
        print("随机采样填充完成")
        return result_image
    
    def improved_random_sampling_fill(self, image, human_mask):
        """
        改进版随机采样填充，考虑距离权重和纹理一致性
        使用统一的采样池，但根据距离进行加权选择
        """
        if human_mask is None:
            return image
        
        print("使用改进版随机采样方法进行人体去除...")
        result_image = image.copy()
        
        # 1. 创建扩展的填充区域
        kernel_expand = np.ones((21, 21), np.uint8)
        expanded_mask = cv2.dilate(human_mask, kernel_expand, iterations=1)
        
        # 2. 创建采样区域
        kernel_sample = np.ones((41, 41), np.uint8)
        sampling_outer = cv2.dilate(expanded_mask, kernel_sample, iterations=1)
        sampling_mask = cv2.bitwise_and(sampling_outer, cv2.bitwise_not(expanded_mask))
        
        # 3. 预先存储整个采样区域的坐标和像素值（统一采样池）
        sampling_coords = np.where(sampling_mask > 0)
        fill_coords = np.where(expanded_mask > 0)
        
        if len(sampling_coords[0]) == 0:
            return self.fallback_fill(image, human_mask)
        
        # 存储采样点的坐标和像素值
        sampling_points = np.column_stack((sampling_coords[0], sampling_coords[1]))  # 坐标
        if len(image.shape) == 3:
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # RGB值
        else:
            sampling_pixels = image[sampling_coords[0], sampling_coords[1]]  # 灰度值
            
        fill_points = np.column_stack((fill_coords[0], fill_coords[1]))
        
        print(f"采样池像素数: {len(sampling_points)}")
        print(f"填充区域像素数: {len(fill_points)}")
        
        # 4. 为了提高效率，如果采样点太多，随机选择一部分作为采样池
        if len(sampling_points) > 1000:
            indices = random.sample(range(len(sampling_points)), 1000)
            sampling_points = sampling_points[indices]
            sampling_pixels = sampling_pixels[indices]
        
        # 5. 对每个填充点，从采样池中根据距离进行加权随机选择
        if SCIPY_AVAILABLE:
            for i, fill_point in enumerate(fill_points):
                # 计算填充点到所有采样点的距离
                distances = cdist([fill_point], sampling_points, metric='euclidean')[0]
                
                # 选择最近的10个采样点进行加权随机选择
                nearest_indices = np.argsort(distances)[:min(200, len(distances))]
                nearest_distances = distances[nearest_indices]
                
                # 计算权重（距离越近权重越大）
                weights = 1.0 / (nearest_distances + 1.0)  # 加1避免除零
                weights = weights / np.sum(weights)  # 归一化
                
                # 根据权重随机选择一个采样点
                chosen_local_idx = np.random.choice(len(nearest_indices), p=weights)
                chosen_global_idx = nearest_indices[chosen_local_idx]
                
                # 从采样池中获取对应的像素值
                fill_y, fill_x = fill_point
                result_image[fill_y, fill_x] = sampling_pixels[chosen_global_idx]
        else:
            # 备用方案：使用基础随机采样（从统一采样池）
            print("使用基础随机采样（scipy不可用）")
            for i, fill_point in enumerate(fill_points):
                # 从采样池中随机选择一个像素值
                random_idx = random.randint(0, len(sampling_pixels) - 1)
                fill_y, fill_x = fill_point
                result_image[fill_y, fill_x] = sampling_pixels[random_idx]
        
        print("改进版随机采样填充完成")
        return result_image
    
    def fallback_fill(self, image, human_mask):
        """备用填充方法：使用图像修复"""
        print("使用备用填充方法...")
        if len(image.shape) == 3:
            return cv2.inpaint(image, human_mask, 3, cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(image, human_mask, 3, cv2.INPAINT_TELEA)
    
    def remove_humans_from_document(self, image, human_mask):
        """从文档图像中去除人体部分 - 使用随机采样方法"""
        if human_mask is None:
            return image
        
        # 优先使用改进版随机采样填充
        if SCIPY_AVAILABLE:
            result_image = self.improved_random_sampling_fill(image, human_mask)
        else:
            result_image = self.random_sampling_fill(image, human_mask)
        
        # 可选：应用轻微的边缘羽化以进一步平滑过渡
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
        
        print("随机采样人体去除处理完成")
        return result_image


class IlluminationCorrector:
    """光照校正处理器 - 简化版，只保留hybrid方法"""
    
    def background_subtraction(self, image, blur_size=101):
        """背景减除法"""
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
        """自适应直方图均衡化"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)
    
    def enhance_document(self, image):
        """文档增强主方法 - 只使用hybrid方法"""
        # 混合方法：先背景减除，再适度CLAHE
        corrected = self.background_subtraction(image)
        corrected = self.adaptive_histogram_equalization(corrected)
        
        # 轻微去噪
        if len(corrected.shape) == 3:
            corrected = cv2.fastNlMeansDenoisingColored(corrected, None, 10, 10, 7, 21)
        else:
            corrected = cv2.fastNlMeansDenoising(corrected, None, 10, 7, 21)
        
        return corrected


class ClickDocumentScanner:
    def __init__(self, image_path, output_size=None, batch_mode=True):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.points = []
        self.auto_detected_points = []
        self.auto_detection_status = "failed"
        self.auto_detection_message = ""
        self.image_path = image_path
        self.output_size = output_size
        self.batch_mode = batch_mode
        self.use_auto_points = False
        self.corrector = IlluminationCorrector()  # 光照校正器
        self.human_detector = HumanDetector()     # 人体检测器
        
        # 调整显示大小
        self.scale = 1.0
        h, w = self.original_image.shape[:2]
        if w > 1200 or h > 800:
            self.scale = min(1200/w, 800/h)
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.display_image = cv2.resize(self.original_image, (new_w, new_h))
        
        # 自动检测角点
        self.auto_detect_corners()
        
        if not batch_mode:
            print("Instructions:")
            if self.auto_detection_status == "success":
                print("  ✓ 自动检测成功！")
                print("  1. 'a' to accept auto-detected points (黄色显示)")
            else:
                print(f"  ✗ 自动检测失败: {self.auto_detection_message}")
                print("  1. 需要手动选择四个角点")
            print("  2. Click four corner points manually if needed")
            print("  3. Press 't' to apply perspective transform after selecting 4 points")
            print("  4. Press 'r' to reselect points")
            print("  5. Press 's' to save result")
            print("  6. Press 'q' to quit")
            # 更新功能说明
            print("  7. Press 'h' for random sampling human detection and removal")
            if self.output_size:
                print(f"  Output size: {self.output_size[0]} x {self.output_size[1]} pixels")
            else:
                print("  Output size: auto-calculated")
            print("=" * 60)
    
    def auto_detect_corners(self):
        """通过轮廓检测自动找到文档的角点"""
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
                self.auto_detection_message = "未找到合适的文档轮廓"
                if not self.batch_mode:
                    print(f"自动检测失败: {self.auto_detection_message}")
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
                self.auto_detection_message = f"成功检测到 4 个角点"
                if not self.batch_mode:
                    print(f"自动检测: {self.auto_detection_message}")
            else:
                self.auto_detection_status = "insufficient_points"
                self.auto_detection_message = f"检测到 {len(approx)} 个点，需要手动选择"
                if not self.batch_mode:
                    print(f"自动检测失败: {self.auto_detection_message}")
                
        except Exception as e:
            self.auto_detection_status = "error"
            self.auto_detection_message = f"检测过程出错: {str(e)}"
            if not self.batch_mode:
                print(f"自动检测失败: {self.auto_detection_message}")
    
    def get_current_points(self):
        """获取当前使用的点（自动检测的或手动选择的）"""
        if self.use_auto_points and len(self.auto_detected_points) == 4:
            return self.auto_detected_points
        else:
            return self.points
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                actual_x = int(x / self.scale)
                actual_y = int(y / self.scale)
                self.points.append([actual_x, actual_y])
                self.use_auto_points = False
                
                if not self.batch_mode:
                    print(f"手动选择点 {len(self.points)}: ({actual_x}, {actual_y})")
                
                self.update_display()
                
                if len(self.points) == 4 and not self.batch_mode:
                    print("已选择4个点，按 't' 进行透视变换")
    
    def update_display(self):
        """更新显示图像"""
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
        """排序点：左上、右上、右下、左下"""
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
        """应用透视变换（不包含光照校正）"""
        current_points = self.get_current_points()
        
        if len(current_points) != 4:
            if not self.batch_mode:
                print("请先选择四个点或接受自动检测结果")
            return None
        
        sorted_points = self.sort_points(current_points)
        
        if self.output_size:
            max_width, max_height = self.output_size
            if not self.batch_mode:
                print(f"使用指定尺寸: {max_width} x {max_height}")
        else:
            tl, tr, br, bl = sorted_points
            
            width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
            width_bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
            max_width = max(int(width_top), int(width_bottom))
            
            height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
            height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
            max_height = max(int(height_left), int(height_right))
            
            if not self.batch_mode:
                print(f"自动计算尺寸: {max_width} x {max_height}")
        
        dst_points = np.array([[0, 0],
                               [max_width - 1, 0],
                               [max_width - 1, max_height - 1],
                               [0, max_height - 1]
                               ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
        warped = cv2.warpPerspective(self.original_image, matrix, (max_width, max_height))
        
        return warped
    
    def apply_illumination_correction(self, image):
        """应用光照校正 - 简化版，只使用hybrid方法"""
        return self.corrector.enhance_document(image)
    
    def apply_human_detection_and_removal(self, image):
        """应用人体检测和去除 - 使用随机采样方法"""
        if self.human_detector.model is None:
            print("人体检测功能不可用")
            return image, False
        
        print("正在检测人体...")
        human_mask, detected = self.human_detector.detect_humans(image)
        
        if detected and human_mask is not None:
            print("检测到人体，正在使用随机采样方法去除...")
            cleaned_image = self.human_detector.remove_humans_from_document(
                image, human_mask
            )
            return cleaned_image, True
        else:
            print("未检测到人体")
            return image, False
    
    def save_image(self, image):
        """保存图像"""
        base_name = os.path.splitext(self.image_path)[0]
        save_path = f"{base_name}_warped.jpg"
        
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base_name}_warped_{counter}.jpg"
            counter += 1
        
        cv2.imwrite(save_path, image)
        if not self.batch_mode:
            print(f"图像已保存到: {save_path}")
        return save_path
    
    def run_batch_image(self):
        """处理批量模式中的单张图片 - 使用随机采样"""
        cv2.namedWindow('Document Scanner - Random Sampling Mode', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Document Scanner - Random Sampling Mode', self.mouse_callback)
        
        if self.auto_detection_status == "success":
            self.use_auto_points = True
        
        self.update_display()
        
        # 状态管理
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
                # 显示原图和角点选择界面
                display_img = self.display_image.copy()
                current_points = self.get_current_points()
                
                # 构建处理结果状态
                if self.auto_detection_status == "success":
                    status_text = "Processing result: Auto detection OK"
                    status_color = (0, 255, 0)  # 绿色
                elif self.auto_detection_status == "insufficient_points":
                    status_text = "Processing result: Auto failed - Not enough points"
                    status_color = (0, 0, 255)  # 红色
                elif self.auto_detection_status == "no_contours":
                    status_text = "Processing result: Auto failed - No contours"
                    status_color = (0, 0, 255)  # 红色
                elif self.auto_detection_status == "error":
                    status_text = "Processing result: Auto failed - Error"
                    status_color = (0, 0, 255)  # 红色
                else:
                    status_text = "Processing result: Auto failed"
                    status_color = (0, 0, 255)  # 红色
                
                if len(current_points) == 4:
                    point_source = "Auto-detected" if (self.use_auto_points and self.auto_detection_status == "success") else "Manual"
                    instructions = f"{filename}\n{status_text}\nSelected 4 points ({point_source})\na: Accept auto-detection\nt: Transform\nr: Reselect\ns: Skip\nESC: Quit"
                else:
                    manual_count = len(self.points)
                    instructions = f"{filename}\n{status_text}\nClick 4 corner points ({manual_count}/4)\na: Accept auto-detection\nr: Reselect\ns: Skip\nESC: Quit"
                    
            elif state == "transformed":
                # 显示透视变换结果，询问是否需要人体检测
                display_img = warped_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Transform completed"
                status_color = (0, 255, 0)  # 绿色
                if YOLO_AVAILABLE and self.human_detector.model is not None:
                    instructions = f"{filename}\n{status_text}\nh: Random sampling human detection\ni: Apply illumination correction\nt: Accept current result\nr: Reselect points\ns: Skip\nESC: Quit"
                else:
                    instructions = f"{filename}\n{status_text}\ni: Apply illumination correction\nt: Accept current result\nr: Reselect points\ns: Skip\nESC: Quit"
                
            elif state == "human_detected":
                # 显示人体检测结果
                display_img = human_cleaned_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Random sampling human removal applied"
                status_color = (0, 255, 0)  # 绿色
                instructions = f"{filename}\n{status_text}\ni: Apply illumination correction\nt: Accept current result\nb: Back to original transform\nr: Reselect points\ns: Skip\nESC: Quit"
                
            elif state == "corrected":
                # 显示光照校正结果
                display_img = corrected_image.copy()
                h, w = display_img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                status_text = "Processing result: Illumination correction applied"
                status_color = (0, 255, 0)  # 绿色
                back_option = "b: Back to human-cleaned" if human_detection_applied else "b: Back to original transform"
                instructions = f"{filename}\n{status_text}\nt: Accept corrected result\n{back_option}\nr: Reselect points\ns: Skip\nESC: Quit"
            
            # 在图像上显示说明
            lines = instructions.split('\n')
            y_offset = 30
            
            for i, line in enumerate(lines):
                # 第二行（Processing result）使用特定颜色，其他行使用默认颜色
                if i == 1 and line.startswith("Processing result:"):
                    # 使用状态对应的颜色
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 白色背景
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)  # 状态颜色
                else:
                    # 其他行使用默认颜色（白色背景，黑色前景）
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, line, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imshow('Document Scanner - Random Sampling Mode', display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('a') and state == "selecting_points":
                # 接受自动检测的点
                if self.auto_detection_status == "success" and len(self.auto_detected_points) == 4:
                    self.use_auto_points = True
                    self.points = []
                    self.update_display()
                    print("已接受自动检测的角点")
                else:
                    print(f"无法接受自动检测: {self.auto_detection_message}")
                    
            elif key == ord('t'):
                if state == "selecting_points":
                    # 从角点选择状态进行透视变换
                    if len(self.get_current_points()) == 4:
                        warped_image = self.apply_perspective_transform()
                        if warped_image is not None:
                            state = "transformed"
                            final_image = warped_image
                            print("透视变换完成")
                        else:
                            print("透视变换失败")
                elif state == "transformed":
                    # 接受当前变换结果
                    final_image = warped_image
                    cv2.destroyAllWindows()
                    return final_image
                elif state == "human_detected":
                    # 接受人体检测结果
                    final_image = human_cleaned_image
                    cv2.destroyAllWindows()
                    return final_image
                elif state == "corrected":
                    # 接受光照校正结果
                    final_image = corrected_image
                    cv2.destroyAllWindows()
                    return final_image
                    
            elif key == ord('h') and state == "transformed":
                # 应用随机采样人体检测
                if YOLO_AVAILABLE and self.human_detector.model is not None:
                    cleaned_image, detected = self.apply_human_detection_and_removal(warped_image)
                    if detected:
                        human_cleaned_image = cleaned_image
                        final_image = human_cleaned_image
                        state = "human_detected"
                        human_detection_applied = True
                        print("Random sampling human detection 和 removal 完成")
                    else:
                        print("未检测到人体，保持原图")
                else:
                    print("人体检测功能不可用")
                    
            elif key == ord('i'):
                # 应用光照校正
                if state == "transformed":
                    print("正在应用光照校正...")
                    corrected_image = self.apply_illumination_correction(warped_image)
                    final_image = corrected_image
                    state = "corrected"
                    print("光照校正完成")
                elif state == "human_detected":
                    print("正在应用光照校正...")
                    corrected_image = self.apply_illumination_correction(human_cleaned_image)
                    final_image = corrected_image
                    state = "corrected"
                    print("光照校正完成")
                
            elif key == ord('b'):
                # 返回上一步
                if state == "corrected":
                    if human_detection_applied:
                        final_image = human_cleaned_image
                        state = "human_detected"
                        print("已取消光照校正，返回人体检测结果")
                    else:
                        final_image = warped_image
                        state = "transformed"
                        print("已取消光照校正，返回透视变换结果")
                elif state == "human_detected":
                    final_image = warped_image
                    state = "transformed"
                    human_detection_applied = False
                    print("已取消人体检测，返回透视变换结果")
                
            elif key == ord('r'):
                # 重新选择点
                self.points = []
                self.use_auto_points = False
                warped_image = None
                human_cleaned_image = None
                corrected_image = None
                final_image = None
                state = "selecting_points"
                human_detection_applied = False
                self.update_display()
                print("重新选择角点")
                
            elif key == ord('s'):
                # 跳过此图片
                cv2.destroyAllWindows()
                return None
                
            elif key == 27:  # ESC键，退出整个批量处理
                cv2.destroyAllWindows()
                return "quit"


def get_image_files(folder_path):
    """获取文件夹中所有图片文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    return sorted(list(set(image_files)))


def images_to_pdf(image_paths, output_pdf_path):
    """将图片列表转换为PDF"""
    if not image_paths:
        print("没有图片可转换为PDF")
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
        print(f"PDF已保存到: {output_pdf_path}")
        return True
    
    except Exception as e:
        print(f"转换PDF时出错: {e}")
        return False


def batch_process_folder(folder_path, output_size=None, output_pdf=None):
    """批量处理文件夹中的图片"""
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"在文件夹 '{folder_path}' 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("=" * 60)
    
    processed_images = []
    temp_files = []
    auto_success_count = 0
    human_detection_count = 0
    
    try:
        for i, image_path in enumerate(image_files):
            print(f"处理中 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                test_image = cv2.imread(image_path)
                if test_image is None:
                    print(f"✗ 无法读取图片: {os.path.basename(image_path)}")
                    continue
                
                scanner = ClickDocumentScanner(image_path, output_size, batch_mode=True)
                
                if scanner.auto_detection_status == "success":
                    auto_success_count += 1
                    print(f"  └─ 自动检测: ✓ 成功")
                else:
                    print(f"  └─ 自动检测: ✗ {scanner.auto_detection_message}")
                
                result = scanner.run_batch_image()
                
                if isinstance(result, str) and result == "quit":
                    print("用户退出批量处理")
                    break
                elif result is not None and not (isinstance(result, str) and result == "quit"):
                    temp_filename = f"temp_warped_{i:03d}.jpg"
                    temp_path = os.path.join(folder_path, temp_filename)
                    cv2.imwrite(temp_path, result)
                    processed_images.append(temp_path)
                    temp_files.append(temp_path)
                    print(f"✓ 已处理: {os.path.basename(image_path)}")
                else:
                    print(f"✗ 已跳过: {os.path.basename(image_path)}")
            
            except Exception as e:
                import traceback
                print(f"✗ 处理失败 {os.path.basename(image_path)}: {e}")
                print(f"详细错误: {traceback.format_exc()}")
                continue
        
        if processed_images:
            if output_pdf is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = os.path.basename(os.path.abspath(folder_path))
                output_pdf = os.path.join(folder_path, f"scanned_documents_{folder_name}_{timestamp}.pdf")
            
            print(f"\n生成PDF: {len(processed_images)} 张图片")
            success = images_to_pdf(processed_images, output_pdf)
            
            if success:
                print(f"批量处理完成!")
                print(f"已处理 {len(processed_images)} 张图片")
                print(f"自动检测成功: {auto_success_count}/{len(image_files)} 张")
                print(f"PDF文件: {output_pdf}")
                if YOLO_AVAILABLE:
                    print(f"人体检测功能: 可用 (随机采样方法)")
                else:
                    print(f"人体检测功能: 不可用 (需要安装ultralytics)")
                if SCIPY_AVAILABLE:
                    print(f"高级边缘平滑和距离计算: 可用")
                else:
                    print(f"高级边缘平滑和距离计算: 部分可用 (建议安装scipy)")
            
        else:
            print("没有图片被成功处理")
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='增强版文档扫描器 - 随机采样人体检测和光照校正',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''
增强版文档扫描器功能说明:
  1. 自动角点检测
  2. 透视变换纠正
  3. 随机采样人体检测和去除 (需要安装ultralytics)
  4. 光照校正 (hybrid方法)
  
安装依赖:
  pip install ultralytics scipy
  
  键盘操作:
  - 角点选择阶段:
    a: 接受自动检测的角点
    t: 执行透视变换
    r: 重新选择角点
    s: 跳过当前图片
    ESC: 退出程序
    
  - 透视变换完成后:
    h: 随机采样人体检测和去除
    i: 应用光照校正 (hybrid方法)
    t: 接受当前结果
    r: 重新选择角点
    s: 跳过当前图片
    ESC: 退出程序
    
  - 随机采样人体检测完成后:
    i: 应用光照校正 (hybrid方法)
    t: 接受当前结果
    b: 返回到变换结果
    r: 重新选择角点
    s: 跳过当前图片
    ESC: 退出程序
    
  - 光照校正完成后:
    t: 接受校正结果
    b: 返回到上一步
    r: 重新选择角点
    s: 跳过当前图片
    ESC: 退出程序

随机采样人体去除算法:
  1. 将人体mask向外扩展10像素作为填充区域
  2. 在填充区域外围20像素范围内进行采样
  3. 对每个需要填充的像素，随机选择采样区域的像素进行填充
  4. 优先使用距离加权的改进算法（需要scipy）

输出尺寸格式:
  预定义尺寸:
    A4        - A4尺寸 (2480x3508, 300 DPI)
    A4_150    - A4尺寸 (1240x1754, 150 DPI)  
    A4_200    - A4尺寸 (1654x2339, 200 DPI)
    A3        - A3尺寸 (3508x4961, 300 DPI)
    LETTER    - Letter尺寸 (2550x3300, 300 DPI)
    LEGAL     - Legal尺寸 (2550x4200, 300 DPI)

使用例子：
  1. 批量处理文件夹: 
     python enhanced_scanner_random_sampling.py --folder=./images --image_size=A4
  
  2. 指定输出PDF: 
     python enhanced_scanner_random_sampling.py --folder=./images --output_pdf=./result.pdf
  
  3. 自动计算尺寸: 
     python enhanced_scanner_random_sampling.py --folder=./images --image_size=auto
        ''')
    
    parser.add_argument('--folder', type=str, required=True, help='包含图片的文件夹路径')
    parser.add_argument('--image_size', type=str, default='A4', help='输出图像尺寸 (默认: A4)')
    parser.add_argument('--output_pdf', type=str, help='输出PDF文件路径 (如未指定将自动生成)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Enhanced Document Scanner with Random Sampling Human Detection")
    print("增强版文档扫描器 - 随机采样人体检测和光照校正")
    print("=" * 60)
    
    # 检查依赖库可用性
    if YOLO_AVAILABLE:
        print("✓ YOLO11可用 - 随机采样人体检测功能已启用")
    else:
        print("✗ YOLO11不可用 - 人体检测功能已禁用")
        print("  安装方法: pip install ultralytics")
    
    if SCIPY_AVAILABLE:
        print("✓ SciPy可用 - 高级边缘平滑和距离计算已启用")
    else:
        print("✗ SciPy不可用 - 使用基础边缘平滑和随机采样")
        print("  安装方法: pip install scipy")
    
    try:
        parsed_size = parse_image_size(args.image_size)
        if parsed_size:
            print(f"设置输出尺寸: {parsed_size[0]} x {parsed_size[1]} 像素")
        else:
            print("输出尺寸: 自动计算")
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    print(f"输入文件夹: {args.folder}")
    print("处理流程: 角点检测 → 透视变换 → [随机采样人体检测] → [光照校正(hybrid)]")
    if args.output_pdf:
        print(f"输出PDF: {args.output_pdf}")
    
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹不存在 - {args.folder}")
        return
    
    if not os.path.isdir(args.folder):
        print(f"错误: 不是文件夹 - {args.folder}")
        return
    
    try:
        batch_process_folder(args.folder, parsed_size, args.output_pdf)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()