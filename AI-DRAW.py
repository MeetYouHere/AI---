import os
import json
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk # Need ImageTk to display PIL images in Tkinter
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import math # For Voronoi distance calculation
import traceback # Import traceback for detailed error logging
import threading # Import threading for potential future optimization

# Check if 'noise' library is available
try:
    import noise
    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("警告: 未安装 'noise' 库. Perlin/Simplex 噪声算法不可用.")
    print("请运行 'pip install noise' 安装.")


# --- 数据存储文件 ---
LEARNING_DATA_FILE = "learning_data.json"
FEEDBACK_DATA_FILE = "feedback_data.json"

# --- 辅助函数：颜色元组与字符串转换 (更安全的格式) ---
def color_tuple_to_str(color: Tuple[int, int, int]) -> str:
    """将颜色元组 (r, g, b) 转换为字符串 'r,g,b'."""
    # Ensure values are within valid range before converting
    valid_color = (max(0, min(255, color[0])),
                   max(0, min(255, color[1])),
                   max(0, min(255, color[2])))
    return f"{valid_color[0]},{valid_color[1]},{valid_color[2]}"

def color_str_to_tuple(color_str: str) -> Optional[Tuple[int, int, int]]:
    """将颜色字符串 'r,g,b' 转换为颜色元组 (r, g, b)."""
    try:
        r, g, b = map(int, color_str.split(','))
        # Validate color values are within 0-255 range
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            return (r, g, b)
        else:
            # print(f"Warning: Invalid color values in string '{color_str}'.") # Avoid excessive printing
            return None
    except ValueError:
        # print(f"Warning: Invalid color string format '{color_str}'.") # Avoid excessive printing
        return None
    except Exception as e:
        print(f"解析颜色字符串时发生未知错误 '{color_str}': {e}")
        return None

# --- 数据加载与保存 ---
def ensure_data_files_exist():
    """创建空数据文件如果它们不存在."""
    if not os.path.exists(LEARNING_DATA_FILE):
        try:
            with open(LEARNING_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            print(f"创建了空文件：{LEARNING_DATA_FILE}")
        except Exception as e:
             print(f"创建文件时出错 {LEARNING_DATA_FILE}: {e}")


    if not os.path.exists(FEEDBACK_DATA_FILE):
        try:
            with open(FEEDBACK_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            print(f"创建了空文件：{FEEDBACK_DATA_FILE}")
        except Exception as e:
            print(f"创建文件时出错 {FEEDBACK_DATA_FILE}: {e}")


ensure_data_files_exist()

def load_data(filepath: str) -> Dict[str, Any]:
    """从 JSON 文件加载数据，处理空文件或无效文件."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Use utf-8 for Chinese text
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"警告: 无法从文件加载数据 {filepath}. 错误: {e}. 将使用空数据启动.")
        return {}
    except Exception as e:
        print(f"加载文件时发生未知错误 {filepath}: {e}")
        return {}


def save_data(data: Dict[str, Any], filepath: str):
    """保存数据到 JSON 文件."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f: # Use utf-8 for Chinese text
            json.dump(data, f, indent=4, ensure_ascii=False) # ensure_ascii=False allows saving Chinese chars directly
    except Exception as e:
        print(f"保存数据到文件时出错 {filepath}: {e}")


# --- AIGenerator 类 (后端逻辑) ---
# 核心逻辑与上一版类似，但会使用新的颜色字符串转换函数和更强的参数验证

class AIGenerator:
    def __init__(self):
        self.learning_data = load_data(LEARNING_DATA_FILE)
        self.feedback_data = load_data(FEEDBACK_DATA_FILE)
        self.last_generated_params: Optional[Dict[str, Any]] = None
        self.last_generated_keyword: Optional[str] = None
        self.last_generated_mode: Optional[str] = None
        self.current_image: Optional[Image.Image] = None # Store the current PIL image object

    def save_all_data(self):
        """保存所有学习和反馈数据."""
        save_data(self.learning_data, LEARNING_DATA_FILE)
        save_data(self.feedback_data, FEEDBACK_DATA_FILE)
        # print("所有数据已保存.") # GUI handles status update

    # --- 方式一：学习与数据提取 ---

    def extract_image_features(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        从图片中提取特征：颜色直方图，简化的颜色共现。
        返回特征字典或 None (如果失败)。
        """
        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            pixels = img.load()

            color_histogram: Dict[str, int] = {}
            color_neighbor_counts: Dict[str, Dict[str, int]] = {} # {(r,g,b): {(nr,ng,nb): count, ...}}

            for y in range(height):
                for x in range(width):
                    p1 = pixels[x, y]
                    p1_str = color_tuple_to_str(p1)

                    color_histogram[p1_str] = color_histogram.get(p1_str, 0) + 1

                    # Check right neighbor
                    if x < width - 1:
                        p2 = pixels[x+1, y]
                        p2_str = color_tuple_to_str(p2)
                        if p1_str not in color_neighbor_counts:
                            color_neighbor_counts[p1_str] = {}
                        color_neighbor_counts[p1_str][p2_str] = color_neighbor_counts[p1_str].get(p2_str, 0) + 1

                    # Check down neighbor
                    if y < height - 1:
                        p2 = pixels[x, y+1]
                        p2_str = color_tuple_to_str(p2)
                        if p1_str not in color_neighbor_counts:
                            color_neighbor_counts[p1_str] = {}
                        color_neighbor_counts[p1_str][p2_str] = color_neighbor_counts[p1_str].get(p2_str, 0) + 1

            # print(f"已从图片 {image_path} 提取特征.") # GUI handles status update
            return {
                "color_histogram": color_histogram,
                "color_neighbor_counts": color_neighbor_counts,
            }

        except FileNotFoundError:
            # print(f"错误: 找不到图片文件 {image_path}") # GUI handles error message
            return None
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {e}")
            return None

    def learn_from_image(self, image_path: str, keyword: str) -> bool:
        """加载图片，提取特征，并与关键词关联保存/合并。"""
        if not keyword:
             # print("错误: 关键词不能为空.") # GUI handles error message
             return False

        features = self.extract_image_features(image_path)
        if features:
            if keyword not in self.learning_data:
                self.learning_data[keyword] = {
                    "color_histogram": {},
                    "color_neighbor_counts": {},
                    "source_images": []
                }

            # 合并直方图
            for color_str, count in features["color_histogram"].items():
                 self.learning_data[keyword]["color_histogram"][color_str] = \
                     self.learning_data[keyword]["color_histogram"].get(color_str, 0) + count

            # 合并颜色共现计数
            for p1_str, neighbors in features["color_neighbor_counts"].items():
                 if p1_str not in self.learning_data[keyword]["color_neighbor_counts"]:
                      self.learning_data[keyword]["color_neighbor_counts"][p1_str] = {}
                 for p2_str, count in neighbors.items():
                     self.learning_data[keyword]["color_neighbor_counts"][p1_str][p2_str] = \
                         self.learning_data[keyword]["color_neighbor_counts"][p1_str].get(p2_str, 0) + count

            # 避免重复记录同一源文件
            if image_path not in self.learning_data[keyword]["source_images"]:
                 self.learning_data[keyword]["source_images"].append(image_path)

            # print(f"关键词 '{keyword}' 的特征已更新.") # GUI handles status update
            self.save_all_data()
            return True
        return False

    def generate_from_learned(self, keyword: str, params: Dict[str, Any]) -> Optional[Image.Image]:
        """
        根据学习到的关键词特征生成图片。
        算法思路：结合Perlin噪声提供结构，根据学习到的颜色直方图分配颜色，
        并使用Perlin值作为索引或权重来选择颜色，结合学习到的颜色分布。
        """
        if keyword not in self.learning_data or not self.learning_data[keyword].get("color_histogram"):
            # print(f"错误: 关键词 '{keyword}' 没有足够的学习数据.") # GUI handles error message
            return None

        width, height = params.get('image_size', (128, 128))
        randomness = float(params.get('randomness', 0.5))
        learned_feature_strength = float(params.get('learned_feature_strength', 1.0))
        seed = int(params.get('seed', random.randint(0, 10000)))
        scale = float(params.get('scale', 50.0))
        octaves = int(params.get('octaves', 6))

        if width <= 0 or height <= 0:
            # print("错误: 图片尺寸必须大于0.") # GUI handles error message
            return None

        img = Image.new('RGB', (width, height))
        pixels = img.load()

        # Prepare learned color data
        color_hist = self.learning_data[keyword]["color_histogram"]
        color_counts: List[Tuple[Tuple[int, int, int], int]] = []
        for color_str, count in color_hist.items():
            color_tuple = color_str_to_tuple(color_str)
            if color_tuple:
                color_counts.append((color_tuple, count))

        if not color_counts:
             # print(f"错误: 学习数据中没有有效的颜色信息.") # GUI handles error message
             return None

        # Sort colors by frequency (optional, can help with mapping)
        color_counts.sort(key=lambda item: item[1], reverse=True)
        colors = [item[0] for item in color_counts]
        counts = [item[1] for item in color_counts]
        total_pixels_learned = sum(counts)

        if total_pixels_learned == 0:
             # print("错误: 学习数据总像素数为零，无法生成.") # GUI handles error message
             return None

        # Calculate cumulative probabilities to map a 0-1 value to a color index
        cumulative_probabilities = np.cumsum([c / total_pixels_learned for c in counts])

        # Use Perlin noise for structure
        perlin_normalized = np.zeros((height, width))
        if PERLIN_AVAILABLE:
             # Ensure scale is not zero or negative
             current_scale = max(1.0, scale * (1.0 + randomness * 0.5))
             current_octaves = max(1, octaves + int(randomness * 4))
             try:
                 for y in range(height):
                     for x in range(width):
                         value = noise.pnoise2(x / current_scale,
                                               y / current_scale,
                                               octaves=current_octaves,
                                               persistence=0.5 + randomness*0.2,
                                               lacunarity=2.0,
                                               repeatx=width,
                                               repeaty=height,
                                               base=seed)
                         perlin_normalized[y, x] = (value + 1) / 2.0 # Map noise value (-1 to 1) to 0-1 range
             except Exception as e:
                 print(f"Perlin噪声生成时出错: {e}")
                 # Fallback to gradient if Perlin fails
                 # Don't change algorithm_type here, just use gradient logic below
                 perlin_normalized = np.zeros((height, width))
                 for y in range(height):
                    for x in range(width):
                        perlin_normalized[y, x] = x / width # Horizontal gradient 0-1

        else:
            # Fallback: Simple gradient
            for y in range(height):
                for x in range(width):
                    perlin_normalized[y, x] = x / width # Horizontal gradient 0-1

        # Map Perlin noise values to learned colors based on their frequency
        for y in range(height):
            for x in range(width):
                perlin_val = perlin_normalized[y, x] # Value is 0-1

                # Base color choice based on Perlin value and learned color distribution
                if random.random() < learned_feature_strength:
                     # Index into cumulative probabilities to pick a color biased by Perlin value
                     target_prob = max(0.0, min(1.0, perlin_val + random.uniform(-randomness*0.1, randomness*0.1)))
                     chosen_color_index = np.searchsorted(cumulative_probabilities, target_prob)
                     # Handle edge case where target_prob might be out of bounds
                     chosen_color_index = min(chosen_color_index, len(colors) - 1)
                     base_color = colors[chosen_color_index]
                else:
                    # Fallback: Pick a random color from the learned histogram (less influenced by Perlin)
                    base_color = random.choice(colors)


                # Apply additional randomness to the color (color variation)
                if randomness > 0:
                    color_variation_amount = int(randomness * 40) # Increased range for more variation
                    chosen_color = tuple(
                        max(0, min(255, c + random.randint(-color_variation_amount, color_variation_amount)))
                        for c in base_color
                    )
                else:
                    chosen_color = base_color

                pixels[x, y] = chosen_color

        # print(f"已根据关键词 '{keyword}' 生成图片.") # GUI handles status update
        return img

    # --- 方式二：结构化算法生成 ---

    def generate_structured(self, params: Dict[str, Any]) -> Optional[Image.Image]:
        """
        使用结构化噪声或其他算法生成图片。
        """
        width, height = params.get('image_size', (128, 128))
        randomness = float(params.get('randomness', 0.5))
        seed = int(params.get('seed', random.randint(0, 10000)))
        algorithm_type = params.get('algorithm_type', 'perlin' if PERLIN_AVAILABLE else 'gradient')
        scale = float(params.get('scale', 100.0))
        octaves = int(params.get('octaves', 6))
        num_points = int(params.get('num_points', 50))
        iterations = int(params.get('iterations', 10))

        if width <= 0 or height <= 0:
            # print("错误: 图片尺寸必须大于0.") # GUI handles error message
            return None
        if num_points <= 1 and algorithm_type == 'voronoi':
             # print("错误: Voronoi点数量必须大于1.") # GUI handles error message
             return None
        # Scale can be small but not zero or negative for Perlin
        if scale <= 0 and algorithm_type in ['perlin', 'simplex']:
             # print("错误: Perlin噪声缩放值必须大于0.") # GUI handles error message
             return None
        if octaves <= 0 and algorithm_type in ['perlin', 'simplex']:
             # print("错误: Perlin噪声层数必须大于0.") # GUI handles error message
             return None
        if iterations < 0 and algorithm_type == 'cellular_automata':
             # print("错误: CA迭代次数必须非负.") # GUI handles error message
             return None


        img = Image.new('RGB', (width, height))
        pixels = img.load()
        np.random.seed(seed)
        random.seed(seed)

        # print(f"使用算法 '{algorithm_type}' 生成图片...") # GUI handles status update

        # Use a flag to track if a valid algorithm ran, fallback if not
        algorithm_executed = False

        if algorithm_type == 'perlin' and PERLIN_AVAILABLE:
             try:
                 current_scale = max(1.0, scale * (1.0 + randomness))
                 current_octaves = max(1, octaves + int(randomness * 4))
                 for y in range(height):
                     for x in range(width):
                         noise_val = noise.pnoise2(x / current_scale,
                                                   y / current_scale,
                                                   octaves=current_octaves,
                                                   persistence=0.5 + randomness*0.3,
                                                   lacunarity=2.0,
                                                   repeatx=width,
                                                   repeaty=height,
                                                   base=seed)
                         color_val = int((noise_val + 1) * 127.5)
                         # Simple color mapping based on value and randomness
                         r = min(255, max(0, color_val + int(randomness * 80 * (noise_val))))
                         g = min(255, max(0, color_val + int(randomness * 80 * (-noise_val))))
                         b = min(255, max(0, color_val + int(randomness * 80 * (noise_val*0.5))))
                         pixels[x, y] = (r, g, b)
                 algorithm_executed = True
             except Exception as e:
                 print(f"Perlin噪声生成时出错: {e}")
                 # Fallback will be handled after checks


        if algorithm_type == 'gradient' or (not algorithm_executed and algorithm_type == 'gradient'): # Run if selected OR as fallback
             try:
                 # More complex multi-color gradient influenced by randomness
                 color1 = (255, 100, 50) # Warm color
                 color2 = (50, 100, 255) # Cool color
                 color3 = (100, 255, 50) # Greenish color

                 for y in range(height):
                    for x in range(width):
                        fx = x / width
                        fy = y / height

                        # Blend factor incorporating diagonal and randomness
                        blend_factor = (fx + fy) / 2.0
                        # Add random wobble to the blend factor based on randomness
                        if randomness > 0:
                             blend_factor += random.uniform(-randomness * 0.2, randomness * 0.2)
                             blend_factor = max(0.0, min(1.0, blend_factor))

                        # Simple 3-color blend based on blend_factor
                        if blend_factor < 0.5:
                            interp = blend_factor * 2
                            r = int(color1[0] * (1-interp) + color2[0] * interp)
                            g = int(color1[1] * (1-interp) + color2[1] * interp)
                            b = int(color1[2] * (1-interp) + color2[2] * interp)
                        else:
                             interp = (blend_factor - 0.5) * 2
                             r = int(color2[0] * (1-interp) + color3[0] * interp)
                             g = int(color2[1] * (1-interp) + color3[1] * interp)
                             b = int(color2[2] * (1-interp) + color3[2] * interp)

                        pixels[x, y] = (r, g, b)
                 algorithm_executed = True
             except Exception as e:
                 print(f"渐变生成时出错: {e}")


        elif algorithm_type == 'random_rectangles' and not algorithm_executed:
            try:
                # Generate random colored rectangles with sizes and colors influenced by randomness
                min_rect_size = 5
                max_rect_size = max(min_rect_size, int(min(width, height) * (1.0 - randomness * 0.8)))
                num_rects = int((width * height / 200) * (1 + randomness * 6))

                for _ in range(num_rects):
                    w = random.randint(min_rect_size, max(min_rect_size, max_rect_size))
                    h = random.randint(min_rect_size, max(min_rect_size, max_rect_size))
                    x = random.randint(0, width - w) if width - w >= 0 else 0
                    y = random.randint(0, height - h) if height - h >= 0 else 0

                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if randomness < 0.3: # Less variation for low randomness, lean towards grayscale
                         avg_color = random.randint(0, 255)
                         color = (avg_color, avg_color, avg_color)
                    elif randomness < 0.6: # Moderate variation, less saturated
                        gray = int(0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2]) # Luminance
                        color = tuple(int(gray * (1-randomness*0.5) + c * randomness*0.5) for c in color)


                    for sy in range(h):
                        for sx in range(w):
                            if x+sx < width and y+sy < height:
                                pixels[x+sx, y+sy] = color
                algorithm_executed = True
            except Exception as e:
                 print(f"随机矩形生成时出错: {e}")


        elif algorithm_type == 'cellular_automata' and not algorithm_executed:
             try:
                 # A slightly more complex CA rule with smoothing and randomness influence
                 grid = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8) # Start with random colors
                 iterations = max(0, iterations) # Ensure iterations is non-negative
                 iterations_with_rand = iterations + int(randomness * 30) # More iterations with randomness

                 # Define a simple convolution kernel for smoothing/averaging neighbors
                 kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) # Sum of 8 neighbors
                 kernel_sum = np.sum(kernel)

                 for i in range(iterations_with_rand):
                     new_grid = np.copy(grid)
                     # Pad grid to handle edges for convolution (mode 'wrap' for tiling patterns)
                     grid_padded = np.pad(grid, ((1,1),(1,1),(0,0)), mode='wrap') # Using 'wrap'

                     for y in range(height):
                         for x in range(width):
                             # Get neighbor region from padded grid
                             neighbors_region = grid_padded[y:y+3, x:x+3, :]

                             # Calculate weighted sum of neighbor colors (using kernel)
                             # Add a small epsilon to avoid division by zero if kernel_sum somehow becomes 0
                             neighbor_sum = np.sum(neighbors_region * kernel[:,:,None], axis=(0, 1))
                             # Handle potential division by zero if kernel_sum is 0
                             avg_neighbor_color = (neighbor_sum // (kernel_sum if kernel_sum > 0 else 1)).astype(np.uint8)


                             # Rule: Blend average neighbor color with current color, influenced by randomness and iteration
                             # Blend more towards neighbor average in later iterations and with higher randomness
                             # Ensure blend_factor is within 0-1
                             # Avoid division by zero if iterations_with_rand is 0
                             iteration_blend = ((i + 1) / iterations_with_rand) if iterations_with_rand > 0 else 0.0
                             blend_factor = min(1.0, (0.2 + randomness * 0.6) * iteration_blend)


                             new_color = (
                                 int(avg_neighbor_color[0] * blend_factor + grid[y, x][0] * (1 - blend_factor)),
                                 int(avg_neighbor_color[1] * blend_factor + grid[y, x][1] * (1 - blend_factor)),
                                 int(avg_neighbor_color[2] * blend_factor + grid[y, x][2] * (1 - blend_factor))
                             )

                             # Add small random perturbation based on randomness
                             if randomness > 0:
                                  perturb_amount = int(randomness * 30)
                                  new_color = tuple(
                                       max(0, min(255, c + random.randint(-perturb_amount, perturb_amount)))
                                       for c in new_color
                                  )

                             new_grid[y, x] = new_color
                     grid = new_grid # Update grid for next iteration

                 for y in range(height):
                     for x in range(width):
                         pixels[x, y] = tuple(grid[y, x])
                 algorithm_executed = True
             except Exception as e:
                  print(f"细胞自动机生成时出错: {e}")


        elif algorithm_type == 'voronoi' and not algorithm_executed:
             try:
                 # Voronoi diagram generation
                 num_sites = max(2, int(num_points * (1 + randomness * 3)))
                 sites: List[Tuple[int, int, Tuple[int, int, int]]] = []
                 for _ in range(num_sites):
                      sx = random.randint(0, width - 1)
                      sy = random.randint(0, height - 1)
                      # Color variation influenced by randomness
                      color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                      if randomness < 0.3: # Less variation for low randomness, lean towards grayscale
                           avg_color = random.randint(0, 255)
                           color = (avg_color, avg_color, avg_color)
                      elif randomness < 0.6: # Moderate variation, less saturated
                          gray = int(0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2])
                          color = tuple(int(gray * (1-randomness*0.5) + c * randomness*0.5) for c in color)


                      sites.append((sx, sy, color))

                 # Determine color for each pixel based on the nearest site
                 # Optimized: Using numpy broadcasting for distance calculation
                 sites_coords = np.array([(s[0], s[1]) for s in sites])
                 sites_colors = [s[2] for s in sites]

                 # Create a grid of pixel coordinates
                 x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
                 pixel_coords = np.stack([x_coords, y_coords], axis=-1) # Shape (height, width, 2)

                 # Reshape sites_coords for broadcasting
                 sites_coords_reshaped = sites_coords[None, None, :, :] # Shape (1, 1, num_sites, 2)

                 # Calculate squared distances from each pixel to all sites
                 # (height, width, 1, 2) - (1, 1, num_sites, 2) -> broadcasting -> (height, width, num_sites, 2)
                 # Sum over the last axis (coords) -> (height, width, num_sites)
                 distances_sq = np.sum((pixel_coords[:, :, None, :] - sites_coords_reshaped)**2, axis=-1)

                 # Find the index of the nearest site for each pixel
                 nearest_site_indices = np.argmin(distances_sq, axis=-1) # Shape (height, width)

                 # Assign colors based on the nearest site index
                 for y in range(height):
                      for x in range(width):
                           nearest_index = nearest_site_indices[y, x]
                           pixels[x, y] = sites_colors[nearest_index]
                 algorithm_executed = True
             except Exception as e:
                 print(f"Voronoi生成时出错: {e}")


        # Fallback to basic noise if no algorithm executed successfully
        if not algorithm_executed:
             # print(f"警告: 算法 '{algorithm_type}' 未知或生成失败. 将生成基础噪声.") # GUI handles status
             # Ensure algorithm_type is set to basic_noise if it wasn't originally, for status update
             algorithm_type = 'basic_noise'
             for y in range(height):
                 for x in range(width):
                     pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


        # print(f"算法 '{algorithm_type}' 生成完成.") # GUI handles status update
        return img


    # --- 方式二：反馈学习 ---

    def record_feedback(self, keyword: str, generation_params: Dict[str, Any], rating: str = "liked"):
        """记录生成图片的参数和用户反馈 (关键词/描述)。"""
        if not keyword:
             # print("错误: 反馈关键词不能为空.") # GUI handles error message
             return

        if keyword not in self.feedback_data:
            self.feedback_data[keyword] = []

        # Store parameters and rating
        feedback_entry = generation_params.copy()
        feedback_entry["rating"] = rating
        # Ensure no non-serializable objects are stored
        if 'image' in feedback_entry:
             del feedback_entry['image']

        self.feedback_data[keyword].append(feedback_entry)

        # print(f"已记录关键词 '{keyword}', 评级 '{rating}' 的反馈.") # GUI handles status update
        self.save_all_data()

    def get_generation_params(self, keyword: str) -> Dict[str, Any]:
        """
        根据关键词获取历史反馈中偏好的参数。
        策略：对所有 'liked' 的参数，数值参数取平均，离散参数取最常见值。
        """
        if keyword not in self.feedback_data:
             return {}

        liked_entries = [entry for entry in self.feedback_data[keyword] if entry.get("rating") == "liked"]
        if not liked_entries:
             return {}

        # --- Averaging/Selecting Logic ---
        averaged_params: Dict[str, Any] = {}
        param_values: Dict[str, List[Any]] = {}

        # Collect all values for each parameter from liked entries
        for entry in liked_entries:
             for key, value in entry.items():
                  if key != 'rating':
                       if key not in param_values:
                            param_values[key] = []
                       param_values[key].append(value)

        # Process collected values
        for key, values in param_values.items():
             # Try to average numerical parameters
             try:
                  # Attempt conversion to numpy array of floats/ints
                  # Filter out None values or types that cannot be converted
                  clean_values = [v for v in values if isinstance(v, (int, float))]
                  if clean_values:
                      num_values = np.array(clean_values, dtype=float)
                      # Check if the conversion was successful for at least some values
                      if np.issubdtype(num_values.dtype, np.number):
                            averaged_params[key] = np.mean(num_values).item() # .item() converts numpy scalar to Python scalar
                            continue # Move to next parameter

              # Handle tuple types specifically, like image_size
             except (ValueError, TypeError):
                  if key == 'image_size':
                      # Collect only valid tuple sizes
                      size_values = [v for v in values if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(i, (int, float)) for i in v)]
                      if size_values:
                          # Average width and height separately
                          avg_width = int(np.mean([s[0] for s in size_values]))
                          avg_height = int(np.mean([s[1] for s in size_values]))
                          averaged_params[key] = (max(1, avg_width), max(1, avg_height)) # Ensure size > 0
                          continue


             # Handle discrete parameters (like algorithm_type, seed if not averaged) - choose the most common
             from collections import Counter
             # Filter out None values for Counter
             valid_values = [v for v in values if v is not None]
             if valid_values:
                 most_common_value = Counter(valid_values).most_common(1)[0][0]
                 averaged_params[key] = most_common_value

        # Special handling for seed: if multiple seeds were liked, averaging is meaningless.
        # Use the most common seed if one exists.
        if 'seed' in param_values and len([v for v in param_values['seed'] if isinstance(v, int)]) > 0:
             valid_seeds = [v for v in param_values['seed'] if isinstance(v, int)]
             if valid_seeds:
                 seed_counts = Counter(valid_seeds)
                 averaged_params['seed'] = seed_counts.most_common(1)[0][0]
             else:
                 # Fallback to random seed if no valid int seeds found
                 # This case should be covered by the outer if, but defensive
                 del averaged_params['seed'] # Indicate no usable seed from feedback
                 print("警告: 反馈中没有有效整数种子，将依赖默认随机种子.")


        # print(f"已为关键词 '{keyword}' 获取并平均偏好参数.") # GUI handles status update
        return averaged_params

    # --- 主生成方法 ---

    def generate_image(self, mode: str, keyword: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]]]:
        """
        根据选择的模式和参数生成图片。
        返回生成的图片对象和实际使用的参数字典，或 None, None (如果失败)。
        这个方法也会在内部处理参数的验证和类型转换。
        """
        if params is None:
            params = {}

        # 1. Define default parameters
        default_algo_type = 'perlin' if PERLIN_AVAILABLE else 'gradient'
        default_params: Dict[str, Any] = {
            'image_size': (128, 128),
            'randomness': 0.5,
            'seed': random.randint(0, 10000), # Default: new random seed each time
            'scale': 100.0,
            'octaves': 6,
            'learned_feature_strength': 1.0,
             'algorithm_type': default_algo_type,
             'num_points': 50,
             'iterations': 10
        }

        # 2. Get feedback parameters (if applicable)
        feedback_params: Dict[str, Any] = {}
        if mode == 'generated' and keyword:
             feedback_params = self.get_generation_params(keyword)

        # 3. Merge parameters: default <- feedback <- user input
        used_params = default_params.copy()
        used_params.update(feedback_params) # Feedback parameters override defaults
        used_params.update(params) # User input parameters override everything

        # --- Post-merge validation and type casting ---
        # Ensure all parameters have correct types and valid ranges before passing to algorithm
        validated_params: Dict[str, Any] = {}

        try:
            # image_size: tuple of two positive ints
            img_size = used_params.get('image_size')
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                 w, h = int(float(str(img_size[0]))), int(float(str(img_size[1]))) # Robust conversion
                 if w > 0 and h > 0:
                      validated_params['image_size'] = (w, h)
                 else:
                     raise ValueError("尺寸必须是正数.")
            else:
                 raise ValueError("图片尺寸格式错误.")

            # randomness: float 0.0-1.0
            rand_val = used_params.get('randomness')
            if isinstance(rand_val, (int, float)):
                validated_params['randomness'] = max(0.0, min(1.0, float(rand_val)))
            else:
                raise ValueError("随机性参数无效或格式错误.")

            # seed: int
            seed_val = used_params.get('seed')
            if isinstance(seed_val, (int, float)):
                 validated_params['seed'] = int(float(seed_val))
            else:
                 # If seed is invalid from params/feedback, use a new random default
                 validated_params['seed'] = random.randint(0, 10000)
                 if seed_val is not None: # Only warn if a value was provided but invalid
                      print(f"警告: 种子参数 '{seed_val}' 无效或格式错误，已使用随机种子.")


            # algorithm_type: str, validate against available, check if Perlin is available
            algo_type = used_params.get('algorithm_type')
            available_algos = ['gradient', 'random_rectangles', 'cellular_automata', 'voronoi']
            if PERLIN_AVAILABLE:
                 available_algos.insert(0, 'perlin')
            if not isinstance(algo_type, str) or algo_type not in available_algos:
                 print(f"警告: 算法 '{algo_type}' 无效. 已使用默认算法 '{default_algo_type}'.")
                 validated_params['algorithm_type'] = default_algo_type
            # Double check if perlin was selected but not available
            elif algo_type == 'perlin' and not PERLIN_AVAILABLE:
                 print(f"警告: Perlin算法不可用. 已使用默认算法 '{default_algo_type}'.")
                 validated_params['algorithm_type'] = default_algo_type
            else:
                 validated_params['algorithm_type'] = algo_type


            # learned_feature_strength: float 0.0-1.0 (only for learned mode)
            feat_strength = used_params.get('learned_feature_strength')
            if isinstance(feat_strength, (int, float)):
                 validated_params['learned_feature_strength'] = max(0.0, min(1.0, float(feat_strength)))
            else:
                 # Use default if invalid
                 validated_params['learned_feature_strength'] = default_params['learned_feature_strength']
                 if feat_strength is not None:
                      print(f"警告: 特征强度参数 '{feat_strength}' 无效或格式错误，已使用默认值.")


            # Algorithm specific parameters (only include if relevant algorithm is selected or they are valid)
            # Scale (float > 0 for Perlin)
            scale_val = used_params.get('scale')
            if isinstance(scale_val, (int, float)):
                 converted_scale = float(scale_val)
                 if converted_scale > 0:
                     validated_params['scale'] = converted_scale
                 else:
                      raise ValueError("缩放参数必须大于0.")
            elif scale_val is not None: raise ValueError("缩放参数无效或格式错误.")
            # If scale is None or not provided, it's okay, algorithm will use its default


            # Octaves (int > 0 for Perlin)
            octaves_val = used_params.get('octaves')
            if isinstance(octaves_val, (int, float)):
                 converted_octaves = int(float(octaves_val))
                 if converted_octaves > 0:
                      validated_params['octaves'] = converted_octaves
                 else:
                      raise ValueError("层数参数必须大于0.")
            elif octaves_val is not None: raise ValueError("层数参数无效或格式错误.")


            # num_points (int > 1 for Voronoi)
            points_val = used_params.get('num_points')
            if isinstance(points_val, (int, float)):
                 converted_points = int(float(points_val))
                 if converted_points > 1:
                     validated_params['num_points'] = converted_points
                 else:
                     raise ValueError("点数量必须大于1.")
            elif points_val is not None: raise ValueError("点数量参数无效或格式错误.")


            # iterations (int >= 0 for CA)
            iters_val = used_params.get('iterations')
            if isinstance(iters_val, (int, float)):
                 converted_iters = int(float(iters_val))
                 if converted_iters >= 0:
                      validated_params['iterations'] = converted_iters
                 else:
                     raise ValueError("迭代次数必须非负.")
            elif iters_val is not None: raise ValueError("迭代次数参数无效或格式错误.")


            # If a parameter was not successfully validated/converted from used_params,
            # ensure it's not included in validated_params unless it's a valid default.
            # This relies on the algorithm functions to handle missing optional parameters by using their own defaults.
            # We've already handled required ones (size, rand, seed, algo_type).

            final_params = default_params.copy() # Start with defaults
            final_params.update(validated_params) # Overlay validated params


        except ValueError as e:
             print(f"错误: 参数验证失败 - {e}")
             # print("图片生成失败.") # GUI handles status
             return None, None # Indicate parameter error
        except Exception as e:
             print(f"参数验证时发生未知错误: {e}")
             traceback.print_exc()
             # print("图片生成失败.") # GUI handles status
             return None, None # Indicate unexpected error


        generated_image = None
        success = False

        try:
            if mode == 'learned':
                if not keyword:
                    # print("错误: '学习模式' 需要关键词.") # Should be caught before here
                    return None, None # Error already handled by caller
                # print(f"尝试根据关键词 '{keyword}' 使用学习数据生成图片...") # GUI handles status update
                generated_image = self.generate_from_learned(keyword, final_params)

            elif mode == 'generated':
                # print(f"尝试使用结构化算法生成随机图片.") # GUI handles status update
                generated_image = self.generate_structured(final_params)

            else:
                # print("无效的模式指定. 请使用 'learned' 或 'generated'.") # Should not happen
                return None, None

            if generated_image:
                success = True

        except Exception as e:
            print(f"生成图片时发生错误: {e}")
            traceback.print_exc() # Print full traceback for debugging
            # GUI shows a general error message

        if success and generated_image:
            # print("图片成功生成.") # GUI handles status update
            self.current_image = generated_image # Store for saving/display
            self.last_generated_params = final_params # Store params for feedback
            self.last_generated_keyword = keyword # Store keyword used for generation
            self.last_generated_mode = mode
            return generated_image, final_params
        else:
            # print("图片生成失败.") # GUI handles status update
            self.current_image = None
            self.last_generated_params = None
            self.last_generated_keyword = None
            self.last_generated_mode = None
            return None, None


# --- GUI 界面 (Tkinter) ---

class ImageGeneratorGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("AI 图片生成器")
        master.geometry("800x700") # Set a default window size

        self.generator = AIGenerator()
        self.current_tk_image: Optional[ImageTk.PhotoImage] = None # To keep reference for Tkinter

        # --- Global Controls and Status (Moved up) ---
        self.frame_global_controls = ttk.Frame(self.master, padding="10")
        self.frame_global_controls.pack(pady=(5,0), padx=10, fill='x')

        ttk.Button(self.frame_global_controls, text="保存当前图片...", command=self._save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.frame_global_controls, text="查看关键词...", command=self._view_keywords).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.frame_global_controls, text="保存所有数据并退出", command=self.master_quit).pack(side=tk.RIGHT, padx=5)

        self.status_label = ttk.Label(self.master, text="状态信息:") # Status label is now created early
        self.status_label.pack(pady=(0, 5), padx=10, fill='x')


        # --- Image Display Area ---
        # Use a Canvas instead of Label for potentially better image handling and future drawing
        # However, Label is simpler for just displaying PhotoImage. Sticking with Label for now.
        self.image_display_frame = ttk.Frame(self.master, borderwidth=2, relief="groove")
        self.image_display_frame.pack(pady=5, padx=10, fill='both', expand=True)
        # Use a label *inside* the frame to center the image
        self.image_display_label = tk.Label(self.image_display_frame) # Label to display the image
        self.image_display_label.pack(expand=True) # Center the label within its frame


        # --- Notebook (Tabs) for Modes ---
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=10, padx=10, fill='both', expand=False) # Don't expand notebook fully

        # Mode 1: Learn & Generate Tab
        self.frame_mode1 = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.frame_mode1, text="模式 1: 学习与生成")
        self._create_mode1_widgets(self.frame_mode1)

        # Mode 2: Generated & Feedback Tab
        self.frame_mode2 = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.frame_mode2, text="模式 2: 算法生成与反馈")
        self._create_mode2_widgets(self.frame_mode2)


        # Initial message
        self.update_status("程序已启动. 数据已加载.")
        if not PERLIN_AVAILABLE:
             self.update_status("警告: 未安装 'noise' 库, Perlin 噪声算法不可用.", is_warning=True)


    def _create_widgets(self):
        """Deprecated - Widgets are created directly in __init__ or dedicated methods."""
        pass # This method is no longer called directly from __init__

    def _create_mode1_widgets(self, parent_frame):
        """Widgets for Mode 1 tab."""
        learn_frame = ttk.LabelFrame(parent_frame, text="从图片学习", padding="10")
        learn_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(learn_frame, text="图片文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.learn_image_path_entry = ttk.Entry(learn_frame, width=40)
        self.learn_image_path_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(learn_frame, text="浏览...", command=self._browse_image).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(learn_frame, text="关联关键词:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.learn_keyword_entry = ttk.Entry(learn_frame, width=40)
        self.learn_keyword_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(learn_frame, text="开始学习", command=self._learn_image).grid(row=1, column=2, padx=5, pady=5)

        learn_frame.columnconfigure(1, weight=1)


        generate_frame = ttk.LabelFrame(parent_frame, text="从学习数据生成", padding="10")
        generate_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(generate_frame, text="生成关键词:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.generate_learned_keyword_entry = ttk.Entry(generate_frame, width=40)
        self.generate_learned_keyword_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(generate_frame, text="尺寸 (WxH):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.generate_learned_size_entry = ttk.Entry(generate_frame, width=20)
        self.generate_learned_size_entry.insert(0, "128x128")
        self.generate_learned_size_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(generate_frame, text="随机性 (0-1):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.generate_learned_randomness_scale = ttk.Scale(generate_frame, from_=0.0, to=1.0, orient="horizontal", length=200)
        self.generate_learned_randomness_scale.set(0.5)
        self.generate_learned_randomness_scale.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.generate_learned_randomness_value_label = ttk.Label(generate_frame, text="0.50")
        self.generate_learned_randomness_scale.bind("<Motion>", lambda e: self._update_scale_label(self.generate_learned_randomness_scale, self.generate_learned_randomness_value_label, resolution=2))
        self.generate_learned_randomness_value_label.grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(generate_frame, text="特征强度 (0-1):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.generate_learned_feature_strength_scale = ttk.Scale(generate_frame, from_=0.0, to=1.0, orient="horizontal", length=200)
        self.generate_learned_feature_strength_scale.set(1.0)
        self.generate_learned_feature_strength_scale.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.generate_learned_feature_strength_value_label = ttk.Label(generate_frame, text="1.00")
        self.generate_learned_feature_strength_scale.bind("<Motion>", lambda e: self._update_scale_label(self.generate_learned_feature_strength_scale, self.generate_learned_feature_strength_value_label, resolution=2))
        self.generate_learned_feature_strength_value_label.grid(row=3, column=2, padx=5, pady=5)


        ttk.Button(generate_frame, text="生成图片", command=self._generate_learned).grid(row=4, column=1, sticky="e", padx=5, pady=10)

        generate_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(0, weight=1)


    def _create_mode2_widgets(self, parent_frame):
        """Widgets for Mode 2 tab."""
        generate_frame = ttk.LabelFrame(parent_frame, text="算法生成", padding="10")
        generate_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(generate_frame, text="尺寸 (WxH):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.generate_random_size_entry = ttk.Entry(generate_frame, width=20)
        self.generate_random_size_entry.insert(0, "128x128")
        self.generate_random_size_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(generate_frame, text="随机性 (0-1):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.generate_random_randomness_scale = ttk.Scale(generate_frame, from_=0.0, to=1.0, orient="horizontal", length=200)
        self.generate_random_randomness_scale.set(0.5)
        self.generate_random_randomness_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.generate_random_randomness_value_label = ttk.Label(generate_frame, text="0.50")
        self.generate_random_randomness_scale.bind("<Motion>", lambda e: self._update_scale_label(self.generate_random_randomness_scale, self.generate_random_randomness_value_label, resolution=2))
        self.generate_random_randomness_value_label.grid(row=1, column=2, padx=5, pady=5)


        ttk.Label(generate_frame, text="算法类型:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        available_algos = ['gradient', 'random_rectangles', 'cellular_automata', 'voronoi']
        if PERLIN_AVAILABLE:
            available_algos.insert(0, 'perlin')
        self.generate_random_algorithm_combobox = ttk.Combobox(generate_frame, values=available_algos, state="readonly")
        self.generate_random_algorithm_combobox.set(available_algos[0] if available_algos else "") # Set default
        self.generate_random_algorithm_combobox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        # Bind algorithm change to potentially show/hide algorithm-specific controls
        self.generate_random_algorithm_combobox.bind("<<ComboboxSelected>>", self._update_algo_params_visibility)

        # Algorithm specific parameters frame container
        self.algo_params_container_frame = ttk.Frame(generate_frame)
        self.algo_params_container_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
        # We will grid specific frames within this container

        self._create_algo_specific_params_widgets(self.algo_params_container_frame) # Create all, manage visibility later


        ttk.Label(generate_frame, text="影响关键词 (可选):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.generate_random_keyword_entry = ttk.Entry(generate_frame, width=40)
        self.generate_random_keyword_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(generate_frame, text="生成图片", command=self._generate_random).grid(row=5, column=1, sticky="e", padx=5, pady=10)

        generate_frame.columnconfigure(1, weight=1)


        feedback_frame = ttk.LabelFrame(parent_frame, text="提供反馈", padding="10")
        feedback_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(feedback_frame, text="描述/关键词:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.feedback_keyword_entry = ttk.Entry(feedback_frame, width=40)
        self.feedback_keyword_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(feedback_frame, text="喜欢这张图", command=lambda: self._send_feedback("liked")).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(feedback_frame, text="不喜欢这张图", command=lambda: self._send_feedback("disliked")).grid(row=0, column=3, padx=5, pady=5)

        feedback_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(0, weight=1)

        # Call once initially to set correct visibility
        self._update_algo_params_visibility()


    def _create_algo_specific_params_widgets(self, parent_frame):
        """Create widgets for algorithm-specific parameters within the container frame."""
        # Perlin/Simplex parameters frame
        self.perlin_params_frame = ttk.Frame(parent_frame)
        ttk.Label(self.perlin_params_frame, text="噪声缩放 (Scale):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.perlin_scale_scale = ttk.Scale(self.perlin_params_frame, from_=10.0, to=500.0, orient="horizontal", length=150)
        self.perlin_scale_scale.set(100.0)
        self.perlin_scale_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.perlin_scale_value_label = ttk.Label(self.perlin_params_frame, text="100.0")
        self.perlin_scale_scale.bind("<Motion>", lambda e: self._update_scale_label(self.perlin_scale_scale, self.perlin_scale_value_label, resolution=1))
        self.perlin_scale_value_label.grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(self.perlin_params_frame, text="噪声层数 (Octaves):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.perlin_octaves_scale = ttk.Scale(self.perlin_params_frame, from_=1, to=10, orient="horizontal", length=150)
        self.perlin_octaves_scale.set(6)
        self.perlin_octaves_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.perlin_octaves_value_label = ttk.Label(self.perlin_params_frame, text="6")
        self.perlin_octaves_scale.bind("<Motion>", lambda e: self._update_scale_label(self.perlin_octaves_scale, self.perlin_octaves_value_label, resolution=0)) # Display as integer
        self.perlin_octaves_value_label.grid(row=1, column=2, padx=5, pady=2)

        self.perlin_params_frame.columnconfigure(1, weight=1)


        # Voronoi parameters frame
        self.voronoi_params_frame = ttk.Frame(parent_frame)
        ttk.Label(self.voronoi_params_frame, text="点数量 (Num Points):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.voronoi_num_points_scale = ttk.Scale(self.voronoi_params_frame, from_=10, to=500, orient="horizontal", length=150)
        self.voronoi_num_points_scale.set(50)
        self.voronoi_num_points_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.voronoi_num_points_value_label = ttk.Label(self.voronoi_params_frame, text="50")
        self.voronoi_num_points_scale.bind("<Motion>", lambda e: self._update_scale_label(self.voronoi_num_points_scale, self.voronoi_num_points_value_label, resolution=0)) # Display as integer
        self.voronoi_num_points_value_label.grid(row=0, column=2, padx=5, pady=2)
        self.voronoi_params_frame.columnconfigure(1, weight=1)


        # Cellular Automata parameters frame
        self.ca_params_frame = ttk.Frame(parent_frame)
        ttk.Label(self.ca_params_frame, text="迭代次数 (Iterations):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ca_iterations_scale = ttk.Scale(self.ca_params_frame, from_=1, to=50, orient="horizontal", length=150)
        self.ca_iterations_scale.set(10)
        self.ca_iterations_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.ca_iterations_value_label = ttk.Label(self.ca_params_frame, text="10")
        self.ca_iterations_scale.bind("<Motion>", lambda e: self._update_scale_label(self.ca_iterations_scale, self.ca_iterations_value_label, resolution=0)) # Display as integer
        self.ca_iterations_value_label.grid(row=0, column=2, padx=5, pady=2)
        self.ca_params_frame.columnconfigure(1, weight=1)


        # Seed parameter frame (applies to multiple algorithms)
        self.seed_frame = ttk.Frame(parent_frame)
        ttk.Label(self.seed_frame, text="种子 (Seed):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.seed_entry = ttk.Entry(self.seed_frame, width=15)
        self.seed_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(self.seed_frame, text="随机种子", command=lambda: self.seed_entry.delete(0, tk.END) or self.seed_entry.insert(0, str(random.randint(0, 10000)))).grid(row=0, column=2, padx=5, pady=2)
        self.seed_frame.columnconfigure(1, weight=1)


    def _update_algo_params_visibility(self, event=None):
        """Shows/hides algorithm-specific parameter frames within the container."""
        selected_algo = self.generate_random_algorithm_combobox.get()

        # Ungrid all algo specific frames from the container
        for frame in [self.perlin_params_frame, self.voronoi_params_frame, self.ca_params_frame, self.seed_frame]:
            frame.grid_forget()

        # Grid relevant frames within the container frame (self.algo_params_container_frame)
        row_idx = 0
        if selected_algo in ['perlin', 'simplex'] and PERLIN_AVAILABLE:
            self.perlin_params_frame.grid(row=row_idx, column=0, sticky="ew")
            row_idx += 1
        elif selected_algo == 'voronoi':
            self.voronoi_params_frame.grid(row=row_idx, column=0, sticky="ew")
            row_idx += 1
        elif selected_algo == 'cellular_automata':
            self.ca_params_frame.grid(row=row_idx, column=0, sticky="ew")
            row_idx += 1
        # Note: Gradient and random_rectangles don't have specific scales/octaves etc. besides randomness and seed

        # Seed is common to most algorithms, grid it after algo-specific ones if they exist
        # Or just grid it always if it's at the bottom of the section
        # Let's always grid seed last in the container
        self.seed_frame.grid(row=row_idx, column=0, sticky="ew")


    def _update_scale_label(self, scale_widget: ttk.Scale, label_widget: ttk.Label, resolution: int = 2):
        """Updates a label with the current value of a scale widget, rounded by resolution."""
        value = scale_widget.get()
        if resolution == 0: # Display as integer
            label_widget.config(text=f"{round(value)}")
        else:
            label_widget.config(text=f"{value:.{resolution}f}")


    def update_status(self, message: str, is_warning: bool = False):
        """Updates the status label with a message."""
        # Ensure status_label exists before trying to configure it
        if hasattr(self, 'status_label') and self.status_label:
             self.status_label.config(text=f"状态信息: {message}", foreground="red" if is_warning else "black")
        print(message) # Also print to console for debugging/logging

    def _browse_image(self):
        """Opens a file dialog to select an image."""
        filepath = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=(("图片文件", "*.png *.jpg *.jpeg *.gif *.bmp"), ("所有文件", "*.*"))
        )
        if filepath:
            self.learn_image_path_entry.delete(0, tk.END)
            self.learn_image_path_entry.insert(0, filepath)
            self.update_status(f"已选择文件: {os.path.basename(filepath)}")

    def _get_params_from_gui(self, mode: str) -> Dict[str, Any]:
        """从 GUI 控件读取参数，并进行初步验证和类型转换."""
        params: Dict[str, Any] = {}
        try:
            if mode == 'learned':
                size_str = self.generate_learned_size_entry.get()
                randomness = self.generate_learned_randomness_scale.get()
                feature_strength = self.generate_learned_feature_strength_scale.get()

                # Robust size parsing
                if 'x' not in size_str: raise ValueError("尺寸格式应为 宽x高 (如 128x128).")
                w_str, h_str = size_str.strip().split('x')
                w, h = int(w_str), int(h_str)
                if w <= 0 or h <= 0: raise ValueError("图片尺寸必须大于0.")
                params['image_size'] = (w, h)

                # Randomness and feature strength are float scales, direct get is fine
                params['randomness'] = randomness
                params['learned_feature_strength'] = feature_strength

            elif mode == 'generated':
                size_str = self.generate_random_size_entry.get()
                randomness = self.generate_random_randomness_scale.get()
                algorithm_type = self.generate_random_algorithm_combobox.get()
                seed_str = self.seed_entry.get()

                # Robust size parsing
                if 'x' not in size_str: raise ValueError("尺寸格式应为 宽x高 (如 128x128).")
                w_str, h_str = size_str.strip().split('x')
                w, h = int(w_str), int(h_str)
                if w <= 0 or h <= 0: raise ValueError("图片尺寸必须大于0.")
                params['image_size'] = (w, h)

                # Randomness is float scale, direct get is fine
                params['randomness'] = randomness
                params['algorithm_type'] = algorithm_type

                # Seed is optional integer
                if seed_str.strip():
                    params['seed'] = int(seed_str.strip())
                # If empty, backend uses random seed default

                # Algorithm specific params (get from relevant scales and entry)
                if algorithm_type in ['perlin', 'simplex'] and PERLIN_AVAILABLE:
                    params['scale'] = self.perlin_scale_scale.get()
                    params['octaves'] = int(self.perlin_octaves_scale.get()) # Cast to int
                elif algorithm_type == 'voronoi':
                    params['num_points'] = int(self.voronoi_num_points_scale.get()) # Cast to int
                elif algorithm_type == 'cellular_automata':
                     params['iterations'] = int(self.ca_iterations_scale.get()) # Cast to int

            # Return dictionary even if validation failed (errorbox is shown), backend does final validation
            return params

        except ValueError as e:
            messagebox.showerror("参数错误", f"无效的参数输入: {e}")
            self.update_status("参数输入错误.", is_warning=True)
            return {} # Return empty dict on error
        except Exception as e:
             messagebox.showerror("参数错误", f"获取参数时发生未知错误: {e}\n{traceback.format_exc()}")
             self.update_status("获取参数时出错.", is_warning=True)
             return {} # Return empty dict on error


    def _display_image(self, img: Optional[Image.Image]):
        """Displays a Pillow image in the Tkinter label."""

        # --- FIX: Explicitly clear the old image reference ---
        self.image_display_label.config(image='')
        self.current_tk_image = None
        # --- End FIX ---


        if img is None:
            self.image_display_label.config(text="图片生成失败或无图片可显示") # Show fallback text
            return
        else:
             self.image_display_label.config(text="") # Clear fallback text


        # Resize image to fit the display area while maintaining aspect ratio
        # Need to update_idletasks to get the actual current size of the frame
        self.master.update_idletasks()
        max_display_width = self.image_display_frame.winfo_width()
        max_display_height = self.image_display_frame.winfo_height()

        if max_display_width <= 1 or max_display_height <= 1:
             # Fallback if frame size is not yet determined
             max_display_width = 500
             max_display_height = 500
             # print("Warning: Frame size not ready, using default max display size.") # Optional warning


        original_width, original_height = img.size
        if original_width > max_display_width or original_height > max_display_height:
             scale = min(max_display_width / original_width, max_display_height / original_height)
             new_width = int(original_width * scale)
             new_height = int(original_height * scale)
             # Use BICUBIC for better quality when scaling down
             # Ensure new dimensions are at least 1x1 pixel
             new_width = max(1, new_width)
             new_height = max(1, new_height)
             try:
                img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
             except ValueError as e:
                 print(f"Resize计算出错 (新尺寸 {new_width}x{new_height}): {e}")
                 img_resized = img # Use original image if resize fails
             except Exception as e:
                  print(f"图片缩放时发生未知错误: {e}")
                  img_resized = img # Use original image if resize fails
        else:
             img_resized = img # No resize needed


        try:
            self.current_tk_image = ImageTk.PhotoImage(img_resized)
            self.image_display_label.config(image=self.current_tk_image)
        except Exception as e:
             messagebox.showerror("显示错误", f"显示图片时出错: {e}\n{traceback.format_exc()}")
             self.update_status("显示图片失败.", is_warning=True)
             self.image_display_label.config(text="图片生成成功，但显示失败") # Indicate generation was okay


    def _learn_image(self):
        """Handles the 'Learn from Image' button click."""
        filepath = self.learn_image_path_entry.get().strip()
        keyword = self.learn_keyword_entry.get().strip()

        if not filepath:
            messagebox.showwarning("输入错误", "请选择图片文件.")
            self.update_status("请选择图片文件.", is_warning=True)
            return
        if not keyword:
            messagebox.showwarning("输入错误", "请输入关键词.")
            self.update_status("请输入关键词.", is_warning=True)
            return
        if not os.path.exists(filepath):
             messagebox.showwarning("文件错误", f"文件不存在: {filepath}")
             self.update_status(f"文件不存在: {filepath}", is_warning=True)
             return

        self.update_status(f"正在从图片 {os.path.basename(filepath)} 学习关键词 '{keyword}'...")
        try:
            success = self.generator.learn_from_image(filepath, keyword)
            if success:
                self.update_status(f"学习成功！关键词 '{keyword}' 已更新.")
            else:
                # learn_from_image prints specific errors, update_status just reflects general state
                self.update_status(f"学习失败 (请检查控制台输出).", is_warning=True)
        except Exception as e:
            messagebox.showerror("学习错误", f"学习时发生错误: {e}\n{traceback.format_exc()}")
            self.update_status("学习过程出错.", is_warning=True)


    def _generate_learned(self):
        """Handles the 'Generate from Learned' button click."""
        keyword = self.generate_learned_keyword_entry.get().strip()
        if not keyword:
            messagebox.showwarning("输入错误", "请输入生成关键词.")
            self.update_status("请输入生成关键词.", is_warning=True)
            return

        params = self._get_params_from_gui(mode='learned')
        if not params: # _get_params_from_gui shows error message
            self.update_status("参数获取失败.", is_warning=True)
            return

        self.update_status(f"正在根据关键词 '{keyword}' 生成图片...")
        try:
            # --- Potential Improvement: Use threading for long generations ---
            # For simple cases or small images, direct call is fine.
            # For larger images, consider:
            # thread = threading.Thread(target=self._run_generation_threaded, args=('learned', keyword, params))
            # thread.start()
            # self.update_status("生成任务已开始...") # Or similar

            generated_image, used_params = self.generator.generate_image(mode='learned', keyword=keyword, params=params)

            if generated_image:
                self._display_image(generated_image)
                self.update_status("图片生成成功！")
            else:
                 self._display_image(None) # Clear display and show error text
                 # generate_image prints specific errors, update_status just reflects general state
                 self.update_status("图片生成失败 (请检查控制台输出).", is_warning=True)
        except Exception as e:
            messagebox.showerror("生成错误", f"生成图片时发生错误: {e}\n{traceback.format_exc()}")
            self.update_status("生成过程出错.", is_warning=True)
            self._display_image(None) # Clear display


    def _generate_random(self):
        """Handles the 'Generate Random Structured' button click."""
        keyword = self.generate_random_keyword_entry.get().strip() # Optional keyword for influence

        params = self._get_params_from_gui(mode='generated')
        if not params: # _get_params_from_gui shows error message
             self.update_status("参数获取失败.", is_warning=True)
             return

        self.update_status(f"正在使用算法 '{params.get('algorithm_type', '未知')}' 生成图片...")
        try:
            # --- Potential Improvement: Use threading for long generations ---
            # thread = threading.Thread(target=self._run_generation_threaded, args=('generated', keyword if keyword else None, params))
            # thread.start()
            # self.update_status("生成任务已开始...")

            generated_image, used_params = self.generator.generate_image(mode='generated', keyword=keyword if keyword else None, params=params)

            if generated_image:
                self._display_image(generated_image)
                self.update_status("图片生成成功！")
            else:
                self._display_image(None) # Clear display and show error text
                # generate_image prints specific errors, update_status just reflects general state
                self.update_status("图片生成失败 (请检查控制台输出).", is_warning=True)
        except Exception as e:
            messagebox.showerror("生成错误", f"生成图片时发生错误: {e}\n{traceback.format_exc()}")
            self.update_status("生成过程出错.", is_warning=True)
            self._display_image(None) # Clear display


    # def _run_generation_threaded(self, mode, keyword, params):
    #     """Helper to run generation in a separate thread."""
    #     # This needs careful handling of updating GUI from the thread
    #     # (e.g., using master.after or a Queue)
    #     # Not fully implemented here as it adds complexity.
    #     pass # Placeholder


    def _save_image(self):
        """Handles the 'Save Current Image' button click."""
        if self.generator.current_image is None:
            messagebox.showwarning("保存失败", "没有可以保存的图片.")
            self.update_status("没有可以保存的图片.", is_warning=True)
            return

        filepath = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=(("PNG图片", "*.png"), ("JPEG图片", "*.jpg *.jpeg"), ("所有文件", "*.*"))
        )
        if filepath:
            try:
                self.generator.current_image.save(filepath)
                self.update_status(f"图片已保存为: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存图片时出错: {e}\n{traceback.format_exc()}")
                self.update_status(f"保存图片失败.", is_warning=True)


    def _send_feedback(self, rating: str):
        """Handles the feedback buttons."""
        if self.generator.last_generated_params is None:
            messagebox.showwarning("反馈失败", "没有最近生成的图片可以提供反馈.")
            self.update_status("没有最近生成的图片可提供反馈.", is_warning=True)
            return

        feedback_keyword = self.feedback_keyword_entry.get().strip()
        if not feedback_keyword:
            messagebox.showwarning("输入错误", "请输入反馈的描述/关键词.")
            self.update_status("请输入反馈的描述/关键词.", is_warning=True)
            return

        # Record feedback
        try:
            self.generator.record_feedback(feedback_keyword, self.generator.last_generated_params, rating)
            self.update_status(f"已提交反馈 '{feedback_keyword}' ({rating}). 感谢!")

            # Optionally clear feedback entry and last generated info after sending
            self.feedback_keyword_entry.delete(0, tk.END)
            self.generator.last_generated_params = None
            self.generator.last_generated_keyword = None
            self.generator.last_generated_mode = None
        except Exception as e:
            messagebox.showerror("反馈错误", f"提交反馈时出错: {e}\n{traceback.format_exc()}")
            self.update_status("提交反馈时出错.", is_warning=True)


    def _view_keywords(self):
        """Displays learned and feedback keywords in a new window."""
        keywords_window = tk.Toplevel(self.master)
        keywords_window.title("关键词信息")
        keywords_window.geometry("400x300")

        # Prevent window from being resized too small
        keywords_window.grid_columnconfigure(0, weight=1)
        keywords_window.grid_rowconfigure(0, weight=1)


        notebook = ttk.Notebook(keywords_window)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)


        # Learned Keywords Tab
        frame_learned = ttk.Frame(notebook, padding="10")
        notebook.add(frame_learned, text="学习关键词")
        frame_learned.grid_columnconfigure(0, weight=1)
        frame_learned.grid_rowconfigure(0, weight=1)

        learned_text = scrolledtext.ScrolledText(frame_learned, wrap=tk.WORD)
        learned_text.grid(row=0, column=0, sticky="nsew")
        learned_text.insert(tk.END, "已学习关键词:\n")
        if self.generator.learning_data:
            for keyword, data in self.generator.learning_data.items():
                learned_text.insert(tk.END, f"- {keyword} (来源图片: {len(data.get('source_images', []))})\n")
        else:
            learned_text.insert(tk.END, "无学习关键词.")
        learned_text.config(state=tk.DISABLED) # Make read-only


        # Feedback Keywords Tab
        frame_feedback = ttk.Frame(notebook, padding="10")
        notebook.add(frame_feedback, text="反馈关键词")
        frame_feedback.grid_columnconfigure(0, weight=1)
        frame_feedback.grid_rowconfigure(0, weight=1)

        feedback_text = scrolledtext.ScrolledText(frame_feedback, wrap=tk.WORD)
        feedback_text.grid(row=0, column=0, sticky="nsew")
        feedback_text.insert(tk.END, "反馈关键词:\n")
        if self.generator.feedback_data:
            for keyword, entries in self.generator.feedback_data.items():
                liked_count = sum(1 for entry in entries if entry.get('rating') == 'liked')
                disliked_count = len(entries) - liked_count
                feedback_text.insert(tk.END, f"- {keyword} (反馈总数: {len(entries)}, 喜欢: {liked_count}, 不喜欢: {disliked_count})\n")
        else:
            feedback_text.insert(tk.END, "无反馈关键词.")
        feedback_text.config(state=tk.DISABLED) # Make read-only


    def master_quit(self):
        """Saves data before closing."""
        self.update_status("正在保存数据...")
        self.generator.save_all_data()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorGUI(root)
    # Override the default close button behavior
    root.protocol("WM_DELETE_WINDOW", app.master_quit)
    root.mainloop()
