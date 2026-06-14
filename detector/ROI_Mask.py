import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import pydicom
import cv2
import gc
import time
from datetime import datetime
import os
import pickle
import json
import hashlib

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

key = random.PRNGKey(42)

def conv2d(x: jnp.ndarray, kernel: jnp.ndarray, stride: int = 1, padding: str = 'SAME') -> jnp.ndarray:
    if padding == 'SAME':
        pad_h = (kernel.shape[0] - 1) // 2
        pad_w = (kernel.shape[1] - 1) // 2
        x = jnp.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    return jax.lax.conv_general_dilated(
        x, kernel, window_strides=(stride, stride),
        padding='VALID', dimension_numbers=dimension_numbers
    )

def max_pool2d(x: jnp.ndarray, pool_size: int = 2, stride: int = 2) -> jnp.ndarray:
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, pool_size, pool_size, 1),
        window_strides=(1, stride, stride, 1),
        padding='VALID'
    )

def conv_transpose2d(x: jnp.ndarray, kernel: jnp.ndarray, stride: int = 2) -> jnp.ndarray:
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    return jax.lax.conv_transpose(
        x, kernel, strides=(stride, stride),
        padding='SAME', dimension_numbers=dimension_numbers
    )

def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-jnp.clip(x, -500, 500)))

def batch_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
               epsilon: float = 1e-5) -> jnp.ndarray:

    mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + epsilon)
    return gamma * x_norm + beta

def initialize_conv_weights(key: jax.random.PRNGKey,
                          shape: Tuple[int, int, int, int]) -> jnp.ndarray:
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = shape[0] * shape[1] * shape[3]
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return random.normal(key, shape) * std

def initialize_bn_params(shape: Tuple[int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.ones(shape)
    beta = jnp.zeros(shape)
    return gamma, beta

class UNet:
    def __init__(self, key: jax.random.PRNGKey):
        self.params = self._initialize_params(key)

    def _initialize_params(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        keys = random.split(key, 20)
        params = {}

        # Encoder
        params['conv1_1'] = initialize_conv_weights(keys[0], (3, 3, 1, 64))
        params['conv1_2'] = initialize_conv_weights(keys[1], (3, 3, 64, 64))
        params['bn1_1_gamma'], params['bn1_1_beta'] = initialize_bn_params((64,))
        params['bn1_2_gamma'], params['bn1_2_beta'] = initialize_bn_params((64,))

        params['conv2_1'] = initialize_conv_weights(keys[2], (3, 3, 64, 128))
        params['conv2_2'] = initialize_conv_weights(keys[3], (3, 3, 128, 128))
        params['bn2_1_gamma'], params['bn2_1_beta'] = initialize_bn_params((128,))
        params['bn2_2_gamma'], params['bn2_2_beta'] = initialize_bn_params((128,))

        params['conv3_1'] = initialize_conv_weights(keys[4], (3, 3, 128, 256))
        params['conv3_2'] = initialize_conv_weights(keys[5], (3, 3, 256, 256))
        params['bn3_1_gamma'], params['bn3_1_beta'] = initialize_bn_params((256,))
        params['bn3_2_gamma'], params['bn3_2_beta'] = initialize_bn_params((256,))

        params['conv4_1'] = initialize_conv_weights(keys[6], (3, 3, 256, 512))
        params['conv4_2'] = initialize_conv_weights(keys[7], (3, 3, 512, 512))
        params['bn4_1_gamma'], params['bn4_1_beta'] = initialize_bn_params((512,))
        params['bn4_2_gamma'], params['bn4_2_beta'] = initialize_bn_params((512,))

        # Bottleneck
        params['conv5_1'] = initialize_conv_weights(keys[8], (3, 3, 512, 1024))
        params['conv5_2'] = initialize_conv_weights(keys[9], (3, 3, 1024, 1024))
        params['bn5_1_gamma'], params['bn5_1_beta'] = initialize_bn_params((1024,))
        params['bn5_2_gamma'], params['bn5_2_beta'] = initialize_bn_params((1024,))

        # Decoder
        params['upconv4'] = initialize_conv_weights(keys[10], (2, 2, 1024, 512)) # Input 1024, Output 512
        params['conv6_1'] = initialize_conv_weights(keys[11], (3, 3, 1024, 512))
        params['conv6_2'] = initialize_conv_weights(keys[12], (3, 3, 512, 512))
        params['bn6_1_gamma'], params['bn6_1_beta'] = initialize_bn_params((512,))
        params['bn6_2_gamma'], params['bn6_2_beta'] = initialize_bn_params((512,))

        params['upconv3'] = initialize_conv_weights(keys[13], (2, 2, 512, 256))  # Input 512, Output 256
        params['conv7_1'] = initialize_conv_weights(keys[14], (3, 3, 512, 256))
        params['conv7_2'] = initialize_conv_weights(keys[15], (3, 3, 256, 256))
        params['bn7_1_gamma'], params['bn7_1_beta'] = initialize_bn_params((256,))
        params['bn7_2_gamma'], params['bn7_2_beta'] = initialize_bn_params((256,))

        params['upconv2'] = initialize_conv_weights(keys[16], (2, 2, 256, 128))  # Input 256, Output 128
        params['conv8_1'] = initialize_conv_weights(keys[17], (3, 3, 256, 128))
        params['conv8_2'] = initialize_conv_weights(keys[18], (3, 3, 128, 128))
        params['bn8_1_gamma'], params['bn8_1_beta'] = initialize_bn_params((128,))
        params['bn8_2_gamma'], params['bn8_2_beta'] = initialize_bn_params((128,))

        params['upconv1'] = initialize_conv_weights(keys[19], (2, 2, 128, 64))   # Input 128, Output 64
        params['conv9_1'] = initialize_conv_weights(keys[0], (3, 3, 128, 64))
        params['conv9_2'] = initialize_conv_weights(keys[1], (3, 3, 64, 64))
        params['bn9_1_gamma'], params['bn9_1_beta'] = initialize_bn_params((64,))
        params['bn9_2_gamma'], params['bn9_2_beta'] = initialize_bn_params((64,))

        # Output layer
        params['conv_out'] = initialize_conv_weights(keys[2], (1, 1, 64, 1))

        return params

    def forward(self, x: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        if len(x.shape) == 3:
            x = jnp.expand_dims(x, axis=-1)

        # Encoder
        # Block 1
        x1 = conv2d(x, params['conv1_1'])
        x1 = batch_norm(x1, params['bn1_1_gamma'], params['bn1_1_beta'])
        x1 = relu(x1)
        x1 = conv2d(x1, params['conv1_2'])
        x1 = batch_norm(x1, params['bn1_2_gamma'], params['bn1_2_beta'])
        x1 = relu(x1)
        p1 = max_pool2d(x1)

        # Block 2
        x2 = conv2d(p1, params['conv2_1'])
        x2 = batch_norm(x2, params['bn2_1_gamma'], params['bn2_1_beta'])
        x2 = relu(x2)
        x2 = conv2d(x2, params['conv2_2'])
        x2 = batch_norm(x2, params['bn2_2_gamma'], params['bn2_2_beta'])
        x2 = relu(x2)
        p2 = max_pool2d(x2)

        # Block 3
        x3 = conv2d(p2, params['conv3_1'])
        x3 = batch_norm(x3, params['bn3_1_gamma'], params['bn3_1_beta'])
        x3 = relu(x3)
        x3 = conv2d(x3, params['conv3_2'])
        x3 = batch_norm(x3, params['bn3_2_gamma'], params['bn3_2_beta'])
        x3 = relu(x3)
        p3 = max_pool2d(x3)

        # Block 4
        x4 = conv2d(p3, params['conv4_1'])
        x4 = batch_norm(x4, params['bn4_1_gamma'], params['bn4_1_beta'])
        x4 = relu(x4)
        x4 = conv2d(x4, params['conv4_2'])
        x4 = batch_norm(x4, params['bn4_2_gamma'], params['bn4_2_beta'])
        x4 = relu(x4)
        p4 = max_pool2d(x4)

        # Bottleneck
        x5 = conv2d(p4, params['conv5_1'])
        x5 = batch_norm(x5, params['bn5_1_gamma'], params['bn5_1_beta'])
        x5 = relu(x5)
        x5 = conv2d(x5, params['conv5_2'])
        x5 = batch_norm(x5, params['bn5_2_gamma'], params['bn5_2_beta'])
        x5 = relu(x5)

        # Decoder
        # Up block 4
        up4 = conv_transpose2d(x5, params['upconv4'])
        target_shape_up4 = up4.shape
        source_shape_x4 = x4.shape
        if target_shape_up4[1:3] != source_shape_x4[1:3]:
            h_diff = source_shape_x4[1] - target_shape_up4[1]
            w_diff = source_shape_x4[2] - target_shape_up4[2]
            crop_h_start = h_diff // 2
            crop_h_end = h_diff - crop_h_start
            crop_w_start = w_diff // 2
            crop_w_end = w_diff - crop_w_start
            x4_cropped = x4[:, crop_h_start:source_shape_x4[1]-crop_h_end, crop_w_start:source_shape_x4[2]-crop_w_end, :]
            merge4 = jnp.concatenate([up4, x4_cropped], axis=-1)
        else:
            merge4 = jnp.concatenate([up4, x4], axis=-1)

        x6 = conv2d(merge4, params['conv6_1'])
        x6 = batch_norm(x6, params['bn6_1_gamma'], params['bn6_1_beta'])
        x6 = relu(x6)
        x6 = conv2d(x6, params['conv6_2'])
        x6 = batch_norm(x6, params['bn6_2_gamma'], params['bn6_2_beta'])
        x6 = relu(x6)

        # Up block 3
        up3 = conv_transpose2d(x6, params['upconv3'])
        target_shape_up3 = up3.shape
        source_shape_x3 = x3.shape
        if target_shape_up3[1:3] != source_shape_x3[1:3]:
            h_diff = source_shape_x3[1] - target_shape_up3[1]
            w_diff = source_shape_x3[2] - target_shape_up3[2]
            crop_h_start = h_diff // 2
            crop_h_end = h_diff - crop_h_start
            crop_w_start = w_diff // 2
            crop_w_end = w_diff - crop_w_start
            x3_cropped = x3[:, crop_h_start:source_shape_x3[1]-crop_h_end, crop_w_start:source_shape_x3[2]-crop_w_end, :]
            merge3 = jnp.concatenate([up3, x3_cropped], axis=-1)
        else:
            merge3 = jnp.concatenate([up3, x3], axis=-1)

        x7 = conv2d(merge3, params['conv7_1'])
        x7 = batch_norm(x7, params['bn7_1_gamma'], params['bn7_1_beta'])
        x7 = relu(x7)
        x7 = conv2d(x7, params['conv7_2'])
        x7 = batch_norm(x7, params['bn7_2_gamma'], params['bn7_2_beta'])
        x7 = relu(x7)

        # Up block 2
        up2 = conv_transpose2d(x7, params['upconv2'])
        target_shape_up2 = up2.shape
        source_shape_x2 = x2.shape
        if target_shape_up2[1:3] != source_shape_x2[1:3]:
            h_diff = source_shape_x2[1] - target_shape_up2[1]
            w_diff = source_shape_x2[2] - target_shape_up2[2]
            crop_h_start = h_diff // 2
            crop_h_end = h_diff - crop_h_start
            crop_w_start = w_diff // 2
            crop_w_end = w_diff - crop_w_start
            x2_cropped = x2[:, crop_h_start:source_shape_x2[1]-crop_h_end, crop_w_start:source_shape_x2[2]-crop_w_end, :]
            merge2 = jnp.concatenate([up2, x2_cropped], axis=-1)
        else:
            merge2 = jnp.concatenate([up2, x2], axis=-1)


        x8 = conv2d(merge2, params['conv8_1'])
        x8 = batch_norm(x8, params['bn8_1_gamma'], params['bn8_1_beta'])
        x8 = relu(x8)
        x8 = conv2d(x8, params['conv8_2'])
        x8 = batch_norm(x8, params['bn8_2_gamma'], params['bn8_2_beta'])
        x8 = relu(x8)

        # Up block 1
        up1 = conv_transpose2d(x8, params['upconv1'])
        target_shape_up1 = up1.shape
        source_shape_x1 = x1.shape
        if target_shape_up1[1:3] != source_shape_x1[1:3]:
            h_diff = source_shape_x1[1] - target_shape_up1[1]
            w_diff = source_shape_x1[2] - target_shape_up1[2]
            crop_h_start = h_diff // 2
            crop_h_end = h_diff - crop_h_start
            crop_w_start = w_diff // 2
            crop_w_end = w_diff - crop_w_start
            x1_cropped = x1[:, crop_h_start:source_shape_x1[1]-crop_h_end, crop_w_start:source_shape_x1[2]-crop_w_end, :]
            merge1 = jnp.concatenate([up1, x1_cropped], axis=-1)
        else:
            merge1 = jnp.concatenate([up1, x1], axis=-1)

        x9 = conv2d(merge1, params['conv9_1'])
        x9 = batch_norm(x9, params['bn9_1_gamma'], params['bn9_1_beta'])
        x9 = relu(x9)
        x9 = conv2d(x9, params['conv9_2'])
        x9 = batch_norm(x9, params['bn9_2_gamma'], params['bn9_2_beta'])
        x9 = relu(x9)

        # Output layer
        output = conv2d(x9, params['conv_out'])
        output = sigmoid(output)

        return output

def dice_loss(pred: jnp.ndarray, target: jnp.ndarray, smooth: float = 1e-5) -> float:
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = jnp.sum(pred_flat * target_flat)
    union = jnp.sum(pred_flat) + jnp.sum(target_flat)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def binary_crossentropy_loss(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
    return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))

def combined_loss(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    dice = dice_loss(pred, target)
    bce = binary_crossentropy_loss(pred, target)
    return 0.5 * dice + 0.5 * bce

@jit
def train_step(params: Dict[str, Any], x: jnp.ndarray, y: jnp.ndarray,
               learning_rate: float, key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], float]:
    unet = UNet(key)

    def loss_fn(params):
        pred = unet.forward(x, params)
        return combined_loss(pred, y)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grads[key]

    return updated_params, loss

def preprocess_mammogram(image: np.ndarray) -> np.ndarray:
    if hasattr(image, 'device') or str(type(image)).startswith('<class \'jaxlib'):
        image = np.array(image)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced.astype(np.float32) / 255.0

def load_image_data(image_path: str, mask_path: str, target_size: Tuple[int, int] = (512, 512)):
    mammogram = load_single_image(image_path, target_size)
    if mammogram.max() > 1.0:
        mammogram = mammogram.astype(np.float32) / 255.0 if mammogram.max() <= 255 else mammogram.astype(np.float32) / mammogram.max()
    else:
        mammogram = mammogram.astype(np.float32)

    mask = load_single_image(mask_path, target_size)
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.float32) if mask.max() <= 255 else (mask > mask.max()/2).astype(np.float32)
    else:
        mask = (mask > 0.5).astype(np.float32)

    return jnp.array(mammogram), jnp.array(mask)

def load_single_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    file_ext = image_path.lower().split('.')[-1]
    if file_ext in ['dcm', 'dicom']:
        # Load DICOM using pydicom
        dicom_data = pydicom.dcmread(image_path)
        image = dicom_data.pixel_array

        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.int16:
            image = np.clip((image + 32768) / 256, 0, 255).astype(np.uint8)

        # Apply DICOM windowing if available
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            center = float(dicom_data.WindowCenter[0]) if isinstance(dicom_data.WindowCenter, pydicom.multival.MultiValue) else float(dicom_data.WindowCenter)
            width = float(dicom_data.WindowWidth[0]) if isinstance(dicom_data.WindowWidth, pydicom.multival.MultiValue) else float(dicom_data.WindowWidth)

            vmin = center - width / 2
            vmax = center + width / 2
            image = np.clip((image - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    elif file_ext == 'pgm':
        # Load PGM using PIL or custom reader
        try:
            image = np.array(Image.open(image_path))
        except:
            with open(image_path, 'rb') as f:
                header = f.readline().decode().strip()
                if header == 'P5':
                    line = f.readline().decode().strip()
                    while line.startswith('#'):
                        line = f.readline().decode().strip()

                    width, height = map(int, line.split())
                    maxval = int(f.readline().decode().strip())

                    if maxval < 256:
                        image = np.frombuffer(f.read(), dtype=np.uint8)
                    else:
                        image = np.frombuffer(f.read(), dtype=np.uint16)
                        image = (image / 256).astype(np.uint8)

                    image = image.reshape((height, width))
                else:
                    raise ValueError(f"Unsupported PGM format: {header}")

    else:
        # Load standard formats (PNG, JPG, TIF) using PIL
        image = Image.open(image_path)

        if image.mode != 'L':
            image = image.convert('L')

        image = np.array(image)

    if image.shape != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    return image

def load_dataset_from_directory(mammogram_dir: str, mask_dir: str,
                              mammogram_ext: str = None, mask_ext: str = None,
                              target_size: Tuple[int, int] = (512, 512),
                              max_samples: int = None):
    """Load dataset from directories with memory management and filename tracking."""
    import os
    import glob

    # Get all mammogram files
    if mammogram_ext:
        mammogram_pattern = os.path.join(mammogram_dir, f"*.{mammogram_ext}")
    else:
        # Search for all supported formats
        extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm', 'dicom', 'pgm']
        mammogram_files = []
        for ext in extensions:
            mammogram_files.extend(glob.glob(os.path.join(mammogram_dir, f"*.{ext}")))
        mammogram_files = sorted(mammogram_files)

    if mammogram_ext:
        mammogram_files = sorted(glob.glob(mammogram_pattern))

    # Get all mask files
    if mask_ext:
        mask_pattern = os.path.join(mask_dir, f"*.{mask_ext}")
        mask_files = sorted(glob.glob(mask_pattern))
    else:
        # Assume masks are PNG by default
        mask_pattern = os.path.join(mask_dir, "*.png")
        mask_files = sorted(glob.glob(mask_pattern))

    print(f"Found {len(mammogram_files)} mammogram files")
    print(f"Found {len(mask_files)} mask files")

    # Match files by name (assuming similar naming convention)
    matched_pairs = []
    matched_filenames = []  # Track original filenames
    
    for mammo_file in mammogram_files:
        mammo_basename = os.path.splitext(os.path.basename(mammo_file))[0]

        # Find corresponding mask
        for mask_file in mask_files:
            mask_basename = os.path.splitext(os.path.basename(mask_file))[0]

            # Simple name matching (you might need to adjust this logic)
            if mammo_basename == mask_basename or mammo_basename in mask_basename or mask_basename in mammo_basename:
                matched_pairs.append((mammo_file, mask_file))
                matched_filenames.append(mammo_basename)  # Store the base filename
                break

    # Limit samples to prevent memory issues
    if max_samples and len(matched_pairs) > max_samples:
        matched_pairs = matched_pairs[:max_samples]
        matched_filenames = matched_filenames[:max_samples]
        print(f"Limited to {max_samples} samples to prevent memory issues")

    print(f"Matched {len(matched_pairs)} image pairs")

    if len(matched_pairs) == 0:
        print("No matching pairs found, using dummy data")
        return None, None, []

    # Load all matched pairs with memory management
    images, masks = load_dataset_batch(matched_pairs, target_size)
    return images, masks, matched_filenames

def load_dataset_batch(file_pairs: list, target_size: Tuple[int, int], batch_size: int = 8):
    """Load dataset in batches to manage memory."""
    all_mammograms = []
    all_masks = []

    for i in range(0, len(file_pairs), batch_size):
        batch_pairs = file_pairs[i:i+batch_size]
        print(f"Loading batch {i//batch_size + 1}/{(len(file_pairs) + batch_size - 1)//batch_size}")

        batch_mammograms = []
        batch_masks = []

        for mammo_file, mask_file in batch_pairs:
                mammogram, mask = load_image_data(mammo_file, mask_file, target_size)
                batch_mammograms.append(mammogram)
                batch_masks.append(mask)

                # Clear variables to free memory
                del mammogram, mask

        # Convert batch to arrays and add to main lists
        if batch_mammograms:
            all_mammograms.extend(batch_mammograms)
            all_masks.extend(batch_masks)

        # Force garbage collection
        gc.collect()

    print(f"Successfully loaded {len(all_mammograms)} image pairs")
    return jnp.stack(all_mammograms), jnp.stack(all_masks)

def calculate_accuracy_metrics(pred: jnp.ndarray, target: jnp.ndarray) -> dict:
    pred_np = np.array(pred)
    target_np = np.array(target)
    
    pred_binary = (pred_np > 0.5).astype(np.float32)
    target_binary = target_np.astype(np.float32)
    
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    iou = intersection / (union + 1e-7)
    
    dice = (2.0 * intersection) / (np.sum(pred_flat) + np.sum(target_flat) + 1e-7)
    
    correct_pixels = np.sum(pred_flat == target_flat)
    total_pixels = len(pred_flat)
    pixel_accuracy = correct_pixels / total_pixels
    
    tp = intersection
    fp = np.sum(pred_flat) - tp
    fn = np.sum(target_flat) - tp
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return {
        'iou': float(iou),
        'dice': float(dice),
        'pixel_accuracy': float(pixel_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }

def calculate_dice_score(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    pred_flat = (pred > 0.5).flatten()
    target_flat = (target > 0.5).flatten()
    
    intersection = jnp.sum(pred_flat * target_flat)
    total = jnp.sum(pred_flat) + jnp.sum(target_flat)
    
    return float((2.0 * intersection) / (total + 1e-7))

def calculate_iou_score(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """Calculate IoU score."""
    pred_flat = (pred > 0.5).flatten()
    target_flat = (target > 0.5).flatten()
    
    intersection = jnp.sum(pred_flat * target_flat)
    union = jnp.sum(pred_flat) + jnp.sum(target_flat) - intersection
    
    return float(intersection / (union + 1e-7))

def main():
    start_time = time.time()
    print_section_header("INITIALIZING U-NET FOR BREAST MAMMOGRAM ROI SEGMENTATION")
    print_timestamp("Starting initialization...")
    
    SAVE_CHECKPOINTS = True
    SAVE_VISUALIZATIONS = True
    CHECKPOINT_DIR = "model_checkpoints"
    VISUALIZATION_DIR = "visualizations"
    PREDICTION_DIR = "predictions"
    
    # Training parameters
    learning_rate = 0.01
    epochs = 50
    use_augmentation = False
    target_size = (512, 512)
    
    RESUME_TRAINING = False  
    # RESUME_CHECKPOINT = "model_checkpoints/unet_checkpoint_epoch_15.pkl"  

    training_config = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'use_augmentation': use_augmentation,
        'target_size': target_size,
        'batch_size': 1,
        'optimizer': 'sgd'
    }
    training_config['config_hash'] = get_training_config_hash(
        learning_rate, epochs, use_augmentation, target_size, 
        {'batch_size': 1, 'optimizer': 'sgd'}
    )
    
    for dir_path in [CHECKPOINT_DIR, VISUALIZATION_DIR, PREDICTION_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"   📁 Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"   📁 Visualization directory: {VISUALIZATION_DIR}")
    print(f"   📁 Prediction directory: {PREDICTION_DIR}")
    
    global key
    key, subkey = random.split(key)
    unet = UNet(subkey)
    print_timestamp("✓ U-Net model initialized successfully")
    
    if RESUME_TRAINING:
        print_section_header("RESUME TRAINING MODE")
        if not os.path.exists(RESUME_CHECKPOINT):
            print_timestamp(f"❌ ERROR: Checkpoint not found: {RESUME_CHECKPOINT}")
            print("   Available checkpoints:")
            for f in os.listdir(CHECKPOINT_DIR):
                if f.endswith('.pkl'):
                    print(f"      - {f}")
            return None
        print_timestamp(f"Loading checkpoint: {RESUME_CHECKPOINT}")
        params, checkpoint_metadata = load_model_checkpoint(RESUME_CHECKPOINT)
        start_epoch = checkpoint_metadata.get('epoch', 0)
        if isinstance(start_epoch, str) and start_epoch == "final":
            print_timestamp("⚠️  WARNING: Resuming from 'final' model")
            print("   Starting from epoch 0 with loaded weights")
            start_epoch = 0
        
        saved_config = checkpoint_metadata.get('training_config', {})
        if saved_config:
            if saved_config.get('learning_rate') != learning_rate:
                print_timestamp(f"⚠️  Learning rate changed: {saved_config.get('learning_rate')} → {learning_rate}")
            if saved_config.get('target_size') != target_size:
                print_timestamp("❌ ERROR: Target size mismatch! Cannot resume with different image size.")
                return None
        
        print_timestamp(f"✓ Resuming from epoch {start_epoch}")
        print(f"   Target epochs: {epochs}")
        
        if start_epoch >= epochs:
            print_timestamp(f"⚠️  Model already trained for {start_epoch} epochs (target: {epochs})")
            response = input("   Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print_timestamp("Training cancelled")
                return params
        
        action = 'resume_training'
    else:
        print_section_header("CHECKPOINT ANALYSIS")
        action_info = determine_training_action(CHECKPOINT_DIR, training_config)
        action = action_info['action']
        
        print_timestamp(f"Training action determined: {action}")
        print(f"   📋 Reason: {action_info['reason']}")
        print(f"   ⚙️  Current config hash: {training_config['config_hash']}")
        
        if action_info['checkpoint_path']:
            print(f"   📁 Checkpoint path: {action_info['checkpoint_path']}")
        
        start_epoch = action_info.get('start_epoch', 0)
        
        if action == 'skip_training':
            params, checkpoint_metadata = load_model_checkpoint(action_info['checkpoint_path'])
        elif action == 'resume_training':
            params, checkpoint_metadata = load_model_checkpoint(action_info['checkpoint_path'])
        else:
            params = unet.params
    
    print_section_header("LOADING DATASET")
    
    mammogram_dir = '../DMID _Dataset/512/TIFF/'
    mask_dir = '../DMID _Dataset/512/Mask/'
    print_timestamp(f"Scanning directories...")
    print(f"   📁 Mammogram directory: {mammogram_dir}")
    print(f"   📁 Mask directory: {mask_dir}")

    all_images, all_masks, all_filenames = load_dataset_from_directory(
        mammogram_dir, mask_dir,
        mammogram_ext=None,
        mask_ext='png',
        target_size=target_size,
        max_samples=200
    )
    
    print_timestamp(f"✓ Dataset loaded successfully")
    print(f"   📊 Total samples: {len(all_images)}")
    
    split_idx = int(0.8 * len(all_images))
    train_images, test_images = all_images[:split_idx], all_images[split_idx:]
    train_masks, test_masks = all_masks[:split_idx], all_masks[split_idx:]
    train_filenames, test_filenames = all_filenames[:split_idx], all_filenames[split_idx:]
    
    print_section_header("PREPROCESSING IMAGES")
    processed_train_images = []
    for i in range(len(train_images)):
        img_np = np.array(train_images[i])
        processed_img = preprocess_mammogram(img_np)
        processed_train_images.append(processed_img)
    train_images = jnp.stack(processed_train_images)
    
    processed_test_images = []
    for i in range(len(test_images)):
        img_np = np.array(test_images[i])
        processed_img = preprocess_mammogram(img_np)
        processed_test_images.append(processed_img)
    test_images = jnp.stack(processed_test_images)
    
    if action == 'skip_training':
        print_section_header("LOADING EXISTING FINAL MODEL")
        print_timestamp("✓ Final model loaded from checkpoint")
        
    else:  
        if RESUME_TRAINING or action == 'resume_training':
            print_section_header(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
        else:
            print_section_header("TRAINING FROM SCRATCH")
            start_epoch = 0
        
        training_start_time = time.time()
        print_timestamp(f"Training from epoch {start_epoch + 1} to {epochs}...")
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            key, epoch_key = random.split(key)
            
            status = "RESUMED" if epoch == start_epoch and start_epoch > 0 else ""
            print(f"\n🔄 EPOCH {epoch + 1}/{epochs} {status}")
            print("-" * 50)

            for i in range(len(train_images)):
                if (i + 1) % 5 == 0 or (i + 1) == len(train_images):
                    print_progress_bar(i + 1, len(train_images), prefix=f'Epoch {epoch + 1} training')
                
                x_batch = jnp.expand_dims(train_images[i], axis=0)
                y_batch = jnp.expand_dims(train_masks[i], axis=0)

                key, train_key = random.split(key)
                params, loss = train_step(params, x_batch, y_batch, learning_rate, train_key)
                epoch_loss += loss

                del x_batch, y_batch
                if i % 4 == 0:
                    gc.collect()

            avg_loss = epoch_loss / len(train_images)
            epoch_duration = time.time() - epoch_start_time
            
            print(f"   ⏱️  Duration: {epoch_duration:.1f}s")
            print(f"   📊 Average Loss: {avg_loss:.6f}")
            
            if SAVE_CHECKPOINTS and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
                save_model_checkpoint(
                    params, epoch + 1, CHECKPOINT_DIR, 
                    training_config, {'avg_loss': float(avg_loss)}
                )
                print(f"   💾 Checkpoint saved: epoch_{epoch + 1}.pkl")
                
                cleanup_old_checkpoints(CHECKPOINT_DIR)

            if (epoch + 1) % 5 == 0:
                learning_rate *= 0.9
                print(f"   📉 Learning rate decayed to: {learning_rate:.6f}")

        if SAVE_CHECKPOINTS:
            save_model_checkpoint(
                params, "final", CHECKPOINT_DIR, training_config
            )
            print_timestamp("✓ Final model saved")

    print_section_header("MODEL EVALUATION")
    
    print_timestamp("Starting evaluation on test data...")
    total_metrics = {'iou': 0, 'dice': 0, 'pixel_accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    all_predictions = []
    detailed_results = []

    for i in range(len(test_images)):
        print_progress_bar(i + 1, len(test_images), prefix='Evaluating')
        
        x_test = jnp.expand_dims(test_images[i], axis=0)
        y_test = test_masks[i]
        filename = test_filenames[i]

        pred = unet.forward(x_test, params)
        pred = pred[0, :, :, 0]
        all_predictions.append(pred)

        metrics = calculate_accuracy_metrics(pred, y_test)
        detailed_results.append({
            'filename': filename,
            'metrics': metrics
        })
        
        for key_name in total_metrics:
            total_metrics[key_name] += metrics[key_name]

        if i < 3:
            print(f"\n   Sample {i + 1}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}, F1={metrics['f1_score']:.3f}")

        del x_test, y_test, metrics
        gc.collect()

    # Generate visualizations if enabled
    if SAVE_VISUALIZATIONS:
        print_section_header("GENERATING VISUALIZATIONS")
        
        print_timestamp("Creating visualizations...")
        save_predictions_as_images(all_predictions, 
            test_filenames,
            PREDICTION_DIR)
        
        num_viz_samples = min(5, len(test_images))
        for i in range(num_viz_samples):
            print_progress_bar(i + 1, num_viz_samples, prefix='Creating visualizations')
            viz_path = os.path.join(VISUALIZATION_DIR, f"comparison_sample_{i+1}.png")
            visualize_results(test_images[i], test_masks[i], all_predictions[i], save_path=viz_path)
        print_timestamp("✓ All visualizations created")

    print_section_header("FINAL RESULTS")
    total_duration = time.time() - start_time
    print_timestamp(f"Total execution time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print("\n🎯 AVERAGE TEST METRICS:")
    print("-" * 40)
    for key_name in total_metrics:
        avg_value = total_metrics[key_name] / len(test_images)
        icon = "🎯" if "accuracy" in key_name or "f1" in key_name else "📊"
        print(f"  {icon} {key_name.upper().replace('_', ' ')}: {avg_value:.4f}")

    print_timestamp("🎉 Complete pipeline execution finished!")
    
    return {
        'params': params,
        'test_metrics': {k: v/len(test_images) for k, v in total_metrics.items()},
        'predictions': all_predictions,
        'execution_time': total_duration,
        'training_action': action if not RESUME_TRAINING else 'resume_training'
    }

def monitor_system_resources():
    try:
        import psutil
        import GPUtil
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3)
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                resources.update({
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_temperature': gpu.temperature
                })
        except:
            pass
            
        return resources
    except ImportError:
        return None

def get_training_config_hash(learning_rate: float, epochs: int, use_augmentation: bool, 
                           target_size: tuple, other_params: dict = None) -> str:
    """Generate a hash for training configuration to detect parameter changes."""
    config = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'use_augmentation': use_augmentation,
        'target_size': target_size
    }
    if other_params:
        config.update(other_params)
    
    # Convert to string and hash
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def save_model_checkpoint(params: Dict[str, Any], epoch: int, save_dir: str = "checkpoints",
                                 training_config: dict = None, metrics: dict = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(epoch, str):  
        checkpoint_path = os.path.join(save_dir, f"unet_{epoch}_model.pkl")
    else:
        checkpoint_path = os.path.join(save_dir, f"unet_checkpoint_epoch_{epoch}.pkl")
    
    checkpoint_data = {
        'params': params,
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'architecture': 'UNet',
            'input_size': (512, 512),
            'num_classes': 1
        },
        'training_config': training_config or {},
        'metrics': metrics or {},
        'config_hash': training_config.get('config_hash') if training_config else None
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if training_config:
        config_path = os.path.join(save_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
    
    return checkpoint_path

def load_model_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], dict]:
    """Load enhanced model checkpoint with metadata."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle both old format (just params) and new format (with metadata)
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        print(f"   ✓ Loaded checkpoint from epoch {checkpoint_data.get('epoch', 'unknown')}")
        print(f"   📅 Saved on: {checkpoint_data.get('timestamp', 'unknown')}")
        if checkpoint_data.get('training_config'):
            print(f"   ⚙️  Config hash: {checkpoint_data.get('config_hash', 'N/A')}")
        return checkpoint_data['params'], checkpoint_data
    else:
        # Old format - just parameters
        return checkpoint_data, {}

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    
    # Look for final model first
    final_model_path = os.path.join(checkpoint_dir, "unet_final_model.pkl")
    if os.path.exists(final_model_path):
        return final_model_path
    
    # Look for epoch checkpoints
    for file in os.listdir(checkpoint_dir):
        if file.startswith("unet_checkpoint_epoch_") and file.endswith(".pkl"):
            try:
                epoch_num = int(file.split("_")[-1].split(".")[0])
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    if checkpoint_files:
        # Sort by epoch number and return the latest
        checkpoint_files.sort(key=lambda x: x[0])
        return checkpoint_files[-1][1]
    
    return None

def list_available_checkpoints(checkpoint_dir: str = "model_checkpoints"):
    print("="*70)
    print("  AVAILABLE CHECKPOINTS")
    print("="*70)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoint_files = []
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(checkpoint_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Extract info
                if isinstance(checkpoint, dict):
                    epoch = checkpoint.get('epoch', 'unknown')
                    timestamp = checkpoint.get('timestamp', 'unknown')
                    metrics = checkpoint.get('metrics', {})
                    
                    checkpoint_files.append({
                        'filename': filename,
                        'path': filepath,
                        'epoch': epoch,
                        'timestamp': timestamp,
                        'metrics': metrics
                    })
                else:
                    # Old format
                    checkpoint_files.append({
                        'filename': filename,
                        'path': filepath,
                        'epoch': 'unknown',
                        'timestamp': 'unknown',
                        'metrics': {}
                    })
            except Exception as e:
                print(f"⚠️  Could not read {filename}: {str(e)}")
    
    # Sort by epoch
    def get_epoch_num(ckpt):
        epoch = ckpt['epoch']
        if epoch == 'final':
            return float('inf')
        elif isinstance(epoch, int):
            return epoch
        else:
            return -1
    
    checkpoint_files.sort(key=get_epoch_num)
    
    print(f"\nFound {len(checkpoint_files)} checkpoints:\n")
    
    for i, ckpt in enumerate(checkpoint_files, 1):
        print(f"{i}. {ckpt['filename']}")
        print(f"   Epoch: {ckpt['epoch']}")
        print(f"   Saved: {ckpt['timestamp'][:19] if ckpt['timestamp'] != 'unknown' else 'unknown'}")
        if ckpt['metrics']:
            print(f"   Loss: {ckpt['metrics'].get('avg_loss', 'N/A'):.6f}")
        print()
    
    print("="*70)
    print("\nTo resume from a checkpoint, set in main_with_resume():")
    print("  RESUME_TRAINING = True")
    print("  RESUME_CHECKPOINT = 'model_checkpoints/[checkpoint_filename]'")
    print("="*70)
    
    return checkpoint_files

def check_training_config_changed(current_config: dict, checkpoint_dir: str) -> bool:
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    
    if not os.path.exists(config_path):
        return True  # No previous config, assume changed
    
    try:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Compare config hashes
        current_hash = current_config.get('config_hash')
        saved_hash = saved_config.get('config_hash')
        
        if current_hash and saved_hash:
            return current_hash != saved_hash
        
        # Fallback: compare key parameters
        key_params = ['learning_rate', 'epochs', 'use_augmentation', 'target_size']
        for param in key_params:
            if current_config.get(param) != saved_config.get(param):
                return True
        
        return False
    except (json.JSONDecodeError, KeyError):
        return True  # Error reading config, assume changed

def determine_training_action(checkpoint_dir: str, current_config: dict) -> dict:
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    action_info = {
        'action': 'train_from_scratch',  # Default action
        'checkpoint_path': None,
        'start_epoch': 0,
        'reason': 'No existing checkpoints found'
    }
    
    if not latest_checkpoint:
        return action_info
    
    # Load checkpoint to check details
    try:
        _, checkpoint_metadata = load_model_checkpoint(latest_checkpoint)
        
        # Check if it's a final model
        if "final" in os.path.basename(latest_checkpoint):
            # Check if config changed
            if check_training_config_changed(current_config, checkpoint_dir):
                action_info.update({
                    'action': 'train_from_scratch',
                    'reason': 'Training configuration changed, starting fresh training'
                })
            else:
                action_info.update({
                    'action': 'skip_training',
                    'checkpoint_path': latest_checkpoint,
                    'reason': 'Final model exists with same configuration'
                })
        else:
            # It's an epoch checkpoint
            epoch = checkpoint_metadata.get('epoch', 0)
            
            if check_training_config_changed(current_config, checkpoint_dir):
                action_info.update({
                    'action': 'train_from_scratch',
                    'reason': 'Training configuration changed, starting fresh training'
                })
            else:
                action_info.update({
                    'action': 'resume_training',
                    'checkpoint_path': latest_checkpoint,
                    'start_epoch': epoch,
                    'reason': f'Resuming training from epoch {epoch}'
                })
    
    except Exception as e:
        action_info.update({
            'action': 'train_from_scratch',
            'reason': f'Error loading checkpoint: {str(e)}'
        })
    
    return action_info

def cleanup_old_checkpoints(checkpoint_dir: str, keep_latest: int = 3):
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("unet_checkpoint_epoch_") and file.endswith(".pkl"):
            try:
                epoch_num = int(file.split("_")[-1].split(".")[0])
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    if len(checkpoint_files) > keep_latest:
        # Sort by epoch and remove older ones
        checkpoint_files.sort(key=lambda x: x[0])
        to_remove = checkpoint_files[:-keep_latest]
        
        for _, file_path in to_remove:
            try:
                os.remove(file_path)
                print(f"   🗑️  Removed old checkpoint: {os.path.basename(file_path)}")
            except OSError:
                pass

def cleanup_memory():
    """Perform comprehensive memory cleanup."""
    import gc
    
    # Clear JAX caches
    try:
        jax.clear_caches()
    except:
        pass
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    print_timestamp("🧹 Memory cleanup completed")

def save_predictions_as_images(predictions: list, filenames: list, output_dir: str, 
                                       test_indices: list = None):
    from PIL import Image
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_timestamp(f"Saving {len(predictions)} predictions to {output_dir}...")
    
    for i, pred in enumerate(predictions):
        # Convert JAX array to NumPy array first
        pred_np = np.array(pred)
        
        # Convert prediction to 0-255 range
        pred_img = (pred_np * 255).astype(np.uint8)
        
        # Use original filename or create one based on test indices
        if test_indices and i < len(test_indices):
            original_idx = test_indices[i]
            if original_idx < len(filenames):
                base_filename = filenames[original_idx]
            else:
                base_filename = f"test_sample_{i+1}"
        elif i < len(filenames):
            base_filename = filenames[i]
        else:
            base_filename = f"prediction_{i+1:03d}"
        
        # Create filename with prediction suffix
        filename = f"{base_filename}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Save as PNG
        Image.fromarray(pred_img).save(save_path)
        
        # Print progress every 5 saves or at the end
        if (i + 1) % 5 == 0 or (i + 1) == len(predictions):
            print_progress_bar(i + 1, len(predictions), prefix='Saving predictions')
    
    print_timestamp(f"✓ Saved {len(predictions)} prediction images with original names")

def visualize_results(mammogram: jnp.ndarray, ground_truth: jnp.ndarray,
                     prediction: jnp.ndarray, save_path: str = None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    mammogram_np = np.array(mammogram)
    ground_truth_np = np.array(ground_truth)
    prediction_np = np.array(prediction)
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original mammogram
    axes[0].imshow(mammogram_np, cmap='gray', aspect='equal')
    axes[0].set_title('Original Mammogram', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(ground_truth_np, cmap='gray', aspect='equal')
    axes[1].set_title('Ground Truth ROI', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    pred_binary = (prediction_np > 0.5).astype(float)
    axes[2].imshow(pred_binary, cmap='gray', aspect='equal')
    axes[2].set_title('Predicted ROI', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add metrics as subtitle
    dice_score = calculate_dice_score(prediction, ground_truth)
    iou_score = calculate_iou_score(prediction, ground_truth)
    fig.suptitle(f'Dice: {dice_score:.3f} | IoU: {iou_score:.3f}', 
                 fontsize=12, y=0.02)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()

def predict_single_image_roi(image_path: str, 
                           model_params: dict,
                           target_size: Tuple[int, int] = (512, 512),
                           checkpoint_path: str = None,
                           preprocess: bool = True,
                           save_prediction: bool = True,
                           output_dir: str = "single_predictions",
                           visualization: bool = True) -> dict:
    print(f"🔍 Predicting ROI for: {os.path.basename(image_path)}")
    
    valid_sizes = [(256, 256), (512, 512), (1024, 1024)]
    if target_size not in valid_sizes:
        raise ValueError(f"Target size must be one of {valid_sizes}")
    
    if save_prediction or visualization:
        os.makedirs(output_dir, exist_ok=True)
    
    print("📂 Loading model from checkpoint...")
    model_params, _ = load_model_checkpoint(checkpoint_path)
    print("✅ Model loaded successfully")
    
    key = jax.random.PRNGKey(42)
    unet = UNet(key)
    
    try:
        print("📖 Loading image...")
        image = load_single_image_for_prediction(image_path, target_size)
        original_shape = image.shape
        
        if preprocess:
            print("🔧 Applying preprocessing...")
            image = preprocess_mammogram(np.array(image))
        
        image_jax = jnp.array(image)
        if len(image_jax.shape) == 2:
            image_jax = jnp.expand_dims(image_jax, axis=[0, -1])  # Add batch and channel dims
        elif len(image_jax.shape) == 3:
            image_jax = jnp.expand_dims(image_jax, axis=0)  # Add batch dim only
        
        print(f"🎯 Input shape: {image_jax.shape}")
        
        print("🧠 Running inference...")
        prediction = unet.forward(image_jax, model_params)
        prediction_2d = prediction[0, :, :, 0] if prediction.shape[-1] == 1 else prediction[0]
        confidence_metrics = calculate_prediction_confidence(prediction_2d)
        
        print("✅ Prediction completed!")
        print(f"📊 Prediction stats: min={np.min(prediction_2d):.3f}, max={np.max(prediction_2d):.3f}, mean={np.mean(prediction_2d):.3f}")
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        results = {
            'prediction_mask': np.array(prediction_2d),
            'prediction_binary': np.array(prediction_2d > 0.5),
            'confidence_metrics': confidence_metrics,
            'input_image': np.array(image),
            'original_shape': original_shape,
            'target_size': target_size,
            'base_filename': base_filename,
            'input_path': image_path
        }
        
        if save_prediction:
            save_single_prediction(results, output_dir)
        
        print(f"🎉 Processing complete! Results saved to: {output_dir}")
        return results
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise e

def load_single_image_for_prediction(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = load_single_image(image_path, target_size)
    
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image.astype(np.float32) / image.max()
    else:
        image = image.astype(np.float32)
    
    return image

def calculate_prediction_confidence(prediction: jnp.ndarray) -> dict:
    pred_np = np.array(prediction)
    
    mean_confidence = float(np.mean(pred_np))
    max_confidence = float(np.max(pred_np))
    min_confidence = float(np.min(pred_np))
    std_confidence = float(np.std(pred_np))
    
    binary_pred = pred_np > 0.5
    roi_area_ratio = float(np.sum(binary_pred) / binary_pred.size)
    
    high_confidence_ratio = float(np.sum(pred_np > 0.8) / pred_np.size)
    low_confidence_ratio = float(np.sum(pred_np < 0.2) / pred_np.size)
    uncertain_ratio = float(np.sum((pred_np >= 0.3) & (pred_np <= 0.7)) / pred_np.size)
    
    return {
        'mean_confidence': mean_confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'std_confidence': std_confidence,
        'roi_area_ratio': roi_area_ratio,
        'high_confidence_ratio': high_confidence_ratio,
        'low_confidence_ratio': low_confidence_ratio,
        'uncertain_ratio': uncertain_ratio
    }

def save_single_prediction(results: dict, output_dir: str):
    base_filename = results['base_filename']
    
    binary_mask = (results['prediction_binary'] * 255).astype(np.uint8)
    mask_path = os.path.join(output_dir, f"{base_filename}.png")
    Image.fromarray(binary_mask).save(mask_path)
    
    print(f"💾 Saved prediction files:")
    print(f"   📄 Binary mask: {mask_path}")

def predict_multiple_images(image_paths: list, checkpoint_path: str = None,
                           output_dir: str = "batch_predictions") -> list:
    print_section_header(f"BATCH PREDICTION FOR {len(image_paths)} IMAGES")
    all_results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing Image {i+1}/{len(image_paths)} ---")
        
        try:
            result = predict_single_image_roi(
                image_path=image_path,
                 model_params=None,
                checkpoint_path=checkpoint_path,
                target_size=(1024, 1024),
                output_dir=output_dir
            )
            all_results.append(result)
            
        except Exception as e:
            print_timestamp(f"❌ Error processing {image_path}: {str(e)}")
            continue
        
        print_progress_bar(i + 1, len(image_paths), prefix='Batch prediction')
    
    print_timestamp(f"✓ Completed batch prediction: {len(all_results)}/{len(image_paths)} successful")
    return all_results

def print_section_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_progress_bar(current, total, prefix='Progress', length=40):
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  

def print_timestamp(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def parse_training_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ROI Segmentation Model')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Total epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='List available checkpoints and exit')
    
    return parser.parse_args()

if __name__ == "__main__":
    print("Setting memory-friendly environment variables...")
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'  

    # prediction = predict_single_image_roi(
    # image_path="../MIAS Dataset/MIAS/mdb300.png",
    # model_params=None,
    # checkpoint_path="model_checkpoints/unet_final_model.pkl",
    # target_size=(1024, 1024)  # High resolution
    # )

    args = parse_training_args()
    if args.list_checkpoints:
        list_available_checkpoints()
    else:
        RESUME_CHECKPOINT = args.resume
        RESUME_TRAINING = args.resume is not None
        epochs = args.epochs
        learning_rate = args.lr
        results = main()
