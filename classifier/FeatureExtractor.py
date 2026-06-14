import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MammographyFeatureExtractor:
    def __init__(self):
        self.features = {}
        
    def load_images(self, mammogram_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)
        if mammogram is None:
            raise ValueError(f"Could not load mammogram from {mammogram_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from {mask_path}")
            
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mammogram, mask
    
    def extract_roi(self, mammogram: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi = cv2.bitwise_and(mammogram, mammogram, mask=mask)
        return roi
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        features = {}
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {key: 0.0 for key in ['compactness', 'circularity', 'irregularity', 
                                       'spiculation_index', 'fractal_dimension', 
                                       'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                                       'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7']}
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0 or perimeter == 0:
            return {key: 0.0 for key in ['compactness', 'circularity', 'irregularity', 
                                       'spiculation_index', 'fractal_dimension', 
                                       'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                                       'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7']}
        
        features['compactness'] = (4 * np.pi * area) / (perimeter ** 2)
        features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
        features['irregularity'] = (perimeter ** 2) / (4 * np.pi * area)
        features['spiculation_index'] = self._calculate_spiculation_index(contour)
        features['fractal_dimension'] = self._calculate_fractal_dimension(contour)
        
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        for i, hu in enumerate(hu_moments.flatten()):
            features[f'hu_moment_{i+1}'] = -np.sign(hu) * np.log10(abs(hu)) if hu != 0 else 0
        
        return features
    
    def extract_global_texture_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        features = {}
        
        processed_mammogram = self._preprocess_mammogram(mammogram)
        
        glcm_features = self._extract_global_glcm_features(processed_mammogram)
        features.update(glcm_features)
        
        lbp_features = self._extract_global_lbp_features(processed_mammogram)
        features.update(lbp_features)
        
        wavelet_features = self._extract_global_wavelet_features(processed_mammogram)
        features.update(wavelet_features)
        
        statistical_features = self._extract_statistical_features(processed_mammogram)
        features.update(statistical_features)
        
        edge_features = self._extract_edge_features(processed_mammogram)
        features.update(edge_features)
        return features
    
    def _preprocess_mammogram(self, mammogram: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(mammogram, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
       
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            breast_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            breast_mask = binary
        
        processed = np.where(breast_mask > 0, mammogram, 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed.astype(np.uint8))
        return processed
    
    def _extract_global_glcm_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        h, w = mammogram.shape
        if h > 512 or w > 512:
            scale_factor = min(512/h, 512/w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            mammogram_small = cv2.resize(mammogram, (new_w, new_h))
        else:
            mammogram_small = mammogram.copy()
        
        non_zero_mask = mammogram_small > 0
        if not np.any(non_zero_mask):
            return {'glcm_contrast': 0, 'glcm_homogeneity': 0, 'glcm_energy': 0, 
                   'glcm_entropy': 0, 'glcm_correlation': 0}
        
        mammogram_norm = mammogram_small.copy()
        mammogram_norm[non_zero_mask] = ((mammogram_small[non_zero_mask] - 
                                        np.min(mammogram_small[non_zero_mask])) / 
                                       (np.max(mammogram_small[non_zero_mask]) - 
                                        np.min(mammogram_small[non_zero_mask])) * 31).astype(np.uint8)
        
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        all_contrast = []
        all_homogeneity = []
        all_energy = []
        all_entropy = []
        all_correlation = []
        
        for distance in distances:
            for angle in angles:
                try:
                    glcm = graycomatrix(mammogram_norm, distances=[distance], angles=[angle], 
                                      levels=32, symmetric=True, normed=True)
                    
                    # Extract properties
                    all_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
                    all_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
                    all_energy.append(graycoprops(glcm, 'energy')[0, 0])
                    all_correlation.append(graycoprops(glcm, 'correlation')[0, 0])
                    
                    # Calculate entropy manually
                    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
                    all_entropy.append(entropy)
                    
                except:
                    continue
        
        features = {
            'glcm_contrast': np.mean(all_contrast) if all_contrast else 0,
            'glcm_homogeneity': np.mean(all_homogeneity) if all_homogeneity else 0,
            'glcm_energy': np.mean(all_energy) if all_energy else 0,
            'glcm_entropy': np.mean(all_entropy) if all_entropy else 0,
            'glcm_correlation': np.mean(all_correlation) if all_correlation else 0
        }
        
        return features
    
    def _extract_global_lbp_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(mammogram, n_points, radius, method='uniform')
        non_zero_mask = mammogram > 0
        lbp_values = lbp[non_zero_mask]
        
        if len(lbp_values) == 0:
            return {'lbp_uniformity': 0, 'lbp_entropy': 0, 'lbp_variance': 0}
        
        hist, _ = np.histogram(lbp_values, bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)  # Normalize
        
        features = {
            'lbp_uniformity': np.sum(hist ** 2),
            'lbp_entropy': -np.sum(hist * np.log2(hist + 1e-10)),
            'lbp_variance': np.var(lbp_values)
        }
        
        return features
    
    def _extract_global_wavelet_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        non_zero_mask = mammogram > 0
        if not np.any(non_zero_mask):
            return {'wavelet_energy_ll': 0, 'wavelet_energy_lh': 0, 
                   'wavelet_energy_hl': 0, 'wavelet_energy_hh': 0,
                   'wavelet_mean_ll': 0, 'wavelet_std_ll': 0}
        
        try:
            coeffs = pywt.dwt2(mammogram.astype(float), 'db4')
            cA, (cH, cV, cD) = coeffs
            features = {
                'wavelet_energy_ll': np.sum(cA ** 2),
                'wavelet_energy_lh': np.sum(cH ** 2),
                'wavelet_energy_hl': np.sum(cV ** 2),
                'wavelet_energy_hh': np.sum(cD ** 2)
            }
            
            total_energy = sum(features.values())
            if total_energy > 0:
                for key in features:
                    features[key] /= total_energy
            
            features['wavelet_mean_ll'] = np.mean(cA)
            features['wavelet_std_ll'] = np.std(cA)
                    
        except:
            features = {
                'wavelet_energy_ll': 0, 'wavelet_energy_lh': 0, 
                'wavelet_energy_hl': 0, 'wavelet_energy_hh': 0,
                'wavelet_mean_ll': 0, 'wavelet_std_ll': 0
            }
        return features
    
    def _extract_statistical_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        non_zero_pixels = mammogram[mammogram > 0]
        
        if len(non_zero_pixels) == 0:
            return {'mean_intensity': 0, 'std_intensity': 0, 'skewness': 0, 
                   'kurtosis': 0, 'intensity_range': 0}
        
        mean_val = np.mean(non_zero_pixels)
        std_val = np.std(non_zero_pixels)

        if std_val > 0:
            normalized = (non_zero_pixels - mean_val) / std_val
            skewness = np.mean(normalized ** 3)
            kurtosis = np.mean(normalized ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        features = {
            'mean_intensity': mean_val,
            'std_intensity': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'intensity_range': np.max(non_zero_pixels) - np.min(non_zero_pixels)
        }
        return features
    
    def _extract_edge_features(self, mammogram: np.ndarray) -> Dict[str, float]:
        blurred = cv2.GaussianBlur(mammogram, (5, 5), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        edges = cv2.Canny(blurred.astype(np.uint8), 50, 150)
        
        non_zero_mask = mammogram > 0
        features = {
            'edge_density': np.sum(edges > 0) / np.sum(non_zero_mask) if np.sum(non_zero_mask) > 0 else 0,
            'mean_gradient_magnitude': np.mean(gradient_magnitude[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 0,
            'std_gradient_magnitude': np.std(gradient_magnitude[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 0,
            'gradient_direction_variance': np.var(gradient_direction[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 0
        }
        return features
    
    def _calculate_spiculation_index(self, contour: np.ndarray) -> float:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
    
        original_perimeter = cv2.arcLength(contour, True)
        smooth_perimeter = cv2.arcLength(smooth_contour, True)
        
        if smooth_perimeter == 0:
            return 0.0
        
        return original_perimeter / smooth_perimeter
    
    def _calculate_fractal_dimension(self, contour: np.ndarray) -> float:
        try:
            img_shape = (512, 512)
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, 1)
            
            scales = np.logspace(0.5, 3.5, num=10, dtype=int)
            Ns = []
            
            for scale in scales:
                h, w = mask.shape
                new_h, new_w = h // scale, w // scale
                if new_h == 0 or new_w == 0:
                    continue
                    
                downsampled = cv2.resize(mask, (new_w, new_h))
                count = np.sum(downsampled > 0)
                if count > 0:
                    Ns.append(count)
            
            if len(Ns) < 2:
                return 1.0
            
            valid_indices = [i for i, n in enumerate(Ns) if n > 0]
            if len(valid_indices) < 2:
                return 1.0
            
            valid_scales = [scales[i] for i in valid_indices]
            valid_Ns = [Ns[i] for i in valid_indices]
            
            coeffs = np.polyfit(np.log(valid_scales), np.log(valid_Ns), 1)
            fractal_dim = -coeffs[0]
            
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception:
            return 1.0
    
    def extract_texture_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        features = {}
        
        lesion_pixels = roi[mask > 0]
        if len(lesion_pixels) == 0:
            return {key: 0.0 for key in ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 
                                       'glcm_entropy', 'glcm_correlation', 'lbp_uniformity',
                                       'lbp_entropy', 'wavelet_energy_ll', 'wavelet_energy_lh',
                                       'wavelet_energy_hl', 'wavelet_energy_hh']}
        
        glcm_features = self._extract_glcm_features(roi, mask)
        features.update(glcm_features)
        
        lbp_features = self._extract_lbp_features(roi, mask)
        features.update(lbp_features)
        
        wavelet_features = self._extract_wavelet_features(roi, mask)
        features.update(wavelet_features)
        return features
    
    def _extract_glcm_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        masked_roi = np.where(mask > 0, roi, 0)
        if np.max(masked_roi) == 0:
            return {'glcm_contrast': 0, 'glcm_homogeneity': 0, 'glcm_energy': 0, 
                   'glcm_entropy': 0, 'glcm_correlation': 0}
        
        masked_roi = ((masked_roi - np.min(masked_roi[mask > 0])) / 
                     (np.max(masked_roi) - np.min(masked_roi[mask > 0])) * 255).astype(np.uint8)
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        all_contrast = []
        all_homogeneity = []
        all_energy = []
        all_entropy = []
        all_correlation = []
        
        for distance in distances:
            for angle in angles:
                try:
                    glcm = graycomatrix(masked_roi, distances=[distance], angles=[angle], 
                                      levels=256, symmetric=True, normed=True)
                    
                    all_contrast.append(graycoprops(glcm, 'contrast')[0, 0])
                    all_homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
                    all_energy.append(graycoprops(glcm, 'energy')[0, 0])
                    all_correlation.append(graycoprops(glcm, 'correlation')[0, 0])
                    
                    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
                    all_entropy.append(entropy)
                    
                except:
                    continue
        
        features = {
            'glcm_contrast': np.mean(all_contrast) if all_contrast else 0,
            'glcm_homogeneity': np.mean(all_homogeneity) if all_homogeneity else 0,
            'glcm_energy': np.mean(all_energy) if all_energy else 0,
            'glcm_entropy': np.mean(all_entropy) if all_entropy else 0,
            'glcm_correlation': np.mean(all_correlation) if all_correlation else 0
        }
        
        return features
    
    def _extract_lbp_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        masked_roi = np.where(mask > 0, roi, 0)
        
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(masked_roi, n_points, radius, method='uniform')
        lbp_values = lbp[mask > 0]
        
        if len(lbp_values) == 0:
            return {'lbp_uniformity': 0, 'lbp_entropy': 0}
        
        hist, _ = np.histogram(lbp_values, bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)  # Normalize

        features = {
            'lbp_uniformity': np.sum(hist ** 2),
            'lbp_entropy': -np.sum(hist * np.log2(hist + 1e-10))
        }
        return features
    
    def _extract_wavelet_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        masked_roi = np.where(mask > 0, roi, 0).astype(float)
        
        try:
            coeffs = pywt.dwt2(masked_roi, 'db4')
            cA, (cH, cV, cD) = coeffs
            features = {
                'wavelet_energy_ll': np.sum(cA ** 2),
                'wavelet_energy_lh': np.sum(cH ** 2),
                'wavelet_energy_hl': np.sum(cV ** 2),
                'wavelet_energy_hh': np.sum(cD ** 2)
            }
            
            total_energy = sum(features.values())
            if total_energy > 0:
                for key in features:
                    features[key] /= total_energy
                    
        except:
            features = {
                'wavelet_energy_ll': 0,
                'wavelet_energy_lh': 0,
                'wavelet_energy_hl': 0,
                'wavelet_energy_hh': 0
            }
        return features
    
    def extract_all_features(self, mammogram_path: str, mask_path: str = None) -> Dict[str, float]:
        if mask_path is not None:
            mammogram, mask = self.load_images(mammogram_path, mask_path)
            roi = self.extract_roi(mammogram, mask)
            texture_features = self.extract_texture_features(roi, mask)
            all_features = {**texture_features}

        else:
            mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)
            if mammogram is None:
                raise ValueError(f"Could not load mammogram from {mammogram_path}")
            
            texture_features = self.extract_global_texture_features(mammogram)
            shape_features = {key: 0.0 for key in ['compactness', 'circularity', 'irregularity', 
                                   'spiculation_index', 'fractal_dimension', 
                                   'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                                   'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7']}
        
            all_features = {**shape_features, **texture_features}
        
        for key, value in all_features.items():
            if np.isnan(value) or np.isinf(value):
                all_features[key] = 0.0
        return all_features
    
    def process_dataset(self, dataset_path: str, output_csv: str = "mammography_features.csv", 
                       process_without_masks: bool = True) -> pd.DataFrame:
        dataset_path = Path(dataset_path)
        mammogram_dir = dataset_path / "TIFF/"
        mask_dir = dataset_path / "Mask/"
        
        if not mammogram_dir.exists() or not mask_dir.exists():
            raise ValueError(f"Expected 'TIFF' and 'Mask' subdirectories in {dataset_path}")
        
        results = []        
        mammogram_files = list(mammogram_dir.glob("*.png")) + list(mammogram_dir.glob("*.jpg"))
        
        print(f"Processing {len(mammogram_files)} mammogram files...")
        
        for i, mammogram_file in enumerate(mammogram_files):
            try:
                # Check if mask exists
                mask_file = None
                if mask_dir.exists():
                    potential_mask = mask_dir / mammogram_file.name
                    if potential_mask.exists():
                        mask_file = potential_mask
                
                if mask_file is not None:
                    # Process with mask
                    features = self.extract_all_features(str(mammogram_file), str(mask_file))
                    features['has_mask'] = True
                    features['mask_path'] = str(mask_file)
                    print(f"Processed {mammogram_file.name} with mask")
                elif process_without_masks:
                    # Process without mask
                    features = self.extract_all_features(str(mammogram_file), None)
                    features['has_mask'] = False
                    features['mask_path'] = None
                    print(f"Processed {mammogram_file.name} without mask")
                else:
                    print(f"Skipping {mammogram_file.name} (no mask found)")
                    continue
                
                # Add metadata
                features['filename'] = mammogram_file.name
                features['mammogram_path'] = str(mammogram_file)
                
                results.append(features)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(mammogram_files)} files")
                    
            except Exception as e:
                print(f"Error processing {mammogram_file.name}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0.0)
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        return df
    
    def visualize_features(self, mammogram_path: str, mask_path: str = None, save_path: str = None):
        if mask_path is not None:
            mammogram, mask = self.load_images(mammogram_path, mask_path)
            roi = self.extract_roi(mammogram, mask)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original mammogram
            axes[0].imshow(mammogram, cmap='gray')
            axes[0].set_title('Original Mammogram')
            axes[0].axis('off')
            
            # ROI mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('ROI Mask')
            axes[1].axis('off')
            
            # ROI
            axes[2].imshow(roi, cmap='gray')
            axes[2].set_title('Extracted ROI')
            axes[2].axis('off')
        else:
            # Visualize without mask
            mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)
            processed = self._preprocess_mammogram(mammogram)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original mammogram
            axes[0].imshow(mammogram, cmap='gray')
            axes[0].set_title('Original Mammogram')
            axes[0].axis('off')
            
            # Preprocessed mammogram
            axes[1].imshow(processed, cmap='gray')
            axes[1].set_title('Preprocessed Mammogram')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    extractor = MammographyFeatureExtractor()
    
    # try:
    #     mammogram_path = "../MIAS Dataset/MIAS/mdb001.png"
    #     mask_path = "../Detector/single_predictions/mdb001.png"
        
    #     features = extractor.extract_all_features(mammogram_path, mask_path)
        
    #     print("Extracted Features:")
    #     for feature_name, value in features.items():
    #         print(f"{feature_name}: {value:.4f}")
        
    #     # Visualize
    #     extractor.visualize_features(mammogram_path, mask_path)
        
    # except Exception as e:
    #     print(f"Single image processing failed: {e}")
    try:
    
        dataset_path = "../DMID _Dataset/512"
        df = extractor.process_dataset(dataset_path, "dmid_features_all_no_nan.csv", process_without_masks=True)
        
        print(f"\nDataset processing complete!")
        print(f"Shape: {df.shape}")
        print(f"Images with masks: {df['has_mask'].sum()}")
        print(f"Images without masks: {(~df['has_mask']).sum()}")
        print(f"Features: {df.columns.tolist()}")
        
        nan_count = df.isnull().sum().sum()
        print(f"\nTotal NaN values in dataset: {nan_count}")
        
        if nan_count > 0:
            print("Columns with NaN values:")
            nan_columns = df.isnull().sum()
            print(nan_columns[nan_columns > 0])
        
        print("\nFeature Summary:")
        numeric_df = df.select_dtypes(include=[np.number])
        print(numeric_df.describe())
        
        print("\nFeatures by mask presence:")
        mask_features = df[df['has_mask'] == True].select_dtypes(include=[np.number])
        no_mask_features = df[df['has_mask'] == False].select_dtypes(include=[np.number])
        
        print(f"Features available with mask: {mask_features.shape[1]}")
        print(f"Features available without mask: {no_mask_features.shape[1]}")
        
    except Exception as e:
        print(f"Dataset processing failed: {e}")

if __name__ == "__main__":
    main()
