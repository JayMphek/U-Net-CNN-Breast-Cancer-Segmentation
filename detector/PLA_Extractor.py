import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LesionAnnotation:
    contour: np.ndarray
    area: float
    center: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  
    confidence: float = 1.0
    lesion_id: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'contour': self.contour.tolist() if isinstance(self.contour, np.ndarray) else self.contour,
            'area': float(self.area),
            'center': self.center,
            'bounding_box': self.bounding_box,
            'confidence': float(self.confidence),
            'lesion_id': int(self.lesion_id)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LesionAnnotation':
        data = data.copy()
        if isinstance(data['contour'], list):
            data['contour'] = np.array(data['contour'])
        return cls(**data)

@dataclass
class MammogramSample:
    sample_id: str
    mammogram_path: str
    roi_mask_path: str
    mammogram_image: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None
    lesion_annotations: List[LesionAnnotation] = None
    
    def __post_init__(self):
        if self.lesion_annotations is None:
            self.lesion_annotations = []

class MammogramLesionDetectionModel:
    def __init__(self, 
                 min_lesion_area: int = 50, 
                 max_lesion_area: int = 50000,
                 image_size: Optional[Tuple[int, int]] = None,
                 model_name: str = "MammogramLesionDetector",
                 version: str = "1.0"):
        
        self.model_name = model_name
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.min_lesion_area = min_lesion_area
        self.max_lesion_area = max_lesion_area
        self.image_size = image_size
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        self.samples: Dict[str, MammogramSample] = {}
        self.processing_history = []
        self.performance_metrics = {}
        
        logger.info(f"Initialized {self.model_name} v{self.version}")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'parameters': {
                'min_lesion_area': self.min_lesion_area,
                'max_lesion_area': self.max_lesion_area,
                'image_size': self.image_size
            },
            'sample_count': len(self.samples),
            'processing_history_count': len(self.processing_history),
            'performance_metrics': self.performance_metrics
        }
    
    def load_image(self, image_path: str, as_grayscale: bool = True) -> np.ndarray:
        if as_grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)
        
        return img
    
    def load_sample(self, sample_id: str, load_images: bool = True) -> MammogramSample:
        if sample_id not in self.samples:
            raise KeyError(f"Sample {sample_id} not found")
        
        sample = self.samples[sample_id]
        
        if load_images:
            sample.mammogram_image = self.load_image(sample.mammogram_path, as_grayscale=True)
            sample.roi_mask = self.load_image(sample.roi_mask_path, as_grayscale=True)
            sample.roi_mask = (sample.roi_mask > 127).astype(np.uint8) * 255
        
        return sample
    
    def apply_morphological_operations(self, mask: np.ndarray, 
                                     kernel_size: int = 3, 
                                     operation: str = 'close') -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            cleaned_mask = cv2.dilate(mask, kernel, iterations=1)
        elif operation == 'erode':
            cleaned_mask = cv2.erode(mask, kernel, iterations=1)
        else:
            cleaned_mask = mask
        
        return cleaned_mask
    
    def detect_lesions_from_mask(self, roi_mask: np.ndarray) -> List[LesionAnnotation]:
        lesion_annotations = []
        
        if roi_mask.dtype != np.uint8:
            roi_mask = (roi_mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lesion_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_lesion_area <= area <= self.max_lesion_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                x, y, w, h = cv2.boundingRect(contour)
                
                annotation = LesionAnnotation(
                    contour=contour,
                    area=area,
                    center=(cx, cy),
                    bounding_box=(x, y, w, h),
                    confidence=1.0,
                    lesion_id=lesion_id
                )
                
                lesion_annotations.append(annotation)
                lesion_id += 1
        
        return lesion_annotations
    
    def predict(self, sample_id: str, 
                clean_mask: bool = True,
                morphology_operation: str = 'close') -> MammogramSample:
        sample = self.load_sample(sample_id)
        
        roi_mask = sample.roi_mask.copy()
        if clean_mask:
            roi_mask = self.apply_morphological_operations(
                roi_mask, operation=morphology_operation
            )
        
        lesion_annotations = self.detect_lesions_from_mask(roi_mask)
        sample.lesion_annotations = lesion_annotations
        
        logger.info(f"Sample {sample_id}: Found {len(lesion_annotations)} lesions")
        
        return sample
    
    def create_annotated_image(self, sample: MammogramSample, 
                              annotation_colors: List[Tuple[int, int, int]] = None,
                              show_contours: bool = True,
                              show_centers: bool = True,
                              show_bounding_boxes: bool = False,
                              contour_thickness: int = 2) -> np.ndarray:
        if len(sample.mammogram_image.shape) == 2:
            annotated_image = cv2.cvtColor(sample.mammogram_image, cv2.COLOR_GRAY2RGB)
        else:
            annotated_image = sample.mammogram_image.copy()
        
        if annotation_colors is None:
            annotation_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]
        
        for i, annotation in enumerate(sample.lesion_annotations):
            color = annotation_colors[i % len(annotation_colors)]
            
            if show_contours:
                cv2.drawContours(annotated_image, [annotation.contour], -1, color, contour_thickness)
            
            if show_centers:
                cv2.circle(annotated_image, annotation.center, 5, color, -1)
                cv2.putText(annotated_image, f"L{annotation.lesion_id}", 
                           (annotation.center[0] + 10, annotation.center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if show_bounding_boxes:
                x, y, w, h = annotation.bounding_box
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 1)
        
        return annotated_image
    
    def create_mask_overlay(self, sample: MammogramSample, 
                           mask_alpha: float = 0.3,
                           mask_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        if len(sample.mammogram_image.shape) == 2:
            base_image = cv2.cvtColor(sample.mammogram_image, cv2.COLOR_GRAY2RGB)
        else:
            base_image = sample.mammogram_image.copy()
        
        mask_colored = np.zeros_like(base_image)
        mask_colored[sample.roi_mask > 0] = mask_color
        
        overlay_image = cv2.addWeighted(base_image, 1 - mask_alpha, mask_colored, mask_alpha, 0)
        
        return overlay_image
    
    def visualize_sample(self, sample: MammogramSample, 
                        save_path: Optional[str] = None,
                        show_plot: bool = True,
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Lesion Detection Results - Sample: {sample.sample_id}', fontsize=16)
        
        # Original mammogram
        axes[0, 0].imshow(sample.mammogram_image, cmap='gray')
        axes[0, 0].set_title('Original Mammogram')
        axes[0, 0].axis('off')
        
        # ROI mask
        axes[0, 1].imshow(sample.roi_mask, cmap='gray')
        axes[0, 1].set_title('ROI Mask')
        axes[0, 1].axis('off')
        
        # Mask overlay
        mask_overlay = self.create_mask_overlay(sample)
        axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title('ROI Mask Overlay')
        axes[1, 0].axis('off')
        
        # Lesion annotations
        annotated_image = self.create_annotated_image(sample)
        axes[1, 1].imshow(annotated_image)
        axes[1, 1].set_title(f'Lesion Annotations ({len(sample.lesion_annotations)} lesions)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_sample_images(self, sample: MammogramSample, output_dir: str,
                          save_formats: List[str] = ['annotated', 'overlay', 'comparison'],
                          image_format: str = 'png') -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save annotated image
        if 'annotated' in save_formats:
            annotated_img = self.create_annotated_image(sample)
            annotated_path = output_dir / f"{sample.sample_id}_annotated.{image_format}"
            cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            saved_paths['annotated'] = str(annotated_path)
        
        # Save overlay image
        if 'overlay' in save_formats:
            overlay_img = self.create_mask_overlay(sample)
            overlay_path = output_dir / f"{sample.sample_id}_overlay.{image_format}"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            saved_paths['overlay'] = str(overlay_path)
        
        # Save comparison visualization
        if 'comparison' in save_formats:
            comparison_path = output_dir / f"{sample.sample_id}_comparison.{image_format}"
            self.visualize_sample(sample, save_path=str(comparison_path), show_plot=False)
            saved_paths['comparison'] = str(comparison_path)
        
        # Save original images for reference
        if 'original' in save_formats:
            orig_mammo_path = output_dir / f"{sample.sample_id}_original.{image_format}"
            cv2.imwrite(str(orig_mammo_path), sample.mammogram_image)
            saved_paths['original'] = str(orig_mammo_path)
            
            orig_mask_path = output_dir / f"{sample.sample_id}_mask.{image_format}"
            cv2.imwrite(str(orig_mask_path), sample.roi_mask)
            saved_paths['mask'] = str(orig_mask_path)
        
        return saved_paths
    
    def process_single_image(self, 
                         mammogram_path: str, 
                         mask_path: str, 
                         sample_id: Optional[str] = None,
                         clean_mask: bool = True,
                         morphology_operation: str = 'close',
                         return_sample: bool = True) -> MammogramSample:
        if sample_id is None:
            sample_id = Path(mammogram_path).stem

        logger.info(f"Processing single image: {sample_id}")

        mammogram_path = Path(mammogram_path)
        mask_path = Path(mask_path)

        if not mammogram_path.exists():
            raise FileNotFoundError(f"Mammogram image not found: {mammogram_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"ROI mask not found: {mask_path}")

        # Load images
        try:
            mammogram_image = self.load_image(str(mammogram_path), as_grayscale=True)
            roi_mask = self.load_image(str(mask_path), as_grayscale=True)
            roi_mask = (roi_mask > 127).astype(np.uint8) * 255

        except Exception as e:
            raise ValueError(f"Failed to load images: {e}")

        sample = MammogramSample(
            sample_id=sample_id,
            mammogram_path=str(mammogram_path),
            roi_mask_path=str(mask_path),
            mammogram_image=mammogram_image,
            roi_mask=roi_mask
        )

        processed_mask = roi_mask.copy()
        if clean_mask:
            processed_mask = self.apply_morphological_operations(
                processed_mask, operation=morphology_operation
            )

        lesion_annotations = self.detect_lesions_from_mask(processed_mask)
        sample.lesion_annotations = lesion_annotations

        self.samples[sample_id] = sample

        logger.info(f"Sample {sample_id}: Found {len(lesion_annotations)} lesions")
        for i, annotation in enumerate(lesion_annotations, 1):
            logger.info(f"  Lesion {i}: Area={annotation.area:.1f}, Center={annotation.center}")

        processing_record = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'mammogram_path': str(mammogram_path),
            'mask_path': str(mask_path),
            'lesions_found': len(lesion_annotations),
            'parameters': {
                'clean_mask': clean_mask,
                'morphology_operation': morphology_operation
            }
        }
        self.processing_history.append(processing_record)
        self.last_updated = processing_record['timestamp']

        if return_sample:
            return sample

        return sample

    def create_single_image_visualization(self, 
                                        sample: MammogramSample,
                                        save_path: Optional[str] = None,
                                        show_plot: bool = True,
                                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Lesion Detection - {sample.sample_id} ({len(sample.lesion_annotations)} lesions found)', 
                     fontsize=14, fontweight='bold')

        # Original mammogram
        axes[0].imshow(sample.mammogram_image, cmap='gray')
        axes[0].set_title('Original Mammogram', fontsize=12)
        axes[0].axis('off')

        # ROI mask overlay
        mask_overlay = self.create_mask_overlay(sample, mask_alpha=0.4)
        axes[1].imshow(mask_overlay)
        axes[1].set_title('ROI Mask Overlay', fontsize=12)
        axes[1].axis('off')

        # Lesion annotations
        annotated_image = self.create_annotated_image(
            sample, 
            show_contours=True, 
            show_centers=True, 
            show_bounding_boxes=False
        )
        axes[2].imshow(annotated_image)
        axes[2].set_title('Detected Lesions', fontsize=12)
        axes[2].axis('off')

        # Add lesion information as text
        if sample.lesion_annotations:
            info_text = "Lesion Details:\n"
            for i, ann in enumerate(sample.lesion_annotations, 1):
                info_text += f"L{i}: Area={ann.area:.0f}px²\n"

            # Add text box with lesion info
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Single image visualization saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def process_and_save_single_image(self,
                                     mammogram_path: str,
                                     mask_path: str,
                                     output_dir: str,
                                     sample_id: Optional[str] = None,
                                     save_formats: List[str] = ['annotated', 'overlay', 'visualization'],
                                     clean_mask: bool = True,
                                     show_visualization: bool = False) -> Dict[str, Any]:
        # Process the image
        sample = self.process_single_image(
            mammogram_path=mammogram_path,
            mask_path=mask_path,
            sample_id=sample_id,
            clean_mask=clean_mask
        )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'sample_id': sample.sample_id,
            'lesions_found': len(sample.lesion_annotations),
            'lesion_details': [ann.to_dict() for ann in sample.lesion_annotations],
            'saved_files': {}
        }

        # Save different image formats
        if save_formats:
            saved_paths = self.save_sample_images(
                sample, 
                str(output_dir), 
                save_formats=save_formats
            )
            results['saved_files'].update(saved_paths)

        # Create and save focused visualization
        if 'visualization' in save_formats:
            viz_path = output_dir / f"{sample.sample_id}_detection_results.png"
            self.create_single_image_visualization(
                sample, 
                save_path=str(viz_path), 
                show_plot=show_visualization
            )
            results['saved_files']['visualization'] = str(viz_path)

        # Save JSON annotation
        json_path = output_dir / f"{sample.sample_id}_lesion_data.json"
        annotation_data = {
            'sample_id': sample.sample_id,
            'mammogram_path': mammogram_path,
            'mask_path': mask_path,
            'processing_timestamp': datetime.now().isoformat(),
            'lesion_count': len(sample.lesion_annotations),
            'lesions': [ann.to_dict() for ann in sample.lesion_annotations],
            'model_info': {
                'model_name': self.model_name,
                'version': self.version,
                'parameters': {
                    'min_lesion_area': self.min_lesion_area,
                    'max_lesion_area': self.max_lesion_area,
                    'clean_mask': clean_mask
                }
            }
        }

        with open(json_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        results['saved_files']['json_annotation'] = str(json_path)

        logger.info(f"Single image processing completed for {sample.sample_id}")
        logger.info(f"Found {len(sample.lesion_annotations)} lesions")
        logger.info(f"Results saved to {output_dir}")

        return results

def demo_single_image_processing():
    print("🔬 SINGLE IMAGE LESION DETECTION DEMO")
    print("=" * 50)
    
    model = MammogramLesionDetectionModel(
        min_lesion_area=50,
        max_lesion_area=50000,
        model_name="SingleImageDetector",
        version="1.0"
    )
    
    mammogram_path = "../MIAS Dataset/MIAS/mdb300.png"
    mask_path = "./single_predictions/mdb300.png"
    output_dir = "./single_image_results/"
    
    try:
        print("\n📊 Method 1: Simple Processing")
        sample = model.process_single_image(
            mammogram_path=mammogram_path,
            mask_path=mask_path,
            sample_id="my_sample"
        )
        print(f"✅ Found {len(sample.lesion_annotations)} lesions")
        
        print("\n📊 Method 2: Complete Workflow")
        results = model.process_and_save_single_image(
            mammogram_path=mammogram_path,
            mask_path=mask_path,
            output_dir=output_dir,
            sample_id="complete_workflow_sample",
            save_formats=['annotated', 'overlay', 'visualization'],
            show_visualization=True
        )
        
        print("✅ Processing completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🔍 Files created:")
        for file_type, path in results['saved_files'].items():
            print(f"   {file_type}: {path}")
            
    except FileNotFoundError as e:
        print(f"⚠️  File not found: {e}")
        print("Please update the file paths in the demo function")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_single_image_processing()
