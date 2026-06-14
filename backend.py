from flask import Flask, request, jsonify, send_from_directory, send_file, redirect
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import time
import socket
from typing import Dict, Any, Tuple
import logging
from werkzeug.utils import secure_filename
import traceback
import cv2
import sys
import mysql.connector
from db_init import DatabaseIntegratedAnalysisService

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from detector.ROI_Mask import predict_single_image_roi
    ROI_MASK_AVAILABLE = True
    print("✅ ROI_Mask module imported successfully")
except ImportError as e:
    ROI_MASK_AVAILABLE = False
    print(f"⚠️ ROI_Mask module not available: {e}")

try:
    from classifier.Classifier import load_model, predict_single_image
    CLASSIFIER_MODULE_AVAILABLE = True
    print("✅ Classifier module imported successfully (from classifier folder)")
except ImportError as e:
    CLASSIFIER_MODULE_AVAILABLE = False
    print(f"⚠️ Classifier module not available: {e}")

try:
    from classifier.FeatureExtractor import MammographyFeatureExtractor
    FEATURE_MODULE_AVAILABLE = True
    print("✅ FeatureExtractor module imported successfully (from classifier folder)")
except ImportError as e:
    FEATURE_MODULE_AVAILABLE = False
    print(f"⚠️ FeatureExtractor module not available: {e}")

try:
    from detector.PLA_Extractor import MammogramLesionDetectionModel
    PLA_EXTRACTOR_AVAILABLE = True
    print("✅ PLA Extractor module imported successfully")
except ImportError as e:
    PLA_EXTRACTOR_AVAILABLE = False
    print(f"⚠️ PLA Extractor module not available: {e}")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODELS_FOLDER'] = './models/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

def find_free_port(start_port=5501, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                logger.info(f"Found free port: {port}")
                return port
        except OSError as e:
            logger.warning(f"Port {port} is not available: {e}")
            continue
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))  
            port = s.getsockname()[1]
            logger.info(f"Using random available port: {port}")
            return port
    except Exception as e:
        logger.error(f"Could not find any available port: {e}")
        return None

def print_norm_classification_summary(result):
    """Helper function to print normal classification summary"""
    print(f"📊 Classification Result: {result['abnormality_type']}")
    print(f"🎯 Confidence: {result['abnormality_confidence']:.3f}")
    print(f"📈 Severity: {result['severity_class']}")
    print(f"💡 Reason: {result['classification_reason']}")

class MammographyAnalysisService:
    def __init__(self):
        self.roi_checkpoint_path = None
        self.pla_extractor = None
        self.classifier = None
        self.classifier_model = None  
        self.feature_extractor = None
        logger.info("MammographyAnalysisService initialized")
    
    def load_models(self):
        try:
            roi_checkpoint_path = os.path.join(app.config['MODELS_FOLDER'], 'unet_final_model.pkl')
            if os.path.exists(roi_checkpoint_path) and ROI_MASK_AVAILABLE:
                self.roi_checkpoint_path = roi_checkpoint_path
                logger.info(f"ROI model configuration loaded successfully from {roi_checkpoint_path}")
            else:
                logger.warning(f"ROI model setup failed - Checkpoint exists: {os.path.exists(roi_checkpoint_path)}, Module available: {ROI_MASK_AVAILABLE}")
                self.roi_checkpoint_path = None
            
            if PLA_EXTRACTOR_AVAILABLE:
                try:
                    self.pla_extractor = MammogramLesionDetectionModel(
                        min_lesion_area=50,
                        max_lesion_area=50000,
                        image_size=(512, 512),
                        model_name="MyMammogramDetector",
                        version="1.0"
                    )
                    logger.info("PLA extractor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize PLA extractor: {e}")
                    self.pla_extractor = None
            else:
                logger.warning("PLA Extractor module not available")
        
            if CLASSIFIER_MODULE_AVAILABLE and FEATURE_MODULE_AVAILABLE:
                try:
                    self.classifier = load_model()
                    self.feature_extractor = MammographyFeatureExtractor()
                    model_path = os.path.join(app.config['MODELS_FOLDER'], 'mammography_classifier.pkl')
                    
                    if self.classifier is not None and os.path.exists(model_path):
                        logger.info(f"Classifier loaded successfully from {model_path}")
                    else:
                        logger.warning(f"Classifier model file not found: {model_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to load classifier: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self.classifier = None
                    self.classifier_model = None
            else:
                logger.warning(f"Classifier modules not available - Classifier: {CLASSIFIER_MODULE_AVAILABLE}, FeatureExtractor: {FEATURE_MODULE_AVAILABLE}")
                self.classifier = None
                self.classifier_model = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
    def validate_model_configuration(self):
        validation_results = {
            'roi_model': False,
            'pla_extractor': False,
            'classifier': False
        }
        
        if (self.roi_checkpoint_path is not None and 
            os.path.exists(self.roi_checkpoint_path) and 
            ROI_MASK_AVAILABLE):
            validation_results['roi_model'] = True
            logger.info("✅ ROI model configuration validated")
        else:
            logger.warning("❌ ROI model configuration invalid")
        
        if self.pla_extractor is not None and PLA_EXTRACTOR_AVAILABLE:
            validation_results['pla_extractor'] = True
            logger.info("✅ PLA extractor configuration validated")
        else:
            logger.warning("❌ PLA extractor configuration invalid")
        
        if (self.classifier is not None and 
            CLASSIFIER_MODULE_AVAILABLE and 
            FEATURE_MODULE_AVAILABLE):
            validation_results['classifier'] = True
            logger.info("✅ Classifier configuration validated")
        else:
            logger.warning("❌ Classifier configuration invalid")
        
        return validation_results

    def preprocess_image_for_roi(self, image_file, target_size: Tuple[int, int] = (512, 512)) -> Tuple[str, Tuple[int, int]]:
        try:
            logger.debug("Preparing image for ROI analysis")
            timestamp = int(time.time())
            temp_filename = f"temp_roi_input_{timestamp}.png"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            image = Image.open(image_file)
            
            original_size = image.size  # (width, height)
            logger.debug(f"Original image dimensions: {original_size}")
            
            if image.mode != 'L':
                image = image.convert('L')
            
            image_resized = image.resize(target_size, Image.LANCZOS)
            image_resized.save(temp_path)
            logger.debug(f"Image prepared for ROI analysis: {temp_path}, resized to: {target_size}")
            return temp_path, original_size
            
        except Exception as e:
            logger.error(f"Error preprocessing image for ROI: {e}")
            raise ValueError(f"Failed to preprocess image for ROI: {str(e)}")
    
    def resize_mask_to_original(self, mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        try:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            resized_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            logger.debug(f"Mask resized from {binary_mask.shape} to {resized_mask.shape}")
            logger.debug(f"Mask values: min={resized_mask.min()}, max={resized_mask.max()}")
            return resized_mask
            
        except Exception as e:
            logger.error(f"Error resizing mask: {e}")
            raise ValueError(f"Failed to resize mask: {str(e)}")
        
    def generate_roi_mask_new(self, image_file) -> Dict[str, Any]:
        try:
            if not ROI_MASK_AVAILABLE:
                logger.warning("ROI_Mask module not available")
                return None

            if self.roi_checkpoint_path is None:
                logger.warning("ROI model not properly configured")
                return None
            
            if not os.path.exists(self.roi_checkpoint_path):
                logger.error(f"ROI checkpoint file not found: {self.roi_checkpoint_path}")
                return None
            
            logger.debug("Generating ROI mask using predict_single_image_roi")
            temp_image_path, original_size = self.preprocess_image_for_roi(image_file)
            
            try:
                roi_output_dir = os.path.join(app.config['RESULTS_FOLDER'], 'roi_predictions')
                os.makedirs(roi_output_dir, exist_ok=True)
                
                try:
                    roi_result = predict_single_image_roi(
                        image_path=temp_image_path,
                        model_params=None,
                        target_size=(512, 512),
                        checkpoint_path=self.roi_checkpoint_path,
                        preprocess=True,
                        save_prediction=True,
                        output_dir=roi_output_dir,
                        visualization=True
                    )
                    
                    logger.debug(f"ROI prediction completed: {roi_result}")
                    
                    if 'prediction_mask' in roi_result:
                        roi_mask = roi_result['prediction_mask']
                        
                        if isinstance(roi_mask, str) and os.path.exists(roi_mask):
                            roi_mask = cv2.imread(roi_mask, cv2.IMREAD_GRAYSCALE)
                        elif not isinstance(roi_mask, np.ndarray):
                            logger.warning("Unable to extract ROI mask from prediction result")
                            return None
                        
                        resized_binary_mask = self.resize_mask_to_original(roi_mask, original_size)
                        
                        roi_result['processed_mask'] = resized_binary_mask
                        roi_result['original_size'] = original_size
                        roi_result['model_processing_size'] = (512, 512)
                        roi_result['mask_resized'] = True
                        roi_result['is_binary'] = True
                        roi_result['method'] = 'predict_single_image_roi'
                        roi_result['success'] = True
                        return roi_result
                    else:
                        logger.warning("No prediction found in ROI result")
                        return None
                        
                except Exception as prediction_error:
                    logger.error(f"ROI prediction function failed: {prediction_error}")
                    logger.error(f"Prediction traceback: {traceback.format_exc()}")
                    return None
                    
            finally:
                try:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                        logger.debug(f"Cleaned up temporary file: {temp_image_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"Error in ROI mask generation: {e}")
            logger.error(f"ROI generation traceback: {traceback.format_exc()}")
            return None
    
    def generate_pla_analysis_new(self, mammogram_path: str, mask_path: str, sample_id: str = None) -> Dict[str, Any]:
        try:
            if not PLA_EXTRACTOR_AVAILABLE:
                raise ValueError("PLA extractor module not available")
                
            if self.pla_extractor is None:
                raise ValueError("PLA extractor not initialized")
            
            logger.debug("Generating PLA analysis using process_and_save_single_image")
            logger.debug(f"Mammogram path: {mammogram_path}")
            logger.debug(f"Mask path: {mask_path}")
            
            timestamp = int(time.time())
            pla_output_dir = os.path.join(app.config['RESULTS_FOLDER'], f'pla_analysis_{timestamp}')
            os.makedirs(pla_output_dir, exist_ok=True)
            
            if sample_id is None:
                sample_id = f"sample_{timestamp}"
            
            pla_result = self.pla_extractor.process_and_save_single_image(
                mammogram_path=mammogram_path,
                mask_path=mask_path,
                output_dir=pla_output_dir,
                sample_id=sample_id,
                save_formats=['annotated', 'overlay', 'visualization'],
                clean_mask=True,
                show_visualization=False
            )
            
            logger.debug(f"PLA analysis completed: {pla_result}")
            
            pla_result['output_directory'] = pla_output_dir
            pla_result['method'] = 'process_and_save_single_image'
            pla_result['sample_id'] = sample_id
            return pla_result
            
        except Exception as e:
            logger.error(f"Error generating PLA analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"PLA analysis failed: {str(e)}")
    
    # Classification method - Fixed
    def classify_mammography_new(self, mammogram_path: str, roi_mask_path: str = None) -> Dict[str, Any]:
        try:
            if not CLASSIFIER_MODULE_AVAILABLE or not FEATURE_MODULE_AVAILABLE:
                raise ValueError("Classifier or FeatureExtractor module not available")
                
            if self.classifier is None or self.feature_extractor is None:
                raise ValueError("Classifier or FeatureExtractor not initialized")
            
            logger.debug(f"Classifying mammography image")
            logger.debug(f"Mammogram path: {mammogram_path}")
            logger.debug(f"ROI mask path: {roi_mask_path}")

            if not hasattr(self.classifier, 'abnormality_classifier') or self.classifier.abnormality_classifier is None:
                logger.warning("Classifier not trained - providing fallback classification")
                return self._provide_fallback_classification(mammogram_path, roi_mask_path)
            
            result = predict_single_image(
                classifier=self.classifier,
                mammogram_path=mammogram_path, 
                mask_path=roi_mask_path,
            )
            
            logger.debug(f"Classification result: {result}")
            
            if 'error' in result:
                logger.error(f"Classification error: {result['error']}")
                return self._provide_fallback_classification(mammogram_path, roi_mask_path, error=result['error'])
            
            abnormality_type = result.get('abnormality_type', 'UNKNOWN')
            abnormality_confidence = result.get('abnormality_confidence', 0.5)
            severity_class = result.get('severity_class', 'N')
            severity_confidence = result.get('severity_confidence', 0.5)
            
            classification_reason = f"Detected {abnormality_type}"
            if severity_class in ['B', 'M']:
                classification_reason += f" with {severity_class} characteristics"
            
            abnormality_descriptions = {
                'NORM': 'Normal',
                'CIRC': 'Circumscribed Mass',
                'SPIC': 'Spiculated Mass',
                'MISC': 'Miscellaneous Mass',
                'ARCH': 'Architectural Distortion',
                'ASYM': 'Asymmetric Density',
                'FOCAL': 'Focal Asymmetric Density',
                'CALC': 'Calcification',
                'UNKNOWN': 'Unknown'
            }
            
            abnormality_description = abnormality_descriptions.get(abnormality_type, abnormality_type)
            
            if severity_class == 'B':
                prediction = 'Benign'
                risk_score = 1.0 - severity_confidence if severity_confidence > 0.5 else 0.3
            elif severity_class == 'M':
                prediction = 'Malignant' 
                risk_score = severity_confidence if severity_confidence > 0.5 else 0.7
            else:
                if abnormality_type == 'NORM':
                    prediction = 'Normal'
                    risk_score = 1.0 - abnormality_confidence
                else:
                    prediction = 'Abnormal'
                    risk_score = abnormality_confidence
            
            classification_result = {
                'prediction': prediction,
                'confidence': float(max(abnormality_confidence, severity_confidence)),
                'risk_score': float(risk_score),
                'abnormality_type': abnormality_type,
                'abnormality_description': abnormality_description,
                'abnormality_confidence': float(abnormality_confidence),
                'severity_class': severity_class,
                'severity_confidence': float(severity_confidence),
                'classification_reason': classification_reason,
                'has_mask': roi_mask_path is not None,
                'method': 'predict_from_images',
                'features_extracted': result.get('features_extracted', 0),
                'features_expected': result.get('features_expected', 0)
            }
            
            logger.debug(f"Processed classification result: {classification_result}")
            return classification_result
            
        except Exception as e:
            logger.error(f"Error in classification method: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._provide_fallback_classification(mammogram_path, roi_mask_path, error=str(e))
    
    def _provide_fallback_classification(self, mammogram_path: str, roi_mask_path: str = None, error: str = None) -> Dict[str, Any]:
        classification_reason = "Fallback classification due to model unavailability"
        if error:
            classification_reason += f": {error}"
            
        return {
            'prediction': 'Unknown',
            'confidence': 0.5,
            'risk_score': 0.5,
            'abnormality_type': 'UNKNOWN',
            'abnormality_description': 'Analysis unavailable',
            'abnormality_confidence': 0.5,
            'severity_class': 'N',
            'severity_confidence': 0.5,
            'classification_reason': classification_reason,
            'has_mask': roi_mask_path is not None,
            'method': 'fallback',
            'error': error or 'Classifier not available',
            'features_extracted': 0,
            'features_expected': 0
        }
    
    def save_visualization(self, image_array: np.ndarray, filename: str, colormap=None, is_binary_mask=False) -> str:
        try:
                if is_binary_mask:
                    binary_mask = np.where(image_array > 127, 255, 0).astype(np.uint8)
                    image_to_save = Image.fromarray(binary_mask, mode='L')
                    logger.debug(f"Saving binary mask with values: min={binary_mask.min()}, max={binary_mask.max()}")
                elif colormap:
                    colored_image = cv2.applyColorMap(image_array.astype(np.uint8), colormap)
                    image_to_save = Image.fromarray(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
                else:
                    image_to_save = Image.fromarray(image_array.astype(np.uint8))
                
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                image_to_save.save(filepath)
                
                logger.debug(f"Visualization saved: {filepath}")
                return filename
                
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return None   
         
    def estimate_lesion_size(self, roi_mask: np.ndarray) -> float:
        try:
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            largest_contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(largest_contour)
            pixels_per_mm = 2.0 
            area_mm2 = area_pixels / (pixels_per_mm ** 2)
            diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
            return round(diameter_mm, 1)
            
        except Exception as e:
            logger.error(f"Error estimating lesion size: {e}")
            return 0.0
    
    def analyze_mammography(self, image_file) -> Dict[str, Any]:
        try:
            start_time = time.time()
            available_methods = [
                self.roi_checkpoint_path is not None,
                self.pla_extractor is not None,
                self.classifier is not None
            ]
            
            if not any(available_methods):
                raise ValueError("No analysis models are loaded. Please ensure required files are available.")
            
            image_file.seek(0)
            timestamp = int(time.time())
            temp_image_filename = f"temp_analysis_{timestamp}.png"
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_image_filename)
            
            original_image = Image.open(image_file)
            original_size = original_image.size
            if original_image.mode != 'L':
                original_image = original_image.convert('L')
            original_image.save(temp_image_path)
            
            roi_mask = None
            roi_mask_path = None
            lesion_size = 0.0
            roi_viz_path = None
            
            try:
                if self.roi_checkpoint_path is not None:
                    image_file.seek(0)
                    roi_result = self.generate_roi_mask_new(image_file)
                    
                    if roi_result is not None:
                        roi_mask = roi_result['processed_mask']
                        roi_mask_filename = f"temp_roi_mask_{timestamp}.png"
                        roi_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], roi_mask_filename)
                        cv2.imwrite(roi_mask_path, roi_mask)                        
                        lesion_size = self.estimate_lesion_size(roi_mask)
                        roi_filename = f"roi_analysis_{timestamp}.png"
                        roi_filename = f"roi_analysis_{timestamp}.png"
                        roi_viz_path = self.save_visualization(
                            roi_mask, 
                            roi_filename, 
                            colormap=None,  # No colormap for binary mask
                            is_binary_mask=True
                        )
                
                pla_result = None
                pla_visualization_path = None
                if PLA_EXTRACTOR_AVAILABLE and self.pla_extractor is not None and roi_mask_path is not None:
                    try:
                        pla_result = self.generate_pla_analysis_new(
                            mammogram_path=temp_image_path,
                            mask_path=roi_mask_path,
                            sample_id=f"analysis_{timestamp}"
                        )
                        if 'saved_files' in pla_result and 'visualization' in pla_result['saved_files']:
                            pla_visualization_path = pla_result['saved_files']['visualization']
                        elif 'output_files' in pla_result:
                            for file_path in pla_result['output_files']:
                                if 'visualization' in file_path.lower():
                                    pla_visualization_path = file_path
                                    break
                                    
                    except Exception as pla_error:
                        logger.warning(f"PLA analysis failed: {pla_error}")
                        pla_result = {'error': str(pla_error), 'method': 'failed'}
                
                classification_result = None
                if CLASSIFIER_MODULE_AVAILABLE and FEATURE_MODULE_AVAILABLE and self.classifier is not None:
                    try:
                        classification_result = self.classify_mammography_new(
                            mammogram_path=temp_image_path,
                            roi_mask_path=roi_mask_path
                        )
                    except Exception as class_error:
                        logger.warning(f"Classification failed: {class_error}")
                        classification_result = self._provide_fallback_classification(
                            temp_image_path, roi_mask_path, str(class_error)
                        )
                else:
                    classification_result = self._provide_fallback_classification(
                        temp_image_path, roi_mask_path, "Classifier not available"
                    )
                
                pla_viz_path = None
                if pla_visualization_path and os.path.exists(pla_visualization_path):
                    import shutil
                    pla_filename = f"pla_analysis_{timestamp}.png"
                    pla_viz_path = pla_filename
                    shutil.copy2(pla_visualization_path, os.path.join(app.config['RESULTS_FOLDER'], pla_filename))
                
                processing_time = round(time.time() - start_time, 2)
                
                results = {
                    'prediction': classification_result['prediction'],
                    'confidence': classification_result['confidence'],
                    'risk_score': classification_result['risk_score'],
                    'abnormality_type': classification_result['abnormality_type'],
                    'abnormality_description': classification_result['abnormality_description'],
                    'abnormality_confidence': classification_result['abnormality_confidence'],
                    'severity_class': classification_result['severity_class'],
                    'severity_confidence': classification_result['severity_confidence'],
                    'classification_reason': classification_result['classification_reason'],
                    
                    # Analysis metrics
                    'lesion_size_mm': lesion_size,
                    'roi_visualization_path': roi_viz_path,
                    'pla_visualization_path': pla_viz_path,
                    'processing_time': processing_time,
                    
                    # Analysis details
                    'classification_details': {
                        'method': classification_result.get('method', 'unknown'),
                        'has_mask': classification_result.get('has_mask', False),
                        'features_extracted': classification_result.get('features_extracted', 0),
                        'features_expected': classification_result.get('features_expected', 0),
                        'error': classification_result.get('error')
                    },
                    'pla_analysis_details': {
                        'method': 'process_and_save_single_image' if pla_result and 'error' not in pla_result else 'fallback',
                        'success': pla_result is not None and 'error' not in pla_result,
                        'output_directory': pla_result.get('output_directory') if pla_result else None,
                        'saved_files': pla_result.get('saved_files') if pla_result else None,
                        'sample_id': pla_result.get('sample_id') if pla_result else None
                    },
                    'roi_analysis_details': {
                        'method': 'predict_single_image_roi',
                        'success': roi_mask is not None,
                        'mask_generated': roi_mask is not None,
                        'visualization_saved': roi_viz_path is not None
                    },
                    'models_loaded': {
                        'roi_model': self.roi_checkpoint_path is not None and ROI_MASK_AVAILABLE,
                        'pla_extractor': self.pla_extractor is not None and PLA_EXTRACTOR_AVAILABLE,
                        'classifier': self.classifier is not None and CLASSIFIER_MODULE_AVAILABLE and FEATURE_MODULE_AVAILABLE
                    }
                }
                
                logger.info(f"Analysis completed successfully in {processing_time}s")
                return results
            
            finally:
                # Cleanup temporary files
                try:
                    if roi_mask_path and os.path.exists(roi_mask_path):
                        os.remove(roi_mask_path)
                        logger.debug(f"Cleaned up ROI mask file: {roi_mask_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup ROI mask file: {cleanup_error}")            
        except Exception as e:
            logger.error(f"Error during mammography analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

analysis_service = MammographyAnalysisService()
db_integrated_service = DatabaseIntegratedAnalysisService(analysis_service)

print("🔥 Initializing AI models and pipelines...")
try:
    analysis_service.load_models()
    validation_results = analysis_service.validate_model_configuration()
    
    print("\n📊 Model Initialization Results:")
    for model_name, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {model_name}: {status_icon} {'Loaded' if status else 'Failed/Missing'}")

    loaded_count = sum(validation_results.values())
    total_count = len(validation_results)
    print(f"\n📈 Summary: {loaded_count}/{total_count} models loaded successfully")
    
    if loaded_count == 0:
        print("⚠️  WARNING: No models loaded - server will run with limited functionality")
    elif loaded_count < total_count:
        print("⚠️  WARNING: Some models missing - partial functionality available")
    else:
        print("🎉 All models loaded successfully - full functionality available")
        
except Exception as e:
    print(f"❌ Error during model initialization: {e}")
    print("⚠️  Server will start but with limited functionality")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")

@app.route('/test')
def test_route():
    logger.info("Test route accessed")
    return "<h1>Flask Server is Working!</h1><p>This is a test route</p>"

@app.route('/')
def index():
    logger.info("Index route accessed")
    try:
        html_files = ['web_interface.html', 'analysis_history.html']
        for html_file in html_files:
            if os.path.exists(html_file):
                logger.info(f"Found {html_file}, serving it")
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"HTML file loaded, size: {len(content)} characters")
                return content
        
        logger.warning("No HTML files found, serving fallback HTML")
        return serve_fallback_html()
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"<h1>Error</h1><p>Error loading page: {str(e)}</p>", 500
    
@app.route('/web_interface.html')  
def web_interface():
    try:
        if os.path.exists('web_interface.html'):
            with open('web_interface.html', 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return redirect('/') 
    except Exception as e:
        logger.error(f"Error serving web_interface.html: {e}")
        return f"<h1>Error</h1><p>Error loading web interface: {str(e)}</p>", 500


@app.route('/analysis_history.html')
def analysis_history():
    try:
        if os.path.exists('analysis_history.html'):
            with open('analysis_history.html', 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return redirect('/') 
    except Exception as e:
        logger.error(f"Error serving analysis_history.html: {e}")
        return f"<h1>Error</h1><p>Error loading analysis history: {str(e)}</p>", 500

@app.route('/<path:filename>')
def serve_html_files(filename):
    try:
        if filename.endswith('.html'):
            file_path = os.path.join(os.getcwd(), filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                logger.info(f"Serving HTML file: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        logger.warning(f"File not found or not HTML: {filename}")
        return jsonify({
            'error': 'Not Found',
            'message': f'The file {filename} was not found or is not accessible.',
            'available_html_files': [f for f in os.listdir('.') if f.endswith('.html')]
        }), 404
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({
            'error': 'Server Error',
            'message': f'Error serving file {filename}: {str(e)}'
        }), 500

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    try:
        static_dirs = ['static', 'assets', 'css', 'js', 'images', '.']
        for static_dir in static_dirs:
            file_path = os.path.join(static_dir, filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                logger.info(f"Serving static file: {filename} from {static_dir}")
                return send_from_directory(static_dir, filename)
        logger.warning(f"Static file not found: {filename}")
        return jsonify({'error': 'Static file not found'}), 404
        
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({'error': 'Error serving static file'}), 500

def serve_fallback_html():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mammography Analysis - Fallback</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; }
            .nav-links { margin: 20px 0; }
            .nav-links a { margin-right: 15px; color: #007bff; text-decoration: none; }
            .nav-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mammography Analysis</h1>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/web_interface.html">Web Interface</a>
                <a href="/analysis_history.html">Analysis History</a>
            </div>
            <p><strong>Status:</strong> Server is running (fallback interface)</p>
            <p><strong>Note:</strong> Main HTML files not found in current directory</p>
            <div class="upload-area">
                <form action="/api/analyze" method="post" enctype="multipart/form-data">
                    <p>Upload Mammography Image:</p>
                    <input type="file" name="mammography" accept=".tiff,.tif,.dcm,.png,.jpg,.jpeg,.pgm" required>
                    <br><br>
                    <button type="submit">Analyze</button>
                </form>
            </div>
            <p><strong>Debug URLs:</strong></p>
            <ul>
                <li><a href="/test">Test Route</a></li>
                <li><a href="/api/debug">API Debug</a></li>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/models">Model Status</a></li>
            </ul>
        </div>
    </body>
    </html>
    '''

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_mammography():
    logger.info(f"Analyze route accessed with method: {request.method}")
    if request.method == 'OPTIONS':
        logger.info("Handling CORS preflight request")
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    try:
        logger.info(f"Request files: {list(request.files.keys())}")
        
        if 'mammography' not in request.files:
            logger.error("No 'mammography' file in request")
            return jsonify({
                'error': 'No file uploaded',
                'message': 'Please upload a mammography image file.'
            }), 400
        
        file = request.files['mammography']
        logger.info(f"Received file: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)  
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload.'
            }), 400
        
        allowed_extensions = {'.tiff', '.tif', '.dcm', '.png', '.jpg', '.jpeg', '.pgm'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Please upload a valid mammography image. Supported formats: {", ".join(allowed_extensions)}'
            }), 400
        
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                results = db_integrated_service.analyze_mammography_with_database(f, filename)

            results['filename'] = filename
            results['upload_time'] = timestamp
            results['file_size'] = os.path.getsize(file_path)
            logger.info(f"Analysis completed for {filename}")
            response = jsonify(results)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        finally:
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'details': 'Please check that model files are properly loaded and accessible.'
        }), 500
    
def add_database_endpoints(app, db_integrated_service):
    @app.route('/api/database/recent/<int:limit>')
    def get_recent_analyses(limit):
        try:
            with db_integrated_service.db_manager:
                recent = db_integrated_service.db_manager.get_recent_analyses(limit)
                return jsonify({
                    'status': 'success',
                    'analyses': recent
                })
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/database/mammogram/<int:mammogram_id>')
    def get_mammogram_details(mammogram_id):
        try:
            with db_integrated_service.db_manager:
                details = db_integrated_service.db_manager.get_analysis_results(mammogram_id)
                return jsonify({
                    'status': 'success',
                    'details': details
                })
        except Exception as e:
            logger.error(f"Error getting mammogram details: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/database/search', methods=['POST'])
    def search_mammograms():
        try:
            search_criteria = request.get_json()
            with db_integrated_service.db_manager:
                results = db_integrated_service.db_manager.search_mammograms(search_criteria)
                return jsonify({
                    'status': 'success',
                    'results': results
                })
        except Exception as e:
            logger.error(f"Error searching mammograms: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/database/export/<format>')
    def export_data(format):
        try:
            with db_integrated_service.db_manager:
                export_path = db_integrated_service.db_manager.export_analysis_data(format)
                if export_path:
                    filename = os.path.basename(export_path)
                    return jsonify({
                        'status': 'success',
                        'export_path': filename,  # Return just filename for download URL
                        'message': f'Data exported successfully to {format.upper()} format'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Export failed'
                    }), 500
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/database/file/<int:mammogram_id>/<file_type>')
    def get_analysis_file(mammogram_id, file_type):
        try:
            with db_integrated_service.db_manager:
                file_path = db_integrated_service.db_manager.get_analysis_file_path(mammogram_id, file_type)
                if file_path and os.path.exists(file_path):
                    return send_file(file_path)
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'File not found'
                    }), 404
        except Exception as e:
            logger.error(f"Error getting analysis file: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
@app.route('/api/database/statistics')
def get_database_statistics():
    try:
        with db_integrated_service.db_manager:
            stats = db_integrated_service.db_manager.get_statistics()
            return jsonify({
                'status': 'success',
                'statistics': stats
            })
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        # Extract filters
        filters = {}
        if request.args.get('prediction'):
            filters['prediction'] = request.args.get('prediction')
        if request.args.get('abnormality_type'):
            filters['abnormality_type'] = request.args.get('abnormality_type')
        if request.args.get('severity_class'):
            filters['severity_class'] = request.args.get('severity_class')
        if request.args.get('min_confidence'):
            filters['min_confidence'] = float(request.args.get('min_confidence'))
        if request.args.get('date_from'):
            filters['date_from'] = request.args.get('date_from')
        if request.args.get('date_to'):
            filters['date_to'] = request.args.get('date_to')
        
        with db_integrated_service.db_manager:
            if filters:
                filters['limit'] = limit
                analyses = db_integrated_service.db_manager.search_mammograms(filters)
                total = len(analyses)
            else:
                all_recent = db_integrated_service.db_manager.get_recent_analyses(limit * 10)  # Get more for pagination
                total = len(all_recent)
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                analyses = all_recent[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'analyses': analyses,
                'total': total,
                'page': page,
                'limit': limit
            })
            
    except Exception as e:
        logger.error(f"Error getting analyses: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/analyses/<int:analysis_id>')
def get_analysis_by_id(analysis_id):
    try:
        with db_integrated_service.db_manager:
            details = db_integrated_service.db_manager.get_analysis_results(analysis_id)
            
            if not details or not details.get('mammogram'):
                return jsonify({
                    'status': 'error',
                    'message': 'Analysis not found'
                }), 404
                
            return jsonify({
                'status': 'success',
                'details': details
            })
            
    except Exception as e:
        logger.error(f"Error getting analysis details: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def debug_analysis_data(analysis_id):
    try:
        logger.info(f"Debug request for analysis ID: {analysis_id}")
        
        with db_integrated_service.db_manager:
            details = db_integrated_service.db_manager.get_analysis_results(analysis_id)
            recent = db_integrated_service.db_manager.get_recent_analyses(20)
            matching_recent = None
            for analysis in recent:
                if analysis.get('id') == analysis_id:
                    matching_recent = analysis
                    break
            
            debug_info = {
                'requested_id': analysis_id,
                'details_found': details is not None,
                'details_has_mammogram': bool(details and details.get('mammogram')),
                'matching_recent_found': matching_recent is not None,
                'details_structure': {},
                'recent_structure': {},
                'comparison': {}
            }
            
            if details:
                debug_info['details_structure'] = {
                    'keys': list(details.keys()),
                    'mammogram_keys': list(details.get('mammogram', {}).keys()) if details.get('mammogram') else [],
                    'classification_keys': list(details.get('classification', {}).keys()) if details.get('classification') else [],
                    'lesions_count': len(details.get('lesions', [])),
                    'analysis_files_count': len(details.get('analysis_files', []))
                }
            
            if matching_recent:
                debug_info['recent_structure'] = {
                    'keys': list(matching_recent.keys()),
                    'id': matching_recent.get('id'),
                    'filename': matching_recent.get('original_filename') or matching_recent.get('filename'),
                    'prediction': matching_recent.get('prediction'),
                    'upload_timestamp': matching_recent.get('upload_timestamp')
                }
                
                if details and details.get('mammogram'):
                    mammogram = details['mammogram']
                    debug_info['comparison'] = {
                        'ids_match': mammogram.get('id') == matching_recent.get('id'),
                        'filenames_match': (mammogram.get('original_filename') or mammogram.get('filename')) == 
                                         (matching_recent.get('original_filename') or matching_recent.get('filename')),
                        'timestamps_match': mammogram.get('upload_timestamp') == matching_recent.get('upload_timestamp')
                    }
            
            if details:
                debug_info['sample_details'] = {
                    'mammogram_id': details.get('mammogram', {}).get('id'),
                    'mammogram_filename': details.get('mammogram', {}).get('original_filename') or details.get('mammogram', {}).get('filename'),
                    'classification_prediction': details.get('classification', {}).get('prediction') if details.get('classification') else None
                }
            
            return jsonify({
                'status': 'success',
                'debug_info': debug_info,
                'raw_details': details,  # Include full raw data for debugging
                'matching_recent': matching_recent
            })
            
    except Exception as e:
        logger.error(f"Error in debug analysis endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/database/visualization/<int:mammogram_id>/<file_type>')
def get_database_visualization(mammogram_id, file_type):
    try:
        logger.info(f"Database visualization requested: mammogram_id={mammogram_id}, file_type={file_type}")

        with db_integrated_service.db_manager:
            analysis_file = db_integrated_service.db_manager.get_analysis_file_by_type(mammogram_id, file_type)
            if not analysis_file:
                logger.warning(f"No analysis file found for mammogram_id={mammogram_id}, file_type={file_type}")
                all_files = db_integrated_service.db_manager.get_all_analysis_files(mammogram_id)
                logger.info(f"Available files for mammogram {mammogram_id}: {[f['file_type'] for f in all_files]}")
                return jsonify({
                    'error': 'Visualization not found',
                    'message': f'No {file_type} visualization found for analysis {mammogram_id}',
                    'available_types': [f['file_type'] for f in all_files] if all_files else []
                }), 404
            
            base_storage_path = os.path.join(db_integrated_service.db_manager.storage_root, 'analysis_files')
            possible_paths = [
                os.path.join(base_storage_path, analysis_file['filename']),
                os.path.join(base_storage_path, file_type, analysis_file['filename']),
                os.path.join(base_storage_path, 'roi_visualizations', analysis_file['filename']),
                os.path.join(base_storage_path, 'pla_visualizations', analysis_file['filename']),
                os.path.join(base_storage_path, analysis_file.get('file_path', '')) if analysis_file.get('file_path') != analysis_file['filename'] else None
            ]
            
            possible_paths = [p for p in possible_paths if p]
            logger.info(f"Searching for file in paths: {possible_paths}")
            
            for file_path in possible_paths:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    logger.info(f"Found visualization file: {file_path}")
                    return send_file(file_path)
            
            logger.error(f"Visualization file not found at any location for {analysis_file['filename']}")
            return jsonify({
                'error': 'File not found on disk',
                'message': f'Visualization file exists in database but not found on disk: {analysis_file["filename"]}',
                'searched_paths': possible_paths,
                'database_record': analysis_file
            }), 404
                
    except Exception as e:
        logger.error(f"Error getting database visualization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Database error',
            'message': str(e)
        }), 500

@app.route('/api/database/visualization/direct/<path:filename>')
def get_database_visualization_direct(filename):
    """Get visualization file directly from database storage using filename"""
    try:
        logger.info(f"Direct database visualization requested: {filename}")
        
        with db_integrated_service.db_manager:
            base_storage_path = os.path.join(db_integrated_service.db_manager.storage_root, 'analysis_files')
            possible_paths = [
                os.path.join(base_storage_path, filename),
                os.path.join(base_storage_path, 'roi_visualizations', filename),
                os.path.join(base_storage_path, 'pla_visualizations', filename),
                os.path.join(base_storage_path, 'visualizations', filename),
                os.path.join(db_integrated_service.db_manager.storage_root, filename)
            ]
            
            logger.info(f"Searching for direct file in paths: {possible_paths}")
            for file_path in possible_paths:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    logger.info(f"Found direct visualization file: {file_path}")
                    return send_file(file_path)
            logger.error(f"Direct visualization file not found: {filename}")

            available_files = []
            try:
                for root, dirs, files in os.walk(base_storage_path):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            rel_path = os.path.relpath(os.path.join(root, file), base_storage_path)
                            available_files.append(rel_path)
            except Exception as list_error:
                logger.warning(f"Could not list available files: {list_error}")
            
            return jsonify({
                'error': 'File not found',
                'message': f'Visualization file not found: {filename}',
                'searched_paths': possible_paths,
                'available_files': available_files[:10] if available_files else []  # Limit to first 10
            }), 404
        
    except Exception as e:
        logger.error(f"Error getting direct visualization: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/database/debug/files/<int:mammogram_id>')
def debug_analysis_files(mammogram_id):
    try:
        all_files = db_integrated_service.db_manager.get_all_analysis_files(mammogram_id)
        base_storage_path = os.path.join(db_integrated_service.db_manager.storage_root, 'analysis_files')
        file_status = []
        for file_record in all_files:
            possible_paths = [
                os.path.join(base_storage_path, file_record['filename']),
                os.path.join(base_storage_path, file_record['file_type'], file_record['filename']),
                os.path.join(base_storage_path, 'roi_visualizations', file_record['filename']),
                os.path.join(base_storage_path, 'pla_visualizations', file_record['filename'])
            ]
            exists_at = None
            for path in possible_paths:
                if os.path.exists(path):
                    exists_at = path
                    break
            
            file_status.append({
                'database_record': file_record,
                'exists_on_disk': exists_at is not None,
                'disk_path': exists_at,
                'searched_paths': possible_paths
            })
        
        storage_files = []
        try:
            for root, dirs, files in os.walk(base_storage_path):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_storage_path)
                        storage_files.append({
                            'filename': file,
                            'relative_path': rel_path,
                            'full_path': full_path,
                            'subdirectory': os.path.dirname(rel_path) if os.path.dirname(rel_path) else 'root'
                        })
        except Exception as e:
            logger.warning(f"Could not list storage files: {e}")
        
        return jsonify({
            'mammogram_id': mammogram_id,
            'database_files': file_status,
            'storage_directory': base_storage_path,
            'all_visualization_files': storage_files,
            'storage_root': db_integrated_service.db_manager.storage_root
        })
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return jsonify({
            'error': 'Debug error',
            'message': str(e)
        }), 500
            
add_database_endpoints(app, db_integrated_service)

@app.route('/api/debug/recent')
def debug_recent_analyses():
    try:
        with db_integrated_service.db_manager:
            recent = db_integrated_service.db_manager.get_recent_analyses(10)
            
            debug_info = {
                'total_count': len(recent),
                'analyses_summary': []
            }
            
            for analysis in recent:
                debug_info['analyses_summary'].append({
                    'id': analysis.get('id'),
                    'filename': analysis.get('original_filename') or analysis.get('filename'),
                    'prediction': analysis.get('prediction'),
                    'upload_timestamp': analysis.get('upload_timestamp'),
                    'all_keys': list(analysis.keys())
                })
            
            return jsonify({
                'status': 'success',
                'debug_info': debug_info,
                'raw_recent': recent[:3]  
            })
            
    except Exception as e:
        logger.error(f"Error in debug recent endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/models')
def model_status():
    logger.info("Model status endpoint accessed")
    
    models_status = {
        'roi_model': {
            'loaded': analysis_service.roi_checkpoint_path is not None and ROI_MASK_AVAILABLE,
            'method': 'predict_single_image_roi',
            'checkpoint_path': analysis_service.roi_checkpoint_path,
            'roi_mask_module_available': ROI_MASK_AVAILABLE,
            'exists': os.path.exists(analysis_service.roi_checkpoint_path) if analysis_service.roi_checkpoint_path else False
        },
        'pla_extractor': {
            'loaded': analysis_service.pla_extractor is not None and PLA_EXTRACTOR_AVAILABLE,
            'method': 'process_and_save_single_image',
            'module_available': PLA_EXTRACTOR_AVAILABLE,
            'class_initialized': analysis_service.pla_extractor is not None
        },
        'classifier': {
            'loaded': (analysis_service.classifier is not None and 
                      CLASSIFIER_MODULE_AVAILABLE and 
                      FEATURE_MODULE_AVAILABLE),
            'method': 'predict_from_images',
            'classifier_module_available': CLASSIFIER_MODULE_AVAILABLE,
            'feature_module_available': FEATURE_MODULE_AVAILABLE,
            'classifier_initialized': analysis_service.classifier is not None,
            'feature_extractor_initialized': analysis_service.feature_extractor is not None,
            'model_trained': (hasattr(analysis_service.classifier, 'abnormality_classifier') and 
                             analysis_service.classifier.abnormality_classifier is not None) if analysis_service.classifier else False
        }
    }
    
    return jsonify({
        'status': 'ok',
        'models': models_status,
        'models_folder': app.config['MODELS_FOLDER'],
        'python_path': sys.path[:3],  # Show first 3 entries
        'current_directory': os.getcwd(),
        'integration_methods': {
            'roi': 'predict_single_image_roi method',
            'pla': 'process_and_save_single_image method',
            'classification': 'predict_from_images method'
        }
    })

@app.route('/api/debug', methods=['GET', 'POST'])
def debug_endpoint():
    logger.info("Debug endpoint accessed")
    debug_info = {
        'method': request.method,
        'endpoint': '/api/debug',
        'files': list(request.files.keys()) if request.files else [],
        'form': dict(request.form) if request.form else {},
        'status': 'debug endpoint working',
        'current_directory': os.getcwd(),
        'python_path': sys.path[:5],  # Show first 5 entries
        'html_file_exists': os.path.exists('web_interface.html'),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'results_folder_exists': os.path.exists(app.config['RESULTS_FOLDER']),
        'models_folder_exists': os.path.exists(app.config['MODELS_FOLDER']),
        'module_availability': {
            'roi_mask_module': ROI_MASK_AVAILABLE,
            'pla_extractor_module': PLA_EXTRACTOR_AVAILABLE,
            'classifier_module': CLASSIFIER_MODULE_AVAILABLE,
            'feature_extractor_module': FEATURE_MODULE_AVAILABLE
        },
        'models_loaded': {
            'roi_model': analysis_service.roi_checkpoint_path is not None and ROI_MASK_AVAILABLE,
            'pla_extractor': analysis_service.pla_extractor is not None and PLA_EXTRACTOR_AVAILABLE,
            'classifier': (analysis_service.classifier is not None and 
                          CLASSIFIER_MODULE_AVAILABLE and 
                          FEATURE_MODULE_AVAILABLE)
        },
        'file_checks': {
            'RandomForest.py_exists': os.path.exists('RandomForest.py'),
            'classifier/RandomForest.py_exists': os.path.exists('classifier/RandomForest.py'),
            'FeatureExtractor.py_exists': os.path.exists('FeatureExtractor.py'),
            'classifier/FeatureExtractor.py_exists': os.path.exists('classifier/FeatureExtractor.py')
        }
    }
    logger.info(f"Debug info: {debug_info}")
    return jsonify(debug_info)

@app.route('/api/visualization/<filename>')
def get_visualization(filename):
    logger.info(f"Visualization requested: {filename}")
    try:
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    except FileNotFoundError:
        logger.error(f"Visualization not found: {filename}")
        return jsonify({'error': 'Visualization not found'}), 404
    
@app.route('/api/debug/analysis/<int:analysis_id>')
def debug_analysis_data(analysis_id):
    try:
        logger.info(f"Debug request for analysis ID: {analysis_id}")
        
        with db_integrated_service.db_manager:
            details = db_integrated_service.db_manager.get_analysis_results(analysis_id)
            recent = db_integrated_service.db_manager.get_recent_analyses(20)
            matching_recent = None
            for analysis in recent:
                if analysis.get('id') == analysis_id:
                    matching_recent = analysis
                    break
            
            debug_info = {
                'requested_id': analysis_id,
                'details_found': details is not None,
                'details_has_mammogram': bool(details and details.get('mammogram')),
                'matching_recent_found': matching_recent is not None,
                'details_structure': {},
                'recent_structure': {},
                'comparison': {}
            }
            
            if details:
                debug_info['details_structure'] = {
                    'keys': list(details.keys()),
                    'mammogram_keys': list(details.get('mammogram', {}).keys()) if details.get('mammogram') else [],
                    'classification_keys': list(details.get('classification', {}).keys()) if details.get('classification') else [],
                    'lesions_count': len(details.get('lesions', [])),
                    'analysis_files_count': len(details.get('analysis_files', []))
                }
            
            if matching_recent:
                debug_info['recent_structure'] = {
                    'keys': list(matching_recent.keys()),
                    'id': matching_recent.get('id'),
                    'filename': matching_recent.get('original_filename') or matching_recent.get('filename'),
                    'prediction': matching_recent.get('prediction'),
                    'upload_timestamp': matching_recent.get('upload_timestamp')
                }
                
                if details and details.get('mammogram'):
                    mammogram = details['mammogram']
                    debug_info['comparison'] = {
                        'ids_match': mammogram.get('id') == matching_recent.get('id'),
                        'filenames_match': (mammogram.get('original_filename') or mammogram.get('filename')) == 
                                         (matching_recent.get('original_filename') or matching_recent.get('filename')),
                        'timestamps_match': str(mammogram.get('upload_timestamp')) == str(matching_recent.get('upload_timestamp'))
                    }
            
            if details:
                debug_info['sample_details'] = {
                    'mammogram_id': details.get('mammogram', {}).get('id'),
                    'mammogram_filename': details.get('mammogram', {}).get('original_filename') or details.get('mammogram', {}).get('filename'),
                    'classification_prediction': details.get('classification', {}).get('prediction') if details.get('classification') else None
                }
            
            return jsonify({
                'status': 'success',
                'debug_info': debug_info,
                'raw_details': details, 
                'matching_recent': matching_recent
            })
            
    except mysql.connector.Error as e:
        logger.error(f"MySQL error in debug analysis endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Database error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500
    except Exception as e:
        logger.error(f"Error in debug analysis endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/health')
def health_check():
    logger.info("Health check accessed")
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '2.5.0',
        'integration_methods': {
            'roi': 'predict_single_image_roi',
            'pla': 'process_and_save_single_image',
            'classification': 'predict_from_images'
        },
        'module_availability': {
            'roi_mask_module': ROI_MASK_AVAILABLE,
            'pla_extractor_module': PLA_EXTRACTOR_AVAILABLE,
            'classifier_module': CLASSIFIER_MODULE_AVAILABLE,
            'feature_extractor_module': FEATURE_MODULE_AVAILABLE
        },
        'models_loaded': {
            'roi_model': analysis_service.roi_checkpoint_path is not None and ROI_MASK_AVAILABLE,
            'pla_extractor': analysis_service.pla_extractor is not None and PLA_EXTRACTOR_AVAILABLE,
            'classifier': (analysis_service.classifier is not None and 
                          CLASSIFIER_MODULE_AVAILABLE and 
                          FEATURE_MODULE_AVAILABLE)
        },
        'routes': [str(rule) for rule in app.url_map.iter_rules()]
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"404 error: {request.url}")
    return jsonify({
        'error': 'Not Found',
        'message': f'The requested URL {request.url} was not found.',
        'available_routes': [str(rule) for rule in app.url_map.iter_rules()]
    }), 404

@app.errorhandler(413)
def too_large(e):
    logger.error("File too large error")
    return jsonify({
        'error': 'File too large',
        'message': 'Please upload a file smaller than 50MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again.'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    
    preferred_port = int(os.environ.get('PORT', 5500))
    available_port = find_free_port(preferred_port)
    
    if available_port is None:
        print("❌ Could not find an available port.")
        exit(1)
    
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📄 web_interface.html exists: {os.path.exists('web_interface.html')}")
    print(f"📁 Models folder: {app.config['MODELS_FOLDER']} (exists: {os.path.exists(app.config['MODELS_FOLDER'])})")
    print(f"📁 Results folder: {app.config['RESULTS_FOLDER']} (exists: {os.path.exists(app.config['RESULTS_FOLDER'])})")
    
    print(f"\n🔧 Module Availability:")
    print(f"  ROI_Mask module: {'✅ Available' if ROI_MASK_AVAILABLE else '❌ Not Available'}")
    print(f"  PLA Extractor module: {'✅ Available' if PLA_EXTRACTOR_AVAILABLE else '❌ Not Available'}")
    print(f"  Classifier module: {'✅ Available' if CLASSIFIER_MODULE_AVAILABLE else '❌ Not Available'}")
    print(f"  FeatureExtractor module: {'✅ Available' if FEATURE_MODULE_AVAILABLE else '❌ Not Available'}")
    
    roi_model_path = os.path.join(app.config['MODELS_FOLDER'], 'unet_final_model.pkl')
    classifier_model_path = os.path.join(app.config['MODELS_FOLDER'], 'mammography_classifier.pkl')
    
    print(f"\n📊 Model Files Status:")
    print(f"  ROI Model (.pkl): {'✅ Found' if os.path.exists(roi_model_path) else '❌ Missing'}")
    print(f"  Classifier Model (.pkl): {'✅ Found' if os.path.exists(classifier_model_path) else '❌ Missing'}")
    
    print(f"""
    🔬 AI Breast Mammography Complete Analysis Server v2.5
    
    🌐 Server: http://localhost:{available_port}
    🔧 Debug Mode: {'✅ Enabled' if debug else '❌ Disabled'}
   
    📋 Available Routes:
    - GET  /                       - Main web interface
    - GET  /test                   - Test route
    - POST /api/analyze            - Analyze mammography image (ROI + PLA + Classification)
    - GET  /api/health             - Health check
    - GET  /api/debug              - Debug information
    - GET  /api/models             - Model loading status
    - GET  /api/visualization/<filename> - Get result visualization
    
    🎯 Complete Analysis Pipeline:
    - ROI masking using predict_single_image_roi method
    - PLA analysis using process_and_save_single_image method  
    - Classification using predict_from_images method
    
    """)
    
    try:
        app.run(host='127.0.0.1', port=available_port, debug=debug, threaded=True)
    except OSError as e:
        print(f"❌ Failed to start server: {e}")