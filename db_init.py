import os
import time
from db_manager import MammographyDatabaseManager
import logging
import traceback

logger = logging.getLogger(__name__)

class DatabaseIntegratedAnalysisService:
    def __init__(self, original_analysis_service):
        self.original_service = original_analysis_service
        self.db_manager = MammographyDatabaseManager(
            host='localhost',
            database='mammography_analysis',
            user='root',
            password='',
            port=3306
        )
        
    def analyze_mammography_with_database(self, image_file, original_filename: str = None):
        start_time = time.time()
        mammogram_id = None
        temp_path = None
        
        try:
            logger.info("Starting database-integrated mammography analysis")
            
            if not hasattr(image_file, 'read') or not hasattr(image_file, 'seek'):
                raise ValueError("Invalid file object provided")
            
            database_available = self.db_manager.connect()
            if not database_available:
                logger.warning("Database connection failed - proceeding with analysis only")
            
            if original_filename is None:
                original_filename = getattr(image_file, 'filename', f"mammogram_{int(time.time())}.png")
            
            if database_available:
                try:
                    uploads_dir = 'uploads'
                    os.makedirs(uploads_dir, exist_ok=True)
                    
                    image_file.seek(0)
                    file_data = image_file.read()
                    
                    if not file_data:
                        raise ValueError("Empty file provided")
                    
                    timestamp = int(time.time())
                    temp_filename = f"temp_upload_{timestamp}_{original_filename}"
                    temp_path = os.path.join(uploads_dir, temp_filename)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(file_data)
                    
                    logger.info(f"Temporary file created: {temp_path} (size: {len(file_data)} bytes)")
                    
                    mammogram_id = self.db_manager.save_mammogram(temp_path, original_filename)
                    if mammogram_id:
                        logger.info(f"Mammogram saved to database with ID: {mammogram_id}")
                        self.db_manager.update_processing_status(mammogram_id, 'processing')
                        self.db_manager.log_system_event(
                            'INFO', 
                            f"Starting analysis for mammogram {original_filename}",
                            {
                                'filename': original_filename, 
                                'file_size': len(file_data),
                                'temp_path': temp_path
                            },
                            mammogram_id
                        )
                    else:
                        logger.warning("Failed to save mammogram to database, continuing with analysis")
                        
                except Exception as db_prep_error:
                    logger.error(f"Error preparing database storage: {db_prep_error}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    mammogram_id = None
            
            image_file.seek(0)
            if database_available and temp_path and os.path.exists(temp_path):
                with open(temp_path, 'rb') as temp_file:
                    analysis_start_time = time.time()
                    results = self.original_service.analyze_mammography(temp_file)
                    analysis_time = time.time() - analysis_start_time
            else:
                analysis_start_time = time.time()
                results = self.original_service.analyze_mammography(image_file)
                analysis_time = time.time() - analysis_start_time
            
            logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
            if database_available and mammogram_id and results:
                try:
                    self._save_analysis_results_to_database(mammogram_id, results, analysis_time, original_filename)
                    results['database'] = {
                        'mammogram_id': mammogram_id,
                        'stored': True,
                        'processing_time': time.time() - start_time
                    }
                    
                except Exception as db_save_error:
                    logger.error(f"Error saving analysis results to database: {db_save_error}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    try:
                        if mammogram_id:
                            self.db_manager.update_processing_status(
                                mammogram_id, 
                                'failed', 
                                error_message=f"Analysis completed but database save failed: {str(db_save_error)}"
                            )
                    except Exception as status_update_error:
                        logger.error(f"Failed to update database status: {status_update_error}")
            
            if 'database' not in results:
                results['database'] = {
                    'mammogram_id': mammogram_id,
                    'stored': database_available and mammogram_id is not None,
                    'processing_time': time.time() - start_time if database_available else None
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in database-integrated analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if mammogram_id:
                try:
                    self.db_manager.update_processing_status(
                        mammogram_id, 
                        'failed', 
                        error_message=str(e)
                    )
                    
                    self.db_manager.log_system_event(
                        'ERROR',
                        f"Analysis failed: {str(e)}",
                        {'error': str(e), 'traceback': traceback.format_exc()},
                        mammogram_id
                    )
                except Exception as db_error:
                    logger.error(f"Failed to update database status: {db_error}")
            
            try:
                logger.info("Attempting fallback to original analysis method")
                image_file.seek(0)
                results = self.original_service.analyze_mammography(image_file)
                results['database'] = {
                    'mammogram_id': None,
                    'stored': False,
                    'error': f"Database integration failed: {str(e)}"
                }
                return results
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
                raise RuntimeError(f"Both database-integrated and fallback analysis failed. Original error: {str(e)}, Fallback error: {str(fallback_error)}")
                
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
            
            if hasattr(self, 'db_manager'):
                self.db_manager.disconnect()
    
    def _save_analysis_results_to_database(self, mammogram_id: int, results: dict, analysis_time: float, original_filename: str):
        try:
            classification_data = {
                'prediction': results.get('prediction', 'Unknown'),
                'confidence': float(results.get('confidence', 0.0)),
                'risk_score': float(results.get('risk_score', 0.0)),
                'abnormality_type': results.get('abnormality_type', 'UNKNOWN'),
                'abnormality_description': results.get('abnormality_description', ''),
                'abnormality_confidence': float(results.get('abnormality_confidence', 0.0)),
                'severity_class': results.get('severity_class', 'N'),
                'severity_confidence': float(results.get('severity_confidence', 0.0)),
                'classification_reason': results.get('classification_reason', ''),
                'method': results.get('classification_details', {}).get('method', 'unknown'),
                'has_mask': results.get('classification_details', {}).get('has_mask', False),
                'features_extracted': int(results.get('classification_details', {}).get('features_extracted', 0)),
                'features_expected': int(results.get('classification_details', {}).get('features_expected', 0))
            }
            
            success = self.db_manager.save_classification_result(mammogram_id, classification_data)
            if success:
                logger.info("Classification results saved to database")
            
            lesion_size = float(results.get('lesion_size_mm', 0.0))
            if lesion_size > 0:
                self.db_manager.save_lesion_analysis(
                    mammogram_id=mammogram_id,
                    lesion_size_mm=lesion_size,
                    analysis_method='roi_mask_analysis',
                    detection_confidence=float(results.get('confidence', 0.0))
                )
                logger.info("Lesion analysis saved to database")
            
            self._save_analysis_files_to_database(mammogram_id, results)
            
            performance_data = {
                'roi_model_used': results.get('models_loaded', {}).get('roi_model', False),
                'pla_extractor_used': results.get('models_loaded', {}).get('pla_extractor', False),
                'classifier_used': results.get('models_loaded', {}).get('classifier', False),
                'total_processing_time': float(results.get('processing_time', 0.0)),
                'roi_method': 'predict_single_image_roi',
                'pla_method': 'process_and_save_single_image',
                'classification_method': results.get('classification_details', {}).get('method', 'unknown'),
                'roi_success': results.get('roi_analysis_details', {}).get('success', False),
                'pla_success': results.get('pla_analysis_details', {}).get('success', False),
                'classification_success': 'error' not in results.get('classification_details', {})
            }
            
            self.db_manager.save_model_performance(mammogram_id, performance_data)
            logger.info("Model performance metrics saved to database")
            
            total_time = analysis_time
            self.db_manager.update_processing_status(
                mammogram_id, 
                'completed', 
                processing_time=total_time
            )
            
            self.db_manager.log_system_event(
                'INFO',
                f"Analysis completed successfully for mammogram {original_filename}",
                {
                    'processing_time': total_time,
                    'prediction': results.get('prediction', 'Unknown'),
                    'lesion_size_mm': lesion_size
                },
                mammogram_id
            )
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            raise
    
    def _save_analysis_files_to_database(self, mammogram_id: int, results: dict):
        try:
            results_folder = 'results'
            roi_viz_path = results.get('roi_visualization_path')
            if roi_viz_path:
                roi_file_path = os.path.join(results_folder, roi_viz_path)
                if os.path.exists(roi_file_path):
                    self.db_manager.save_analysis_file(
                        mammogram_id=mammogram_id,
                        file_path=roi_file_path,
                        file_type='roi_visualization',
                        generation_method='predict_single_image_roi'
                    )
                    logger.info("ROI visualization saved to database")
            
            pla_viz_path = results.get('pla_visualization_path')
            if pla_viz_path:
                pla_file_path = os.path.join(results_folder, pla_viz_path)
                if os.path.exists(pla_file_path):
                    self.db_manager.save_analysis_file(
                        mammogram_id=mammogram_id,
                        file_path=pla_file_path,
                        file_type='pla_visualization',
                        generation_method='process_and_save_single_image'
                    )
                    logger.info("PLA visualization saved to database")
                    
        except Exception as e:
            logger.warning(f"Error saving analysis files: {e}")
    
