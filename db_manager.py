import mysql.connector
from mysql.connector import Error
import hashlib
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import traceback
from PIL import Image
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MammographyDatabaseManager:
    def __init__(self, host='localhost', database='mammography_analysis', 
                 user='root', password='', port=3306):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        
        # File storage paths
        self.storage_root = 'database_storage'
        self.mammograms_path = os.path.join(self.storage_root, 'mammograms')
        self.analysis_files_path = os.path.join(self.storage_root, 'analysis_files')
        
        # Create storage directories
        self._create_storage_directories()
        
    def _create_storage_directories(self):
        directories = [
            self.storage_root,
            self.mammograms_path,
            self.analysis_files_path,
            os.path.join(self.analysis_files_path, 'roi_visualizations'),
            os.path.join(self.analysis_files_path, 'pla_visualizations'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                autocommit=False
            )
            if self.connection.is_connected():
                logger.info(f"Successfully connected to MySQL database: {self.database}")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
        
    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
            
    def _calculate_file_hash(self, file_path: str) -> str:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_image_dimensions(self, file_path: str) -> Tuple[int, int]:
        try:
            with Image.open(file_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            logger.warning(f"Could not get image dimensions: {e}")
            return (0, 0)
    
    def save_mammogram(self, original_file_path: str, original_filename: str) -> Optional[int]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(original_filename)[1].lower()
            new_filename = f"mammo_{timestamp}_{hashlib.md5(original_filename.encode()).hexdigest()[:8]}{file_extension}"
            
            permanent_path = os.path.join(self.mammograms_path, new_filename)
            shutil.copy2(original_file_path, permanent_path)
            
            file_size = os.path.getsize(permanent_path)
            file_hash = self._calculate_file_hash(permanent_path)
            width, height = self._get_image_dimensions(permanent_path)
            
            query = """
            INSERT INTO mammograms 
            (filename, original_filename, file_size, file_extension, file_path, 
             file_hash, image_width, image_height, processing_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                new_filename, original_filename, file_size, file_extension,
                permanent_path, file_hash, width, height, 'uploaded'
            )
            
            cursor.execute(query, values)
            mammogram_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Mammogram saved with ID: {mammogram_id}")
            return mammogram_id
            
        except Error as e:
            logger.error(f"Error saving mammogram: {e}")
            self.connection.rollback()
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def update_processing_status(self, mammogram_id: int, status: str, 
                               processing_time: float = None, error_message: str = None):
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            
            if status == 'completed':
                query = """
                UPDATE mammograms 
                SET processing_status = %s, analysis_timestamp = NOW(), processing_time = %s
                WHERE id = %s
                """
                values = (status, processing_time, mammogram_id)
            elif status == 'failed':
                query = """
                UPDATE mammograms 
                SET processing_status = %s, error_message = %s
                WHERE id = %s
                """
                values = (status, error_message, mammogram_id)
            else:
                query = """
                UPDATE mammograms 
                SET processing_status = %s
                WHERE id = %s
                """
                values = (status, mammogram_id)
            
            cursor.execute(query, values)
            self.connection.commit()
            return True
            
        except Error as e:
            logger.error(f"Error updating processing status: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def save_classification_result(self, mammogram_id: int, classification_data: Dict[str, Any]) -> bool:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO classifications 
            (mammogram_id, prediction, confidence, risk_score, abnormality_type,
             abnormality_description, abnormality_confidence, severity_class,
             severity_confidence, classification_reason, classification_method,
             has_mask, features_extracted, features_expected)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                mammogram_id,
                classification_data.get('prediction', 'Unknown'),
                classification_data.get('confidence', 0.0),
                classification_data.get('risk_score', 0.0),
                classification_data.get('abnormality_type', 'UNKNOWN'),
                classification_data.get('abnormality_description', ''),
                classification_data.get('abnormality_confidence', 0.0),
                classification_data.get('severity_class', 'N'),
                classification_data.get('severity_confidence', 0.0),
                classification_data.get('classification_reason', ''),
                classification_data.get('method', 'unknown'),
                classification_data.get('has_mask', False),
                classification_data.get('features_extracted', 0),
                classification_data.get('features_expected', 0)
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Classification result saved for mammogram ID: {mammogram_id}")
            return True
            
        except Error as e:
            logger.error(f"Error saving classification result: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def save_lesion_analysis(self, mammogram_id: int, lesion_size_mm: float, 
                           analysis_method: str = None, detection_confidence: float = None,
                           lesion_area: int = None, centroid_x: int = None, centroid_y: int = None) -> bool:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO lesions 
            (mammogram_id, lesion_size_mm, lesion_area_pixels, centroid_x, centroid_y,
             detection_confidence, analysis_method)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                mammogram_id, lesion_size_mm, lesion_area, centroid_x, centroid_y,
                detection_confidence, analysis_method
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Lesion analysis saved for mammogram ID: {mammogram_id}")
            return True
            
        except Error as e:
            logger.error(f"Error saving lesion analysis: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def save_analysis_file(self, mammogram_id: int, file_path: str, file_type: str, 
                          generation_method: str = None) -> bool:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            # Create subdirectory based on file type
            type_mapping = {
                'roi_mask': 'roi_masks',
                'roi_visualization': 'roi_visualizations',
                'pla_analysis': 'pla_analyses',
                'pla_visualization': 'pla_visualizations',
                'overlay': 'overlays',
                'annotated': 'annotated'
            }
            
            subdirectory = type_mapping.get(file_type, 'misc')
            target_dir = os.path.join(self.analysis_files_path, subdirectory)
            os.makedirs(target_dir, exist_ok=True)
            
            original_filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"mammo_{mammogram_id}_{file_type}_{timestamp}_{original_filename}"
            permanent_path = os.path.join(target_dir, new_filename)
            
            shutil.copy2(file_path, permanent_path)
            file_size = os.path.getsize(permanent_path)
            
            cursor = self.connection.cursor()
            query = """
            INSERT INTO analysis_files 
            (mammogram_id, file_type, filename, file_path, file_size, generation_method)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            filename = VALUES(filename),
            file_path = VALUES(file_path),
            file_size = VALUES(file_size),
            generation_method = VALUES(generation_method),
            generation_timestamp = CURRENT_TIMESTAMP
            """
            
            values = (mammogram_id, file_type, new_filename, permanent_path, file_size, generation_method)
            cursor.execute(query, values)
            self.connection.commit()
            
            logger.info(f"Analysis file saved: {file_type} for mammogram ID: {mammogram_id}")
            return True
            
        except Error as e:
            logger.error(f"Error saving analysis file: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def save_model_performance(self, mammogram_id: int, performance_data: Dict[str, Any]) -> bool:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO model_performance 
            (mammogram_id, roi_model_used, pla_extractor_used, classifier_used,
             roi_processing_time, pla_processing_time, classification_processing_time,
             total_processing_time, roi_method, pla_method, classification_method,
             roi_success, pla_success, classification_success)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                mammogram_id,
                performance_data.get('roi_model_used', False),
                performance_data.get('pla_extractor_used', False),
                performance_data.get('classifier_used', False),
                performance_data.get('roi_processing_time', 0.0),
                performance_data.get('pla_processing_time', 0.0),
                performance_data.get('classification_processing_time', 0.0),
                performance_data.get('total_processing_time', 0.0),
                performance_data.get('roi_method', ''),
                performance_data.get('pla_method', ''),
                performance_data.get('classification_method', ''),
                performance_data.get('roi_success', False),
                performance_data.get('pla_success', False),
                performance_data.get('classification_success', False)
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Model performance saved for mammogram ID: {mammogram_id}")
            return True
            
        except Error as e:
            logger.error(f"Error saving model performance: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def log_system_event(self, level: str, message: str, context: Dict = None, 
                        mammogram_id: int = None) -> bool:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO system_logs (mammogram_id, log_level, log_message, log_context)
            VALUES (%s, %s, %s, %s)
            """
            
            context_json = json.dumps(context) if context else None
            values = (mammogram_id, level, message, context_json)
            cursor.execute(query, values)
            self.connection.commit()
            return True
            
        except Error as e:
            logger.error(f"Error logging system event: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_mammogram_by_id(self, mammogram_id: int) -> Optional[Dict[str, Any]]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM mammograms WHERE id = %s"
            cursor.execute(query, (mammogram_id,))
            result = cursor.fetchone()
            return result
            
        except Error as e:
            logger.error(f"Error retrieving mammogram: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_analysis_results(self, mammogram_id: int) -> Dict[str, Any]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get mammogram info
            cursor.execute("SELECT * FROM mammograms WHERE id = %s", (mammogram_id,))
            mammogram = cursor.fetchone()
            
            # Get classification
            cursor.execute("SELECT * FROM classifications WHERE mammogram_id = %s", (mammogram_id,))
            classification = cursor.fetchone()
            
            # Get lesion analysis
            cursor.execute("SELECT * FROM lesions WHERE mammogram_id = %s", (mammogram_id,))
            lesions = cursor.fetchall()
            
            # Get analysis files
            cursor.execute("SELECT * FROM analysis_files WHERE mammogram_id = %s", (mammogram_id,))
            files = cursor.fetchall()
            
            # Get model performance
            cursor.execute("SELECT * FROM model_performance WHERE mammogram_id = %s", (mammogram_id,))
            performance = cursor.fetchone()
            
            return {
                'mammogram': mammogram,
                'classification': classification,
                'lesions': lesions,
                'analysis_files': files,
                'performance': performance
            }
            
        except Error as e:
            logger.error(f"Error retrieving analysis results: {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            stats = {}
            
            # Total mammograms
            cursor.execute("SELECT COUNT(*) as total FROM mammograms")
            stats['total_mammograms'] = cursor.fetchone()['total']
            
            # Processing status breakdown
            cursor.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM mammograms 
                GROUP BY processing_status
            """)
            stats['status_breakdown'] = cursor.fetchall()
            
            # Classification breakdown
            cursor.execute("""
                SELECT prediction, COUNT(*) as count 
                FROM classifications 
                GROUP BY prediction
            """)
            stats['prediction_breakdown'] = cursor.fetchall()
            
            # Average processing time
            cursor.execute("SELECT AVG(processing_time) as avg_time FROM mammograms WHERE processing_time IS NOT NULL")
            result = cursor.fetchone()
            stats['avg_processing_time'] = result['avg_time'] if result['avg_time'] else 0
            
            # Lesion size statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_lesions,
                    AVG(lesion_size_mm) as avg_size,
                    MIN(lesion_size_mm) as min_size,
                    MAX(lesion_size_mm) as max_size
                FROM lesions 
                WHERE lesion_size_mm > 0
            """)
            stats['lesion_statistics'] = cursor.fetchone()
            
            return stats
            
        except Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return []
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT 
                m.id, m.filename, m.original_filename, m.upload_timestamp,
                m.processing_status, m.processing_time,
                c.prediction, c.confidence, c.abnormality_type,
                l.lesion_size_mm
            FROM mammograms m
            LEFT JOIN classifications c ON m.id = c.mammogram_id
            LEFT JOIN lesions l ON m.id = l.mammogram_id
            ORDER BY m.upload_timestamp DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error retrieving recent analyses: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()     
    
    def search_mammograms(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return []
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            base_query = """
            SELECT 
                m.id, m.filename, m.original_filename, m.upload_timestamp,
                m.processing_status, m.processing_time,
                c.prediction, c.confidence, c.abnormality_type, c.severity_class,
                l.lesion_size_mm
            FROM mammograms m
            LEFT JOIN classifications c ON m.id = c.mammogram_id
            LEFT JOIN lesions l ON m.id = l.mammogram_id
            WHERE 1=1
            """
            
            conditions = []
            params = []
            
            # Add search conditions
            if 'prediction' in search_criteria:
                conditions.append("c.prediction = %s")
                params.append(search_criteria['prediction'])
            
            if 'abnormality_type' in search_criteria:
                conditions.append("c.abnormality_type = %s")
                params.append(search_criteria['abnormality_type'])
            
            if 'severity_class' in search_criteria:
                conditions.append("c.severity_class = %s")
                params.append(search_criteria['severity_class'])
            
            if 'min_confidence' in search_criteria:
                conditions.append("c.confidence >= %s")
                params.append(search_criteria['min_confidence'])
            
            if 'min_lesion_size' in search_criteria:
                conditions.append("l.lesion_size_mm >= %s")
                params.append(search_criteria['min_lesion_size'])
            
            if 'max_lesion_size' in search_criteria:
                conditions.append("l.lesion_size_mm <= %s")
                params.append(search_criteria['max_lesion_size'])
            
            if 'date_from' in search_criteria:
                conditions.append("DATE(m.upload_timestamp) >= %s")
                params.append(search_criteria['date_from'])
            
            if 'date_to' in search_criteria:
                conditions.append("DATE(m.upload_timestamp) <= %s")
                params.append(search_criteria['date_to'])
            
            if 'processing_status' in search_criteria:
                conditions.append("m.processing_status = %s")
                params.append(search_criteria['processing_status'])
            
            # Add conditions to query
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            # Add ordering and limit
            base_query += " ORDER BY m.upload_timestamp DESC"
            
            if 'limit' in search_criteria:
                base_query += " LIMIT %s"
                params.append(search_criteria['limit'])
            
            cursor.execute(base_query, params)
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error searching mammograms: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def export_analysis_data(self, output_format: str = 'json') -> Optional[str]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT 
                m.id, m.filename, m.original_filename, m.file_size,
                m.upload_timestamp, m.analysis_timestamp, m.processing_time,
                m.processing_status, m.image_width, m.image_height,
                c.prediction, c.confidence, c.risk_score,
                c.abnormality_type, c.abnormality_description, c.abnormality_confidence,
                c.severity_class, c.severity_confidence, c.classification_reason,
                c.has_mask, c.features_extracted, c.features_expected,
                l.lesion_size_mm, l.lesion_area_pixels, l.detection_confidence,
                mp.roi_model_used, mp.pla_extractor_used, mp.classifier_used,
                mp.total_processing_time, mp.roi_success, mp.pla_success, mp.classification_success
            FROM mammograms m
            LEFT JOIN classifications c ON m.id = c.mammogram_id
            LEFT JOIN lesions l ON m.id = l.mammogram_id
            LEFT JOIN model_performance mp ON m.id = mp.mammogram_id
            WHERE m.processing_status = 'completed'
            ORDER BY m.upload_timestamp DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            for result in results:
                for key, value in result.items():
                    if isinstance(value, datetime):
                        result[key] = value.isoformat()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == 'json':
                export_path = os.path.join(self.storage_root, f'export_{timestamp}.json')
                with open(export_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif output_format.lower() == 'csv':
                import csv
                export_path = os.path.join(self.storage_root, f'export_{timestamp}.csv')
                if results:
                    with open(export_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
            else:
                logger.error(f"Unsupported export format: {output_format}")
                return None
            
            logger.info(f"Data exported to: {export_path}")
            return export_path
            
        except Error as e:
            logger.error(f"Error exporting data: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_analysis_file_path(self, mammogram_id: int, file_type: str) -> Optional[str]:
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT file_path FROM analysis_files WHERE mammogram_id = %s AND file_type = %s"
            cursor.execute(query, (mammogram_id, file_type))
            result = cursor.fetchone()
            
            if result and os.path.exists(result['file_path']):
                return result['file_path']
            return None
            
        except Error as e:
            logger.error(f"Error retrieving analysis file path: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    

    def delete_mammogram_analysis(self, mammogram_id):
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("DELETE FROM analysis_files WHERE mammogram_id = %s", (mammogram_id,))
            affected_files = cursor.rowcount
            
            cursor.execute("DELETE FROM lesions WHERE mammogram_id = %s", (mammogram_id,))
            affected_lesions = cursor.rowcount
            
            cursor.execute("DELETE FROM classifications WHERE mammogram_id = %s", (mammogram_id,))
            affected_classification = cursor.rowcount
            
            cursor.execute("DELETE FROM mammograms WHERE id = %s", (mammogram_id,))
            affected_mammogram = cursor.rowcount
            
            if affected_mammogram == 0:
                logger.warning(f"No mammogram found with ID {mammogram_id}")
                return False
            
            self.connection.commit()
            logger.info(f"Successfully deleted mammogram analysis with ID {mammogram_id}")
            logger.info(f"Deleted records: mammogram={affected_mammogram}, classification={affected_classification}, lesions={affected_lesions}, files={affected_files}")
            return True
            
        except mysql.connector.Error as e:
            logger.error(f"MySQL error deleting mammogram {mammogram_id}: {e}")
            self.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting mammogram {mammogram_id}: {e}")
            self.connection.rollback()
            return False

    def get_analysis_file_by_type(self, mammogram_id, file_type):
        try:
            query = """
            SELECT filename, file_path, file_type, created_at
            FROM analysis_files 
            WHERE mammogram_id = %s AND file_type = %s
            ORDER BY generation_timestamp DESC
            LIMIT 1
            """
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (mammogram_id, file_type))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'filename': result['filename'],
                    'file_path': result['file_path'],
                    'file_type': result['file_type'],
                    'created_at': result['generation_timestamp']
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting analysis file by type: {e}")
            return None

    def get_all_analysis_files(self, mammogram_id):
        try:
            query = """
            SELECT filename, file_path, file_type, created_at
            FROM analysis_files 
            WHERE mammogram_id = %s
            ORDER BY file_type, generation_timestamp DESC
            """
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (mammogram_id,))
            results = cursor.fetchall()
            cursor.close()
            
            files = []
            for result in results:
                files.append({
                    'filename': result['filename'],
                    'file_path': result['file_path'],
                    'file_type': result['file_type'],
                    'created_at': result['generation_timestamp']
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting all analysis files: {e}")
            return []

    def check_file_exists(self, file_path):
        try:
            full_path = os.path.join(self.storage_root, 'analysis_files', file_path)
            return os.path.exists(full_path) and os.path.isfile(full_path)
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def test_database_connection():
    db_manager = MammographyDatabaseManager()
    
    if db_manager.connect():
        print("✅ Database connection successful")
        
        stats = db_manager.get_statistics()
        print(f"📊 Total mammograms in database: {stats.get('total_mammograms', 0)}")
        
        recent = db_manager.get_recent_analyses(5)
        print(f"📈 Recent analyses: {len(recent)} records")
        
        db_manager.disconnect()
        return True
    else:
        print("❌ Database connection failed")
        return False

if __name__ == "__main__":
    print("🧪 Testing Database Manager...")
    test_database_connection()
