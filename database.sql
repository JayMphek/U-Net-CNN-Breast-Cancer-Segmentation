-- Mammography Analysis Database Schema
CREATE DATABASE IF NOT EXISTS mammography_analysis;
USE mammography_analysis;

-- Main mammogram records table
CREATE TABLE mammograms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    file_extension VARCHAR(10) NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_timestamp TIMESTAMP NULL,
    file_path VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) UNIQUE,
    
    -- Image metadata
    image_width INT,
    image_height INT,
    
    -- Processing status
    processing_status ENUM('uploaded', 'processing', 'completed', 'failed') DEFAULT 'uploaded',
    processing_time DECIMAL(10,2),
    error_message TEXT,
    
    INDEX idx_filename (filename),
    INDEX idx_upload_time (upload_timestamp),
    INDEX idx_status (processing_status),
    INDEX idx_hash (file_hash)
);

-- Classification results table
CREATE TABLE classifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mammogram_id INT NOT NULL,
    
    -- Main classification results
    prediction VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    
    -- Abnormality details
    abnormality_type VARCHAR(20) NOT NULL,
    abnormality_description VARCHAR(255),
    abnormality_confidence DECIMAL(5,4) NOT NULL,
    
    -- Severity classification
    severity_class VARCHAR(5) NOT NULL,
    severity_confidence DECIMAL(5,4) NOT NULL,
    
    -- Classification metadata
    classification_reason TEXT,
    classification_method VARCHAR(50),
    has_mask BOOLEAN DEFAULT FALSE,
    features_extracted INT DEFAULT 0,
    features_expected INT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (mammogram_id) REFERENCES mammograms(id) ON DELETE CASCADE,
    INDEX idx_mammogram_id (mammogram_id),
    INDEX idx_prediction (prediction),
    INDEX idx_abnormality_type (abnormality_type),
    INDEX idx_severity_class (severity_class)
);

-- Lesion analysis table
CREATE TABLE lesions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mammogram_id INT NOT NULL,
    
    -- Lesion measurements
    lesion_size_mm DECIMAL(8,2) DEFAULT 0.0,
    lesion_area_pixels INT,
    lesion_perimeter_pixels INT,
    
    -- Lesion position (if extractable)
    centroid_x INT,
    centroid_y INT,
    bounding_box_x INT,
    bounding_box_y INT,
    bounding_box_width INT,
    bounding_box_height INT,
    
    -- Analysis metadata
    detection_confidence DECIMAL(5,4),
    analysis_method VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (mammogram_id) REFERENCES mammograms(id) ON DELETE CASCADE,
    INDEX idx_mammogram_id (mammogram_id),
    INDEX idx_lesion_size (lesion_size_mm)
);

-- Analysis files table (stores paths to generated analysis images)
CREATE TABLE analysis_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mammogram_id INT NOT NULL,
    
    -- File information
    file_type ENUM('roi_mask', 'roi_visualization', 'pla_analysis', 'pla_visualization', 'overlay', 'annotated') NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    
    -- Generation metadata
    generation_method VARCHAR(50),
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (mammogram_id) REFERENCES mammograms(id) ON DELETE CASCADE,
    INDEX idx_mammogram_id (mammogram_id),
    INDEX idx_file_type (file_type),
    UNIQUE KEY unique_mammo_file_type (mammogram_id, file_type)
);

-- Model performance tracking table
CREATE TABLE model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mammogram_id INT NOT NULL,
    
    -- Model status
    roi_model_used BOOLEAN DEFAULT FALSE,
    pla_extractor_used BOOLEAN DEFAULT FALSE,
    classifier_used BOOLEAN DEFAULT FALSE,
    
    -- Performance metrics
    roi_processing_time DECIMAL(8,3),
    pla_processing_time DECIMAL(8,3),
    classification_processing_time DECIMAL(8,3),
    total_processing_time DECIMAL(8,3),
    
    -- Model versions/methods
    roi_method VARCHAR(100),
    pla_method VARCHAR(100),
    classification_method VARCHAR(100),
    
    -- Success status
    roi_success BOOLEAN DEFAULT FALSE,
    pla_success BOOLEAN DEFAULT FALSE,
    classification_success BOOLEAN DEFAULT FALSE,
    
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (mammogram_id) REFERENCES mammograms(id) ON DELETE CASCADE,
    INDEX idx_mammogram_id (mammogram_id),
    INDEX idx_analysis_time (analysis_timestamp)
);

-- System logs table
CREATE TABLE system_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mammogram_id INT NULL,
    
    log_level ENUM('INFO', 'WARNING', 'ERROR', 'DEBUG') NOT NULL,
    log_message TEXT NOT NULL,
    log_context JSON,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (mammogram_id) REFERENCES mammograms(id) ON DELETE SET NULL,
    INDEX idx_mammogram_id (mammogram_id),
    INDEX idx_log_level (log_level),
    INDEX idx_timestamp (timestamp)
);

-- Statistics view for quick analysis overview
CREATE VIEW analysis_statistics AS
SELECT 
    DATE(upload_timestamp) as analysis_date,
    COUNT(*) as total_mammograms,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_analyses,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_analyses,
    AVG(processing_time) as avg_processing_time,
    COUNT(CASE WHEN c.prediction = 'Normal' THEN 1 END) as normal_count,
    COUNT(CASE WHEN c.prediction = 'Abnormal' THEN 1 END) as abnormal_count,
    COUNT(CASE WHEN c.prediction = 'Malignant' THEN 1 END) as malignant_count,
    COUNT(CASE WHEN c.prediction = 'Benign' THEN 1 END) as benign_count,
    AVG(l.lesion_size_mm) as avg_lesion_size
FROM mammograms m
LEFT JOIN classifications c ON m.id = c.mammogram_id
LEFT JOIN lesions l ON m.id = l.mammogram_id
GROUP BY DATE(upload_timestamp)
ORDER BY analysis_date DESC;

-- Lesion size distribution view
CREATE VIEW lesion_size_distribution AS
SELECT 
    CASE 
        WHEN lesion_size_mm = 0 THEN 'No Lesion'
        WHEN lesion_size_mm < 10 THEN 'Small (< 10mm)'
        WHEN lesion_size_mm < 20 THEN 'Medium (10-20mm)'
        WHEN lesion_size_mm < 50 THEN 'Large (20-50mm)'
        ELSE 'Very Large (> 50mm)'
    END as size_category,
    COUNT(*) as count,
    AVG(lesion_size_mm) as avg_size,
    MIN(lesion_size_mm) as min_size,
    MAX(lesion_size_mm) as max_size
FROM lesions
GROUP BY 
    CASE 
        WHEN lesion_size_mm = 0 THEN 'No Lesion'
        WHEN lesion_size_mm < 10 THEN 'Small (< 10mm)'
        WHEN lesion_size_mm < 20 THEN 'Medium (10-20mm)'
        WHEN lesion_size_mm < 50 THEN 'Large (20-50mm)'
        ELSE 'Very Large (> 50mm)'
    END
ORDER BY avg_size;

-- Classification accuracy view (for model evaluation)
CREATE VIEW classification_summary AS
SELECT 
    abnormality_type,
    prediction,
    severity_class,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(risk_score) as avg_risk_score,
    AVG(abnormality_confidence) as avg_abnormality_confidence,
    AVG(severity_confidence) as avg_severity_confidence
FROM classifications
GROUP BY abnormality_type, prediction, severity_class
ORDER BY abnormality_type, prediction;

-- Insert sample data for testing (optional)
INSERT INTO mammograms (filename, original_filename, file_size, file_extension, file_path, file_hash, image_width, image_height, processing_status) VALUES
('sample_001.png', 'test_mammogram.png', 1024000, '.png', '/uploads/sample_001.png', 'abc123def456', 512, 512, 'completed');

-- Create indexes for better performance
CREATE INDEX idx_combined_status_time ON mammograms(processing_status, upload_timestamp);
CREATE INDEX idx_classification_combined ON classifications(prediction, abnormality_type, severity_class);
CREATE INDEX idx_analysis_files_combined ON analysis_files(mammogram_id, file_type);
