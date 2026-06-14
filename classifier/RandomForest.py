import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        total = len(y)
        impurity = 1.0
        for count in counts.values():
            prob = count / total
            impurity -= prob ** 2
        return impurity
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None
            
        parent_impurity = self._gini_impurity(y)
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = max(1, int(np.sqrt(n)))
        features = random.sample(range(n), min(n_features, n))
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_impurity = self._gini_impurity(y[left_mask])
                right_impurity = self._gini_impurity(y[right_mask])
                
                left_weight = np.sum(left_mask) / m
                right_weight = np.sum(right_mask) / m
                
                weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        
    def fit(self, X, y):
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.trees = []
        
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        probabilities = []
        
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            vote_counts = Counter(votes)
            probs = []
            for class_label in self.classes_:
                probs.append(vote_counts.get(class_label, 0) / len(self.trees))
            probabilities.append(probs)
        
        return np.array(probabilities)

class MammographyClassifier:
    def __init__(self, random_state=50):
        self.random_state = random_state
        self.abnormality_classifier = None
        self.severity_classifier = None
        self.scaler = StandardScaler()
        self.abnormality_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_metadata(self, metadata_path):
        metadata = []
        print(f"Loading metadata from: {metadata_path}")

        try:
            with open(metadata_path, 'r') as file:
                lines = file.readlines()
                print(f"Total lines in metadata file: {len(lines)}")

                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        print(f"Line {i+1}: {parts}")
                        if len(parts) >= 7:  
                            try:
                                metadata.append({
                                    'image_ref': parts[0],          
                                    'view': parts[1],               
                                    'abnormality_type': parts[2],   
                                    'abnormality_class': parts[3], 
                                    'x_coord': float(parts[4]) if parts[4] != '-' else np.nan,  
                                    'y_coord': float(parts[5]) if parts[5] != '-' else np.nan,  
                                    'radius': float(parts[6]) if parts[6] != '-' else np.nan    
                                })
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse line {i+1}: {line}")
                                print(f"Error: {e}")
                        else:
                            print(f"Warning: Line {i+1} has insufficient parts ({len(parts)}): {line}")

            print(f"Successfully parsed {len(metadata)} metadata entries")

            if len(metadata) == 0:
                print("ERROR: No metadata entries were parsed!")
                return pd.DataFrame()

            df = pd.DataFrame(metadata)
            print("Sample metadata entries:")
            print(df.head())
            print("Metadata columns:", df.columns.tolist())

            return df

        except FileNotFoundError:
            print(f"ERROR: Metadata file not found: {metadata_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR loading metadata: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def merge_data(self, features_csv_path, metadata_txt_path):
        features_df = pd.read_csv(features_csv_path)
        metadata_df = self.load_metadata(metadata_txt_path)
        features_df['image_ref'] = features_df['filename'].str.extract(r'IMG(\d+)')[0].astype(str).str.zfill(3)
        metadata_df['image_ref'] = metadata_df['image_ref'].str.replace('IMG', '').str.zfill(3)
        merged_df = pd.merge(features_df, metadata_df, on='image_ref', how='inner')

        print(f"Features dataset: {len(features_df)} samples")
        print(f"Metadata dataset: {len(metadata_df)} samples")
        print(f"Merged dataset: {len(merged_df)} samples")

        return merged_df
    
    def prepare_data_for_abnormality_classification(self, merged_df):
        filtered_df = merged_df[merged_df['abnormality_type'] != 'NORM'].copy()
        
        print(f"\nAbnormality classification data preparation:")
        print(f"Total samples after removing NORM: {len(filtered_df)}")
        print("Abnormality type distribution:")
        print(filtered_df['abnormality_type'].value_counts())
        
        feature_columns = filtered_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in feature_columns if col not in ['x_coord', 'y_coord', 'radius']]
        
        X = filtered_df[feature_columns].fillna(0).values
        y = filtered_df['abnormality_type'].values
        
        return X, y, feature_columns, filtered_df
    
    def prepare_data_for_severity_classification(self, merged_df):
        filtered_df = merged_df[
            (merged_df['abnormality_type'] != 'NORM') & 
            (merged_df['abnormality_class'] != 'N')
        ].copy()
        
        print(f"\nSeverity classification data preparation:")
        print(f"Total samples after removing NORM and undefined: {len(filtered_df)}")
        print("Severity distribution:")
        print(filtered_df['abnormality_class'].value_counts())
        
        feature_columns = filtered_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in feature_columns if col not in ['x_coord', 'y_coord', 'radius']]
        
        X = filtered_df[feature_columns].fillna(0).values
        y = filtered_df['abnormality_class'].values
        
        return X, y, feature_columns, filtered_df
    
    def train_abnormality_classifier(self, X, y):
        print("\n" + "="*50)
        print("TRAINING ABNORMALITY TYPE CLASSIFIER")
        print("="*50)

        print("Class distribution before encoding:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} samples")

        y_encoded = self.abnormality_encoder.fit_transform(y)

        min_class_count = np.min(np.bincount(y_encoded))
        test_size = 0.2

        if min_class_count < 2:
            print(f"Warning: Some classes have only {min_class_count} sample(s). Using random split instead of stratified.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state, 
                stratify=y_encoded
            )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.abnormality_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )

        print("Training Random Forest classifier...")
        self.abnormality_classifier.fit(X_train_scaled, y_train)

        y_pred = self.abnormality_classifier.predict(X_test_scaled)
        y_pred_proba = self.abnormality_classifier.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAbnormality Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")

        class_names = self.abnormality_encoder.classes_
        test_classes = np.unique(y_test)
        pred_classes = np.unique(y_pred)
        print(f"Classes in test set: {[class_names[i] for i in test_classes]}")
        print(f"Classes predicted: {[class_names[i] for i in pred_classes]}")

        print(f"\nClassification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names, 
            zero_division=0,
            output_dict=True
        )

        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name}: precision={metrics['precision']:.3f}, "
                      f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}, "
                      f"support={int(metrics['support'])}")

        print(f"\nOverall metrics:")
        print(f"Macro avg: precision={report['macro avg']['precision']:.3f}, "
              f"recall={report['macro avg']['recall']:.3f}, f1={report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg: precision={report['weighted avg']['precision']:.3f}, "
              f"recall={report['weighted avg']['recall']:.3f}, f1={report['weighted avg']['f1-score']:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, class_names, "Abnormality Type Classification")
        return accuracy

    def train_severity_classifier(self, X, y):
        print("\n" + "="*50)
        print("TRAINING SEVERITY CLASSIFIER")
        print("="*50)

        print("Class distribution before encoding:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} samples")

        y_encoded = self.severity_encoder.fit_transform(y)
        min_class_count = np.min(np.bincount(y_encoded))
        test_size = 0.2

        if min_class_count < 2:
            print(f"Warning: Some classes have only {min_class_count} sample(s). Using random split instead of stratified.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_state,
                stratify=y_encoded
            )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.severity_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )

        print("Training Random Forest classifier...")
        self.severity_classifier.fit(X_train_scaled, y_train)

        y_pred = self.severity_classifier.predict(X_test_scaled)
        y_pred_proba = self.severity_classifier.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nSeverity Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")

        class_names = self.severity_encoder.classes_
        test_classes = np.unique(y_test)
        pred_classes = np.unique(y_pred)
        print(f"Classes in test set: {[class_names[i] for i in test_classes]}")
        print(f"Classes predicted: {[class_names[i] for i in pred_classes]}")

        print(f"\nClassification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names, 
            zero_division=0,
            output_dict=True
        )

        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name}: precision={metrics['precision']:.3f}, "
                      f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}, "
                      f"support={int(metrics['support'])}")

        print(f"\nOverall metrics:")
        print(f"Macro avg: precision={report['macro avg']['precision']:.3f}, "
              f"recall={report['macro avg']['recall']:.3f}, f1={report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg: precision={report['weighted avg']['precision']:.3f}, "
              f"recall={report['weighted avg']['recall']:.3f}, f1={report['weighted avg']['f1-score']:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, class_names, "Severity Classification")
        return accuracy
    
    def plot_confusion_matrix(self, cm, class_names, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def predict_sample(self, features):
        if self.abnormality_classifier is None or self.severity_classifier is None:
            raise ValueError("Models must be trained first")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        abnormality_pred = self.abnormality_classifier.predict(features_scaled)[0]
        abnormality_proba = self.abnormality_classifier.predict_proba(features_scaled)[0]
        abnormality_type = self.abnormality_encoder.inverse_transform([abnormality_pred])[0]
        
        severity_pred = self.severity_classifier.predict(features_scaled)[0]
        severity_proba = self.severity_classifier.predict_proba(features_scaled)[0]
        severity_class = self.severity_encoder.inverse_transform([severity_pred])[0]
        
        results = {
            'abnormality_type': abnormality_type,
            'abnormality_confidence': np.max(abnormality_proba),
            'severity_class': severity_class,
            'severity_confidence': np.max(severity_proba)
        }
        
        return results
    
    def train_full_pipeline(self, features_csv_path, metadata_txt_path):
        print("Starting Mammography Classification Pipeline")
        print("="*60)
        
        merged_df = self.merge_data(features_csv_path, metadata_txt_path)
        
        X_abnorm, y_abnorm, feature_cols, _ = self.prepare_data_for_abnormality_classification(merged_df)
        self.feature_columns = feature_cols
        abnormality_accuracy = self.train_abnormality_classifier(X_abnorm, y_abnorm)
        
        X_severity, y_severity, _, _ = self.prepare_data_for_severity_classification(merged_df)
        severity_accuracy = self.train_severity_classifier(X_severity, y_severity)
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Abnormality Type Classification Accuracy: {abnormality_accuracy:.4f}")
        print(f"Severity Classification Accuracy: {severity_accuracy:.4f}")
        print(f"Total features used: {len(self.feature_columns)}")
        
        return merged_df
    
    def save_model(self, model_path: str):
        if self.abnormality_classifier is None or self.severity_classifier is None:
            raise ValueError("Models must be trained before saving")

        model_data = {
            'abnormality_classifier': self.abnormality_classifier,
            'severity_classifier': self.severity_classifier,
            'scaler': self.scaler,
            'abnormality_encoder': self.abnormality_encoder,
            'severity_encoder': self.severity_encoder,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state
        }

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved successfully to: {model_path}")

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.abnormality_classifier = model_data['abnormality_classifier']
        self.severity_classifier = model_data['severity_classifier']
        self.scaler = model_data['scaler']
        self.abnormality_encoder = model_data['abnormality_encoder']
        self.severity_encoder = model_data['severity_encoder']
        self.feature_columns = model_data['feature_columns']
        self.random_state = model_data.get('random_state', 50)

        print(f"Model loaded successfully from: {model_path}")
        print(f"Feature columns: {len(self.feature_columns)}")
        print(f"Abnormality classes: {self.abnormality_encoder.classes_}")
        print(f"Severity classes: {self.severity_encoder.classes_}")

    def predict_from_images(self, mammogram_path: str, mask_path: str = None, 
                           feature_extractor=None):
        if self.abnormality_classifier is None or self.severity_classifier is None:
            raise ValueError("Models must be trained or loaded before making predictions")

        if feature_extractor is None:
            from classifier.FeatureExtractor import MammographyFeatureExtractor
            feature_extractor = MammographyFeatureExtractor()

        try:
            print(f"Extracting features from: {mammogram_path}")
            if mask_path:
                print(f"Using mask: {mask_path}")

            features_dict = feature_extractor.extract_all_features(mammogram_path, mask_path)
            feature_vector = []
            missing_features = []

            for col in self.feature_columns:
                if col in features_dict:
                    feature_vector.append(features_dict[col])
                else:
                    feature_vector.append(0.0)  
                    missing_features.append(col)

            if missing_features:
                print(f"Warning: {len(missing_features)} features were missing and set to 0")
                print(f"Missing features: {missing_features[:5]}...")  

            feature_vector = np.array(feature_vector).reshape(1, -1)
            results = self.predict_sample(feature_vector.flatten())

            results.update({
                'mammogram_path': mammogram_path,
                'mask_path': mask_path,
                'has_mask': mask_path is not None,
                'features_extracted': len(features_dict),
                'features_expected': len(self.feature_columns)
            })

            return results

        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'mammogram_path': mammogram_path,
                'mask_path': mask_path
            }

    def batch_predict_from_directory(self, image_directory: str, mask_directory: str = None,
                                    output_csv: str = None, feature_extractor=None):
        if self.abnormality_classifier is None or self.severity_classifier is None:
            raise ValueError("Models must be trained or loaded before making predictions")

        if feature_extractor is None:
            from FeatureExtractor import MammographyFeatureExtractor
            feature_extractor = MammographyFeatureExtractor()

        image_dir = Path(image_directory)
        mask_dir = Path(mask_directory) if mask_directory else None

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))
            image_files.extend(image_dir.glob(ext.upper()))

        print(f"Found {len(image_files)} image files to process")

        results = []
        successful_predictions = 0
        failed_predictions = 0

        for i, image_path in enumerate(image_files):
            try:
                mask_path = None
                if mask_dir and mask_dir.exists():
                    potential_mask = mask_dir / image_path.name
                    if potential_mask.exists():
                        mask_path = str(potential_mask)

                prediction = self.predict_from_images(str(image_path), mask_path, feature_extractor)
                prediction['filename'] = image_path.name
                prediction['image_index'] = i + 1
                results.append(prediction)

                if 'error' not in prediction:
                    successful_predictions += 1
                    print(f"✓ {image_path.name}: {prediction['abnormality_type']} "
                          f"({prediction['abnormality_confidence']:.3f}), "
                          f"{prediction['severity_class']} ({prediction['severity_confidence']:.3f})")
                else:
                    failed_predictions += 1
                    print(f"✗ {image_path.name}: {prediction['error']}")

            except Exception as e:
                failed_predictions += 1
                results.append({
                    'filename': image_path.name,
                    'image_index': i + 1,
                    'error': str(e),
                    'mammogram_path': str(image_path),
                    'mask_path': mask_path
                })
                print(f"✗ {image_path.name}: {str(e)}")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} files "
                      f"({successful_predictions} successful, {failed_predictions} failed)")

        results_df = pd.DataFrame(results)

        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")

        print(f"\nBatch prediction completed:")
        print(f"Total files: {len(image_files)}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Failed predictions: {failed_predictions}")
        print(f"Success rate: {successful_predictions/len(image_files)*100:.1f}%")

        return results_df

    def print_prediction_summary(self, prediction_result):
        if 'error' in prediction_result:
            print("❌ PREDICTION ERROR")
            print(f"Error: {prediction_result['error']}")
            return

        print("🔍 MAMMOGRAPHY ANALYSIS RESULTS")
        print("=" * 50)
        print(f"📁 Image: {os.path.basename(prediction_result['mammogram_path'])}")

        if prediction_result.get('has_mask', False):
            print(f"🎯 Mask: {os.path.basename(prediction_result['mask_path'])}")
        else:
            print("🎯 Mask: Not provided (global analysis)")

        print("\n📊 CLASSIFICATION RESULTS:")
        print("-" * 30)

        abnorm_type = prediction_result['abnormality_type']
        abnorm_conf = prediction_result['abnormality_confidence']

        if abnorm_type == 'CALC':
            abnorm_emoji = "🔸"
            abnorm_name = "Calcification"
        elif abnorm_type == 'MASS':
            abnorm_emoji = "⚫"
            abnorm_name = "Mass"
        elif abnorm_type == 'MISC':
            abnorm_emoji = "❓"
            abnorm_name = "Miscellaneous"
        else:
            abnorm_emoji = "❔"
            abnorm_name = abnorm_type

        print(f"{abnorm_emoji} Abnormality Type: {abnorm_name}")
        print(f"   Confidence: {abnorm_conf:.1%}")

        severity = prediction_result['severity_class']
        severity_conf = prediction_result['severity_confidence']

        if severity == 'M':
            severity_emoji = "🔴"
            severity_name = "Malignant"
        elif severity == 'B':
            severity_emoji = "🟡"
            severity_name = "Benign"
        else:
            severity_emoji = "⚪"
            severity_name = severity

        print(f"{severity_emoji} Severity: {severity_name}")
        print(f"   Confidence: {severity_conf:.1%}")

        print(f"\n📈 Features: {prediction_result.get('features_extracted', 'N/A')} extracted")

        if severity == 'M' and severity_conf > 0.7:
            print("⚠️  HIGH RISK: Strong indication of malignancy")
        elif severity == 'M' and severity_conf > 0.5:
            print("⚠️  MODERATE RISK: Possible malignancy")
        elif severity == 'B' and severity_conf > 0.7:
            print("✅ LOW RISK: Likely benign finding")
        else:
            print("❓ UNCERTAIN: Classification confidence is low")

def main():
    classifier = MammographyClassifier(random_state=100)
    features_csv_path = "dmid_features_all_no_nan.csv"
    metadata_txt_path = "../DMID _Dataset/Info.txt"
    try:
        merged_df = classifier.train_full_pipeline(features_csv_path, metadata_txt_path)
        model_save_path = "models/mammography_classifier.pkl"
        classifier.save_model(model_save_path)
        
        print("\n" + "="*60)
        print("EXAMPLE PREDICTION")
        print("="*60)
        
        sample_features = merged_df[classifier.feature_columns].iloc[0].values
        prediction = classifier.predict_sample(sample_features)
        
        print(f"Sample prediction:")
        print(f"Abnormality Type: {prediction['abnormality_type']} "
              f"(Confidence: {prediction['abnormality_confidence']:.3f})")
        print(f"Severity: {prediction['severity_class']} "
              f"(Confidence: {prediction['severity_confidence']:.3f})")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files. Please check:")
        print(f"- Features CSV: {features_csv_path}")
        print(f"- Metadata TXT: {metadata_txt_path}")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()