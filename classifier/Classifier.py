from classifier.FeatureExtractor import MammographyFeatureExtractor
from RandomForest import MammographyClassifier
import os

def load_model():
    print("📂 LOADING EXISTING MODEL")
    print("=" * 60)
    
    classifier = MammographyClassifier()
    model_path = "./models/mammography_classifier.pkl"
    
    try: 
        classifier.load_model(model_path)
        return classifier
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def predict_single_image(classifier, mammogram_path, mask_path=None):
    print("\n🔍 SINGLE IMAGE PREDICTION")
    print("=" * 60)
    
    if not os.path.exists(mammogram_path):
        print(f"❌ Mammogram not found: {mammogram_path}")
        return None
    
    print(f"📁 Mammogram: {os.path.basename(mammogram_path)}")
    
    has_roi = mask_path is not None and os.path.exists(mask_path)
    
    if not has_roi:
        result = {
            'abnormality_type': 'NORM',
            'abnormality_confidence': 1.0,
            'severity_class': 'N',
            'severity_confidence': 1.0,
            'mammogram_path': mammogram_path,
            'mask_path': None,
            'has_mask': False,
            'classification_reason': 'No ROI detected'
        }
        
        print_norm_classification_summary(result)
        return result
    
    print(f"🎯 ROI Mask: {os.path.basename(mask_path)}")
    
    feature_extractor = MammographyFeatureExtractor()
    
    try:
        result = classifier.predict_from_images(
            mammogram_path, 
            mask_path,
            feature_extractor
        )
        
        result['classification_reason'] = 'ROI-based analysis'
        
        classifier.print_prediction_summary(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def print_norm_classification_summary(result):
    print("\n🔍 MAMMOGRAPHY ANALYSIS RESULTS")
    print("=" * 50)
    print(f"📁 Image: {os.path.basename(result['mammogram_path'])}")
    
    if result.get('mask_path'):
        print(f"🎯 Mask: {os.path.basename(result['mask_path'])}")
    else:
        print("🎯 Mask: Not found")
    
    print(f"📋 Reason: {result.get('classification_reason', 'No ROI detected')}")
    
    print("\n📊 CLASSIFICATION RESULTS:")
    print("-" * 30)
    print("✅ Classification: NORMAL")
    print("   No abnormalities detected")
    print("   Confidence: 100%")
    
    if 'error' in result:
        print(f"\n⚠️  Note: Classification failed, defaulted to normal")
        print(f"   Error: {result['error']}")

def main():
    print("🏥 MAMMOGRAPHY CLASSIFICATION PIPELINE")
    print("=" * 70)

    classifier = load_model()
    if classifier is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    mammogram_path = "../../MIAS Dataset/MIAS/mdb028.png"
    mask_path = "../../Detector/single_predictions/mdb028.png"

    result = predict_single_image(classifier, mammogram_path, mask_path)

    print("\n✅ Pipeline completed!")

if __name__ == "__main__":
    main()
