"""
Script to convert the model to a newer TensorFlow/Keras format
Run this locally before deploying to fix compatibility issues
"""
import tensorflow as tf
from tensorflow import keras
import os

def convert_model():
    """Convert old model to new format"""
    old_model_path = 'wildlife_models/final_wildlife_model.h5'
    new_model_path = 'wildlife_models/final_wildlife_model_v2.h5'
    keras_model_path = 'wildlife_models/final_wildlife_model.keras'
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    try:
        print(f"\nLoading model from: {old_model_path}")
        # Try to load the old model
        model = keras.models.load_model(old_model_path, compile=False)
        print("✓ Model loaded successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Re-save in both formats
        print(f"\nSaving model in new H5 format to: {new_model_path}")
        model.save(new_model_path, save_format='h5')
        print("✓ Saved as H5!")
        
        print(f"\nSaving model in Keras format to: {keras_model_path}")
        model.save(keras_model_path)
        print("✓ Saved as Keras!")
        
        # Test loading the new models
        print("\nTesting new H5 model...")
        test_model_h5 = keras.models.load_model(new_model_path, compile=False)
        print("✓ New H5 model loads successfully!")
        
        print("\nTesting new Keras model...")
        test_model_keras = keras.models.load_model(keras_model_path, compile=False)
        print("✓ New Keras model loads successfully!")
        
        print("\n" + "="*60)
        print("SUCCESS! Models converted successfully!")
        print("="*60)
        print(f"\nYou can now use either:")
        print(f"  - {new_model_path}")
        print(f"  - {keras_model_path} (recommended)")
        print("\nUpload the new model file to Google Drive and update the link.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTrying alternative method...")
        
        try:
            # Alternative: Load with safe_mode=False
            model = tf.keras.models.load_model(
                old_model_path, 
                compile=False,
                safe_mode=False
            )
            print("✓ Loaded with safe_mode=False")
            
            # Save in new format
            model.save(new_model_path, save_format='h5')
            model.save(keras_model_path)
            print("✓ Models saved successfully!")
            
        except Exception as e2:
            print(f"\n❌ Alternative method also failed: {e2}")
            print("\nThe model file may be corrupted or incompatible.")
            print("You may need to retrain the model with the current TensorFlow version.")

if __name__ == "__main__":
    convert_model()
