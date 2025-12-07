# Add this cell to your training.ipynb notebook and run it
# This will re-save your model in a format compatible with newer TensorFlow versions

import tensorflow as tf
from tensorflow import keras
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Load the existing model
model_path = 'wildlife_models/final_wildlife_model.h5'

try:
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Model loaded successfully!")
    
    # Display model summary
    print("\nModel architecture:")
    model.summary()
    
    # Save in new Keras 3 format (.keras file)
    new_model_path = 'wildlife_models/final_wildlife_model_v2.keras'
    print(f"\nSaving model in new format to: {new_model_path}")
    model.save(new_model_path)
    print("✓ Model saved in new format!")
    
    # Test loading the new model
    print("\nTesting new model...")
    test_model = keras.models.load_model(new_model_path, compile=False)
    print("✓ New model loads successfully!")
    
    print("\n" + "="*60)
    print("SUCCESS! Upload the new model file to Google Drive:")
    print(new_model_path)
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nIf the model won't load, you'll need to retrain it.")
