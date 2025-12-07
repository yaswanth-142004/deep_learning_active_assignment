import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os
import gdown
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Wildlife Image Classifier",
    page_icon="🦁",
    layout="wide"
)

# Configuration
# This is the file ID extracted from your link: 
# https://drive.google.com/file/d/1w3avcoCrXwvHTaETNfvXZlfqbXPhoBS2/view?usp=sharing.keras
MODEL_FILE_ID = '1w3avcoCrXwvHTaETNfvXZlfqbXPhoBS2'
MODEL_DIR = 'wildlife_models'
MODEL_FILENAME = 'final_wildlife_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

def get_model_path():
    """
    Checks if model exists locally. If not, downloads it from Google Drive.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Check Local
    if os.path.exists(MODEL_PATH):
        # Optional: Check if file is not empty (e.g. > 1MB)
        if os.path.getsize(MODEL_PATH) > 1024 * 1024:
            return MODEL_PATH
        else:
            st.warning("Local model file found but seems corrupted (too small). Redownloading...")
    
    # 2. Download if not found
    st.info("Model not found locally. Downloading from Google Drive...")
    try:
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            st.success("Download complete!")
            return MODEL_PATH
        else:
            st.error("Download failed. File not created.")
            return None
            
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

@st.cache_resource
def load_classification_model():
    model_path = get_model_path()
    
    if not model_path:
        return None
        
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile (optional, but good practice for inference to avoid warnings)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

@st.cache_data
def load_class_names():
    # Placeholder: You need to ensure this JSON exists or is downloaded similarly
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return json.load(f)
    else:
        # Fallback if JSON is missing (just for demo purposes)
        st.warning(f"Class names file not found at {CLASS_NAMES_PATH}. Using numeric labels.")
        return {str(i): f"Class {i}" for i in range(100)}

def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array, class_names):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Handle class name lookup safely
    idx_str = str(predicted_class_idx)
    predicted_class_name = class_names.get(idx_str, f"Unknown ({idx_str})")
    
    # Get top 5
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = []
    for idx in top_5_indices:
        name = class_names.get(str(idx), f"Class {idx}")
        score = predictions[0][idx] * 100
        top_5_predictions.append((name, score))
    
    return predicted_class_name, confidence, top_5_predictions

# Main App
def main():
    st.title("🦁 Wildlife Image Classifier")
    st.markdown("Upload an image of wildlife to classify it using the deep learning model.")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_classification_model()
        
    if model is None:
        st.error("Failed to load model. Please check logs.")
        return

    class_names = load_class_names()
    st.sidebar.success("✅ Model Ready")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Results")
            
            if st.button("Classify Image", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        img_array = preprocess_image(img)
                        pred_class, conf, top_5 = predict_image(model, img_array, class_names)
                        
                        st.success(f"Prediction: **{pred_class}**")
                        st.info(f"Confidence: **{conf:.2f}%**")
                        st.progress(min(conf/100, 1.0))
                        
                        st.markdown("---")
                        st.markdown("**Top Predictions:**")
                        for name, score in top_5:
                            st.write(f"- {name}: {score:.1f}%")
                            
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()