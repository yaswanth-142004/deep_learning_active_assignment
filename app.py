import os
# [IMPORTANT] Set these environment variables BEFORE importing tensorflow
# This forces the use of legacy Keras to match your older model file
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import gdown
from PIL import Image
import json

# Try importing TensorFlow and handle the version mismatch gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except ImportError:
    st.error("TensorFlow not found. Please add 'tensorflow' to requirements.txt")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Wildlife Image Classifier",
    page_icon="🦁",
    layout="wide"
)

# Configuration
MODEL_FILE_ID = '1w3avcoCrXwvHTaETNfvXZlfqbXPhoBS2'
MODEL_DIR = 'wildlife_models'
MODEL_FILENAME = 'final_wildlife_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

def get_model_path():
    """Checks if model exists locally. If not, downloads it from Google Drive."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Check Local
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) > 10 * 1024 * 1024: # Check if > 10MB
            return MODEL_PATH
        else:
            st.warning("Local model file found but seems corrupted (too small). Redownloading...")
            try:
                os.remove(MODEL_PATH)
            except:
                pass
    
    # 2. Download if not found
    with st.spinner("Model not found locally. Downloading from Google Drive (100MB)..."):
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
        # Attempt to load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile for inference
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except ValueError as e:
        if "batch_normalization" in str(e):
            st.error("⚠️ **Compatibility Error Detected**")
            st.warning("""
            The model was trained with an older version of TensorFlow/Keras.
            
            **To fix this on Streamlit Cloud:**
            1. Create a file named `requirements.txt` in your repo (if not exists).
            2. Add exactly this line: `tensorflow<2.16`
            3. Reboot the app.
            """)
            st.code("tensorflow<2.16", language="text")
        st.error(f"Error details: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    # Placeholder: Ensure this JSON exists or uses a fallback
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return json.load(f)
    else:
        # Fallback list for demo purposes
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
    
    idx_str = str(predicted_class_idx)
    predicted_class_name = class_names.get(idx_str, f"Unknown ({idx_str})")
    
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
    st.markdown("Upload an image of wildlife to classify it.")
    
    model = load_classification_model()
        
    if model is None:
        st.stop()

    class_names = load_class_names()
    st.sidebar.success("✅ Model Ready")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("Classify Image", type="primary"):
                with st.spinner("Analyzing..."):
                    img = Image.open(uploaded_file)
                    img_array = preprocess_image(img)
                    pred_class, conf, top_5 = predict_image(model, img_array, class_names)
                    
                    st.success(f"Prediction: **{pred_class}**")
                    st.info(f"Confidence: **{conf:.2f}%**")
                    st.progress(min(conf/100, 1.0))
                    
                    st.markdown("---")
                    st.write("**Top Predictions:**")
                    for name, score in top_5:
                        st.write(f"- {name}: {score:.1f}%")

if __name__ == "__main__":
    main()