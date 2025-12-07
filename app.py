import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os
import gzip
import shutil
import requests
from pathlib import Path
import gdown

# Set page configuration
st.set_page_config(
    page_title="Wildlife Image Classifier",
    page_icon="🦁",
    layout="wide"
)

# Model download URL - Replace this with your actual file URL
# For Google Drive: Use direct download link
# For Dropbox: Replace ?dl=0 with ?dl=1
# For GitHub Releases: Use the release asset URL
MODEL_URL = os.environ.get('MODEL_URL', 'https://drive.google.com/uc?export=download&id=147a4ElVj6Pg2m9k26REa98iCrzxKb4Hx')  # Set this in Streamlit Cloud secrets or environment

def download_model(url, destination):
    """Download model file from URL"""
    if not url:
        return False
    
    try:
        # Check if it's a Google Drive URL
        if 'drive.google.com' in url:
            # Extract file ID
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                st.error("Invalid Google Drive URL format")
                return False
            
            # Use gdown for reliable Google Drive downloads
            download_url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(download_url, destination, quiet=False)
            return True
        else:
            # Regular download for other URLs
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
            return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

def decompress_model_if_needed():
    """Decompress the model file if it doesn't exist"""
    model_path = 'wildlife_models/final_wildlife_model.h5'
    compressed_path = 'wildlife_models/final_wildlife_model.h5.gz'
    
    # Create directory if it doesn't exist
    Path('wildlife_models').mkdir(exist_ok=True)
    
    # If model doesn't exist
    if not os.path.exists(model_path):
        # Try to decompress if compressed version exists
        if os.path.exists(compressed_path):
            with st.spinner('Decompressing model file...'):
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(model_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            return True
        # Try to download if URL is provided
        elif MODEL_URL:
            with st.spinner('Downloading model file (this may take a few minutes)...'):
                if MODEL_URL.endswith('.gz'):
                    # Download compressed and decompress
                    if download_model(MODEL_URL, compressed_path):
                        with gzip.open(compressed_path, 'rb') as f_in:
                            with open(model_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        return True
                else:
                    # Download directly
                    return download_model(MODEL_URL, model_path)
        else:
            st.error("Model file not found. Please upload the model or set MODEL_URL.")
            return False
    return True

# Load the model and class names
@st.cache_resource
def load_classification_model():
    # Ensure model is available
    if not decompress_model_if_needed():
        return None
    
    model_path = 'wildlife_models/final_wildlife_model.h5'
    if not os.path.exists(model_path):
        return None
    
    try:
        # Try loading with compile=False to avoid optimizer issues
        model = load_model(model_path, compile=False)
        
        # Recompile the model with a simple loss function
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Try alternative loading method
        try:
            import h5py
            # Load with custom settings
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=None,
                compile=False,
                safe_mode=False
            )
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e2:
            st.error(f"Alternative loading also failed: {str(e2)}")
            return None

@st.cache_data
def load_class_names():
    with open('wildlife_models/class_names.json', 'r') as f:
        class_names = json.load(f)
    return class_names

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Resize image
    img = img.resize(target_size)
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values (assuming model was trained with normalized data)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array, class_names):
    """Make prediction and return results"""
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class_name = class_names[str(predicted_class_idx)]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [
        (class_names[str(idx)], predictions[0][idx] * 100)
        for idx in top_5_indices
    ]
    
    return predicted_class_name, confidence, top_5_predictions

# Main App
def main():
    # Header
    st.title("🦁 Wildlife Image Classifier")
    st.markdown("Upload an image of wildlife to classify it using our deep learning model!")
    st.markdown("---")
    
    # Load model and class names
    try:
        model = load_classification_model()
        if model is None:
            st.error("⚠️ Could not load the model. Please check the setup.")
            st.info("For local testing, place the model file in: wildlife_models/final_wildlife_model.h5")
            return
            
        class_names = load_class_names()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info(
        f"This classifier can identify **{len(class_names)}** different wildlife species. "
        "Upload an image and get instant predictions!"
    )
    
    st.sidebar.markdown("### Supported Animals")
    st.sidebar.markdown("Lion, Tiger, Bear, Elephant, Zebra, Giraffe, and many more!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a wildlife animal"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Classification Results")
            
            # Add a classify button
            if st.button("Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess the image
                        img_array = preprocess_image(img)
                        
                        # Make prediction
                        predicted_class, confidence, top_5 = predict_image(
                            model, img_array, class_names
                        )
                        
                        # Display main prediction
                        st.success("Classification Complete!")
                        st.markdown(f"### **Predicted Animal:** {predicted_class}")
                        st.markdown(f"### **Confidence:** {confidence:.2f}%")
                        
                        # Display confidence as progress bar
                        st.progress(confidence / 100)
                        
                        # Display top 5 predictions
                        st.markdown("---")
                        st.markdown("#### Top 5 Predictions:")
                        
                        for i, (class_name, conf) in enumerate(top_5, 1):
                            st.write(f"{i}. **{class_name}**: {conf:.2f}%")
                            st.progress(conf / 100)
                        
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
    
    else:
        # Show example message when no image is uploaded
        st.info("👆 Please upload an image to get started!")
        
        # Display some sample information
        st.markdown("### How to use:")
        st.markdown("1. Click on 'Browse files' button above")
        st.markdown("2. Select an image of a wildlife animal")
        st.markdown("3. Click 'Classify Image' button")
        st.markdown("4. View the prediction results!")

if __name__ == "__main__":
    main()
