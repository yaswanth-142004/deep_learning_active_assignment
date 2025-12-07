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
# IMPORTANT: Add .keras or .h5 at the end to specify format
MODEL_URL = os.environ.get('MODEL_URL', 'https://drive.google.com/file/d/1w3avcoCrXwvHTaETNfvXZlfqbXPhoBS2/view?usp=sharing.keras')  # Set this in Streamlit Cloud secrets or environment

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
            
            # Use gdown with fuzzy mode for better compatibility
            download_url = f'https://drive.google.com/uc?id={file_id}'
            try:
                # Try with fuzzy=True to handle virus scan warning better
                gdown.download(download_url, destination, quiet=False, fuzzy=True)
                
                # Verify the downloaded file is valid
                if os.path.exists(destination):
                    file_size = os.path.getsize(destination)
                    if file_size < 10 * 1024 * 1024:  # Less than 10MB is suspicious
                        st.warning(f"Downloaded file is only {file_size / (1024*1024):.2f} MB. Retrying with alternative method...")
                        os.remove(destination)
                        # Try alternative download
                        gdown.cached_download(download_url, destination, quiet=False)
                
                return True
            except Exception as e:
                st.error(f"gdown failed: {str(e)}")
                return False
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
    # Try both .keras and .h5 formats
    keras_model_path = 'wildlife_models/final_wildlife_model.keras'
    h5_model_path = 'wildlife_models/final_wildlife_model.h5'
    compressed_h5_path = 'wildlife_models/final_wildlife_model.h5.gz'
    compressed_keras_path = 'wildlife_models/final_wildlife_model.keras.gz'
    
    # Create directory if it doesn't exist
    Path('wildlife_models').mkdir(exist_ok=True)
    
    # Check if either format exists
    if os.path.exists(keras_model_path) or os.path.exists(h5_model_path):
        return True
    
    # If model doesn't exist, try to decompress or download
    if os.path.exists(compressed_keras_path):
        with st.spinner('Decompressing Keras model file...'):
            with gzip.open(compressed_keras_path, 'rb') as f_in:
                with open(keras_model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return True
    elif os.path.exists(compressed_h5_path):
        with st.spinner('Decompressing H5 model file...'):
            with gzip.open(compressed_h5_path, 'rb') as f_in:
                with open(h5_model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return True
    elif MODEL_URL:
        with st.spinner('Downloading model file (this may take a few minutes)...'):
            # Determine destination based on URL extension
            if '.keras' in MODEL_URL or 'keras' in MODEL_URL.lower():
                if MODEL_URL.endswith('.gz'):
                    destination = compressed_keras_path
                    final_path = keras_model_path
                else:
                    destination = keras_model_path
                    final_path = None
            else:
                if MODEL_URL.endswith('.gz'):
                    destination = compressed_h5_path
                    final_path = h5_model_path
                else:
                    destination = h5_model_path
                    final_path = None
            
            if download_model(MODEL_URL, destination):
                # Decompress if needed
                if destination.endswith('.gz') and final_path:
                    with gzip.open(destination, 'rb') as f_in:
                        with open(final_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                return True
            return False
    else:
        st.error("Model file not found. Please upload the model or set MODEL_URL.")
        return False

# Load the model and class names
@st.cache_resource
def load_classification_model():
    # Ensure model is available
    if not decompress_model_if_needed():
        return None
    
    # Try .keras format first (newer, more compatible)
    keras_model_path = 'wildlife_models/final_wildlife_model.keras'
    h5_model_path = 'wildlife_models/final_wildlife_model.h5'
    
    model_path = keras_model_path if os.path.exists(keras_model_path) else h5_model_path
    
    if not os.path.exists(model_path):
        return None
    
    try:
        import h5py
        
        st.info(f"Loading model from: {os.path.basename(model_path)}")
        
        # Check file size and validate it's a proper file
        file_size = os.path.getsize(model_path)
        st.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            st.error("Downloaded file is too small. It may be an error page from Google Drive.")
            st.warning("""
            **Google Drive Download Issue**
            
            The file downloaded is too small to be the actual model. This usually means:
            1. The Google Drive link is incorrect
            2. The file permissions are not set to "Anyone with the link can view"
            3. Google Drive returned an error page instead of the file
            
            Please verify:
            - Your Google Drive link is publicly accessible
            - You're using the correct file ID
            - The file in Google Drive is the actual .keras or .h5 model file
            """)
            return None
        
        # Verify it's a valid HDF5 file
        try:
            with h5py.File(model_path, 'r') as f:
                st.success(f"✓ Valid HDF5 file detected")
        except Exception as e:
            st.error(f"File is not a valid HDF5/Keras model: {str(e)}")
            st.warning("The downloaded file might be an HTML error page. Check your Google Drive sharing settings.")
            return None
        
        # Try loading with safe_mode=False for compatibility
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
            st.success("✓ Model loaded successfully with safe_mode=False!")
        except:
            # Fallback: Load with default settings
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success("✓ Model loaded successfully!")
        
        # Recompile the model with fresh optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error loading model: {error_msg}")
        
        # Provide helpful error message
        if "file signature not found" in error_msg.lower():
            st.warning("""
            **Download Error: Invalid File**
            
            The downloaded file is not a valid model file. This typically means Google Drive 
            returned an HTML page instead of your file.
            
            **Solutions:**
            1. Make sure your file is shared as "Anyone with the link can view"
            2. Try using the .keras format instead of .h5
            3. Use a different file hosting service (Dropbox, Hugging Face, etc.)
            4. For local testing, place the model file directly in: wildlife_models/
            """)
        elif "Invalid dtype" in error_msg or "tuple" in error_msg:
            st.warning("""
            **Model Compatibility Issue**
            
            Run this in your training notebook to convert the model:
            ```python
            import tensorflow as tf
            model = tf.keras.models.load_model('wildlife_models/final_wildlife_model.h5', compile=False)
            model.save('wildlife_models/final_wildlife_model_v2.keras')
            ```
            Then upload the .keras file to Google Drive.
            """)
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
