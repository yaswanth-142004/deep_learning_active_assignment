# Wildlife Classifier - Deployment Guide

## Model File Storage (Too Large for GitHub)

The trained model file (`final_wildlife_model.h5`) is **283 MB** and exceeds GitHub's 100 MB file size limit. Here are solutions for deployment:

## Option 1: Google Drive (Recommended for Streamlit Cloud)

1. **Upload your model to Google Drive:**
   - Upload `wildlife_models/final_wildlife_model.h5.gz` to Google Drive
   - Right-click → Get link → Set to "Anyone with the link"
   - Copy the file ID from the URL: `https://drive.google.com/file/d/FILE_ID/view`

2. **Use direct download link:**
   ```
   https://drive.google.com/uc?export=download&id=FILE_ID
   ```

3. **Set environment variable in Streamlit Cloud:**
   - Go to App Settings → Secrets
   - Add: `MODEL_URL = "your_google_drive_url"`

## Option 2: Dropbox

1. Upload model to Dropbox
2. Get share link and replace `?dl=0` with `?dl=1`
3. Set as `MODEL_URL` environment variable

## Option 3: GitHub Releases

1. Create a GitHub Release
2. Attach the compressed model file as an asset
3. Use the release asset download URL

## Option 4: Hugging Face Hub (Recommended for ML Models)

1. Create account at huggingface.co
2. Upload model file
3. Use Hugging Face download URL

## Local Testing

For local testing, simply place the model file here:
```
wildlife_models/final_wildlife_model.h5
```

The app will automatically use the local file if available.

## Streamlit Cloud Deployment Steps

1. **Push code to GitHub** (without model files):
   ```bash
   git add .
   git commit -m "Add streamlit app for wildlife classification"
   git push
   ```

2. **Upload model to cloud storage** (Google Drive, Dropbox, etc.)

3. **Deploy on Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect your GitHub repository
   - Add `MODEL_URL` in Secrets/Environment variables
   - Deploy!

## Environment Variables

Set in Streamlit Cloud → App Settings → Secrets:
```toml
MODEL_URL = "https://your-storage-url/final_wildlife_model.h5.gz"
```

## Alternative: Git LFS (If you prefer)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "wildlife_models/*.h5"
git lfs track "wildlife_models/*.gz"

# Add and commit
git add .gitattributes
git add wildlife_models/
git commit -m "Add model with Git LFS"
git push
```

**Note:** GitHub LFS has bandwidth limits on free tier.

## Files Structure

```
wildlife_models/
├── class_names.json          # Include in git
├── training_history.json     # Include in git
├── final_wildlife_model.h5   # Excluded (too large)
└── final_wildlife_model.h5.gz # Excluded (too large)
```
