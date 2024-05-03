import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# Feature Extraction Model
feature_extractor = keras.applications.vgg16.VGG16(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)

# Preprocess Image (updated)
def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    image = keras.applications.vgg16.preprocess_input(image)  
    features = feature_extractor.predict(image)
    return features.reshape(1, -1)  # Reshaping for KNN

# Load Dataset
def load_dataset(dataset_path):
    images = []
    features = []
    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        image = Image.open(img_path).convert('RGB')
        features.append(preprocess_image(image).flatten()) 
        images.append(img_path)
    return images, np.array(features)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5) 

# Streamlit UI (updated with small improvement)
st.title("Image Similarity Search")
uploaded_image = st.file_uploader("Upload an Image", type=['jpg'])

if uploaded_image:
    with st.spinner("Loading and searching..."): 
        user_image_features = preprocess_image(Image.open(uploaded_image))
        user_image_features = user_image_features.reshape(1, -1)  # Reshape for KNN

        # Load dataset and fit KNN Model (Do this ONLY once at the start)
        if 'dataset_features' not in st.session_state:  # Check if dataset is loaded
            images, dataset_features = load_dataset('dataset') 
            dataset_features = dataset_features.reshape((dataset_features.shape[0], -1))
            knn.fit(dataset_features, images)
            st.session_state['dataset_features'] = dataset_features

        # Find Similar Images (Always do this) 
        distances, indices = knn.kneighbors(user_image_features)
        similar_images = [images[i] for i in indices[0]]

    # Display Results 
    st.image(uploaded_image, caption="Uploaded Image")
    st.subheader("Similar Images:")
    cols = st.columns(4) 
    for i, img_path in enumerate(similar_images):
        cols[i % 4].image(Image.open(img_path))