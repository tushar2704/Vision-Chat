import streamlit as st
import cv2
import os
from groq import Groq
from PIL import Image
import numpy as np
api_key = st.secrets["GROQ_API_KEY"]
# Initialize Groq client
groq_client = Groq(api_key=api_key)


def image_xray():
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Create columns for different CV operations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Grayscale Conversion")
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, use_column_width=True)
            
            st.subheader("Edge Detection")
            edges = cv2.Canny(gray_image, 100, 200)
            st.image(edges, use_column_width=True)
        
        with col2:
            st.subheader("Blur Effect")
            blurred = cv2.GaussianBlur(img_array, (15, 15), 0)
            st.image(blurred, use_column_width=True)
            
            st.subheader("Image Analysis")
            # Use Groq for image analysis
            try:
                response = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Analyze this image and describe what you see in detail"
                        }
                    ],
                    model="mixtral-8x7b-32768"
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error("Error in image analysis")
                
    # Add sidebar for parameters
    with st.sidebar:
        st.subheader("Parameters")
        blur_level = st.slider("Blur Level", 1, 31, 15, step=2)
        edge_low = st.slider("Edge Detection (Low Threshold)", 0, 200, 100)
        edge_high = st.slider("Edge Detection (High Threshold)", 0, 400, 200)
