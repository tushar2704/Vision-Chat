import os
import streamlit as st
from groq import Groq
import pandas as pd
import numpy as np
import plotly.express as px
from langchain_groq import ChatGroq
from PIL import Image
import io
import base64
from dotenv import load_dotenv
# Load environment variables
# load_dotenv()
api_key = st.secrets["GROQ_API_KEY"]

from src.components.navigation import page_config, custom_style, footer
from src.utils.image_analysis import image_xray
from src.utils.capture import camera_observer
# Secrets and API Configuration
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)


def encode_image_to_base64(image):
    """Convert PIL Image to base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')



def analyze_image_with_groq(image):
    """
    Analyze satellite image using Groq's vision model.
    
    Args:
        image (PIL.Image): Uploaded image to analyze
    
    Returns:
        str: Detailed image analysis result
    """
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image)
        
        # Modify message structure to remove system message
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text", 
                            "text": "You are an expert imagery Xray & analysis. Now Perform a comprehensive analysis of this image. Focus on object, content, action, person, building or any other significant observations."
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Image analysis error: {e}")
        return "Unable to analyze image. Please try again."




def main():
    """Main application workflow"""
    st.title("VisionChat AI Image Analysis")
    custom_style()
    # Sidebar navigation
    page = st.sidebar.radio(
        "Choose Analysis Module",
        ["Image Analysis", "Vision LLM", "Capture Analysis"]
    )
   
    if page == "Image Analysis":
        
        image_xray()
   
    elif page == "Vision LLM":
        image_classification_module()
    
    elif page == "Capture Analysis":
        
        camera_observer()
        
        
    footer()

def image_classification_module():
    """Vision LLM Image Classification and Analysis Module"""
    st.header("🖼️ Vision LLM Image Classification")
    
    uploaded_file = st.file_uploader(
        "Upload any Image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a any image for AI analysis"
    )
   
    if uploaded_file is not None:
        # Open image with PIL
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded any Image", use_column_width=True)
        
        with col2:
            with st.spinner('Analyzing image with AI...'):
                analysis_result = analyze_image_with_groq(image)
            
            st.subheader("Vision LLM Analysis Results")
            st.write(analysis_result)




if __name__ == "__main__":
    main()