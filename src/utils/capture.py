import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

class ActionDetector:
    def __init__(self):
        # Initialize HOG descriptor for people detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.prev_frame = None

    def detect_people(self, image):
        """
        Detect people in the image using HOG descriptor
        
        Args:
            image (numpy.ndarray): Input image in BGR color space
        
        Returns:
            tuple: Detected boxes and their weights
        """
        try:
            boxes, weights = self.hog.detectMultiScale(
                image,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            return boxes, weights
        except Exception as e:
            st.error(f"Error in people detection: {e}")
            return [], []

    def detect_motion(self, current_frame):
        """
        Detect motion using frame differencing
        
        Args:
            current_frame (numpy.ndarray): Grayscale current frame
        
        Returns:
            list: Contours of detected motion
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return []

        try:
            # Calculate absolute difference between frames
            frame_diff = cv2.absdiff(self.prev_frame, current_frame)
            
            # Apply threshold
            _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Update previous frame
            self.prev_frame = current_frame
            
            # Filter contours by area
            significant_contours = [
                contour for contour in contours 
                if cv2.contourArea(contour) > 500
            ]
            
            return significant_contours
        except Exception as e:
            st.error(f"Error in motion detection: {e}")
            return []

    def process_image(self, image):
        """
        Process the image for people and motion detection
        
        Args:
            image (numpy.ndarray): Input image in BGR color space
        
        Returns:
            numpy.ndarray: Processed image with detections
        """
        # Create a copy of the image to avoid modifying the original
        processed_image = image.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # People detection
        boxes, _ = self.detect_people(image)
        for (x, y, w, h) in boxes:
            cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(processed_image, 'Person Detected', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Motion detection
        contours = self.detect_motion(gray)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(processed_image, 'Motion Detected', (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return processed_image

def camera_observer():
    """
    Streamlit UI for camera and image input action detection
    """
    st.subheader("📸 Action Detection")
    
    # Create an instance of ActionDetector
    detector = ActionDetector()
    
    # Create two columns for input selection
    col1, col2 = st.columns(2)
    with col1:
        use_camera = st.checkbox("Use Camera Input")
    with col2:
        use_upload = st.checkbox("Upload Image")
    
    # Input selection logic
    input_image = None
    if use_camera:
        input_image = st.camera_input("Take a picture")
    elif use_upload:
        input_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    # Process the image if input is available
    if input_image:
        try:
            # Open and convert image
            image = Image.open(input_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process the image
            processed_image = detector.process_image(image_cv)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), 
                         caption="Processed Image")
            with col2:
                st.write("Detection Results:")
                st.write(f"- Number of people detected: {len(detector.detect_people(image_cv)[0])}")
                
                contours = detector.detect_motion(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY))
                if contours:
                    st.write(f"- Motion detected: {len(contours)} significant motion areas")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
