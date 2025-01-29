import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
from ultralytics import YOLO
from preprocessing import Preprocessing  # Ensure this class is in preprocessing.py


# Define class colors for visualization
CLASS_COLORS = {
    0: (0, 255, 0),   # Pedestrian - Green
    2: (205, 0, 0),   # Car - Blue
    7: (0, 0, 255),   # Truck - Red
    9: (255, 255, 0)  # Signal - Yellow
}

CLASS_NAMES = {
    0: 'Pedestrian',
    2: 'Car',
    7: 'Truck',
    9: 'Signal'
}


def draw_filtered_detections(frame, detections):
    """
    Draws bounding boxes with class labels and different colors on the frame.
    """
    for (x, y, w, h, confidence, class_id) in detections:
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default white if class not found
        label = f"{CLASS_NAMES.get(class_id, 'Unknown')} {confidence:.2f}"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Add text label
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def process_video_streams(input_video_path, output_video_path, model_path="yolov8s.pt"):
    model = YOLO(model_path)
    allowed_classes = list(CLASS_NAMES.keys())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    preprocessor = Preprocessing()
    
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    combined_width = frame_width * 2
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (combined_width, frame_height))

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO on original frame
        results_unprocessed = model(frame, device=device)[0]
        detections_unprocessed = extract_detections(results_unprocessed, allowed_classes)
        annotated_unprocessed = draw_filtered_detections(frame.copy(), detections_unprocessed)
        
        # Apply preprocessing
        processed_frame = preprocessor.preprocess_frame(frame)
        
        # Run YOLO on processed frame
        results_processed = model(processed_frame, device=device)[0]
        detections_processed = extract_detections(results_processed, allowed_classes)
        annotated_processed = draw_filtered_detections(processed_frame.copy(), detections_processed)
        
        # Combine both frames
        combined_frame = np.hstack((annotated_unprocessed, annotated_processed))
        out.write(combined_frame)
    
    cap.release()
    out.release()


def extract_detections(results, allowed_classes):
    """Extracts valid detections from YOLO results."""
    detections = []
    for result in results.boxes.data:
        x1, y1, x2, y2, confidence, class_id = result.cpu().numpy()
        if int(class_id) in allowed_classes:
            detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence, int(class_id)))
    return detections


def main():
    st.set_page_config(
        page_title="Fog Removal System",
        page_icon="background.jpg",
        layout="wide")
    # Streamlit page title
    col1, col2 = st.columns([1, 5])  # Adjust column width ratios for better layout

    with col1: 
        st.image("background.jpg", width=230) 

    with col2: 
        st.title("FOG REMOVAL SYSTEM FOR AUTONOMOUS VEHICLES")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output.close()
        
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                process_video_streams(temp_input.name, temp_output.name)
                st.success("Processing complete!")

            # Close the file properly before loading it in Streamlit
            with open(temp_output.name, "rb") as file:
                video_bytes = file.read()
            
            st.video(video_bytes)  # Load the video correctly

            st.download_button("Download Processed Video", data=open(temp_output.name, 'rb'),
                            file_name="processed_video.mp4", mime="video/mp4")


if __name__ == "__main__":
    main()
