## 🚘 Fog Removal System for Autonomous Vehicles

This project is a real-time video processing system that enhances visibility in foggy conditions using image dehazing techniques and YOLOv8 object detection. It compares original and enhanced detection results side by side, making it ideal for evaluating the impact of preprocessing on object recognition in autonomous driving.


🔧 Features
- 🔍 Image Dehazing using Dark Channel Prior, Transmission Map Estimation, Guided Filtering, and Gamma Correction
- 📦 CLAHE for local contrast enhancement
- 🎯 YOLOv8 object detection before and after preprocessing
- 📊 Side-by-side comparison of raw and processed detections
- 🎞️ Full video pipeline processing with Streamlit interface
- ⚡ GPU support via PyTorch (CUDA if available)

📁 File Structure
fog-removal-system/
├── app.py                 # Streamlit frontend for video processing
├── preprocessing.py       # Preprocessing pipeline class
├── requirements.txt       # Required Python packages
├── background.jpg         # Logo or banner for UI
├── README.md              # You're here!

🧠 How It Works
- Preprocessing (Preprocessing.py)
- Dark Channel Prior estimation
- Atmospheric light estimation
- Transmission map and guided filter refinement
- Gamma correction and CLAHE for enhanced contrast
- Object Detection
- YOLOv8 inference on original and preprocessed frames
- Filters detections for key classes: Pedestrian, Car, Truck, Traffic Signal
- Comparison Visualization
- Annotates both original and enhanced frames
- Merges them side-by-side into a new output video

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/your-username/fog-removal-system.git
cd fog-removal-system
2. Install Dependencies
pip install -r requirements.txt
3. Run the App
streamlit run app.py
Make sure you have the YOLOv8 weights file (e.g., yolov8s.pt). Download it from Ultralytics.

📹 Demo Workflow
- Upload any .mp4, .avi, .mkv, or .mov video
- Click "Process Video"
- Watch side-by-side results of raw vs fog-removed detection
- Download the processed output


📦 Dependencies
- opencv-python
- numpy
- torch
- streamlit
- ultralytics

Add to your requirements.txt:
- opencv-python
- numpy
- torch
- streamlit
- ultralytics

💡 Use Cases
- Improving perception in foggy environments for autonomous vehicles
- Comparative evaluation of detection performance with and without preprocessing
- Prototype for ADAS (Advanced Driver Assistance Systems)

🙌 Acknowledgements
- Ultralytics YOLOv8
- Dark Channel Prior and Guided Filtering techniques
- Streamlit for intuitive video-based interfaces
