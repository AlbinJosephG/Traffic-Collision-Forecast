 <div align="center">

# Traffic Collision Forecast

### Deep Learning-Based Traffic Collision Prediction System Using CNN-LSTM, YOLOv8 & ByteTrack

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<br/>

> A video-based traffic collision forecasting application that detects and tracks road users, extracts vehicle trajectories and interaction features, and uses CNN-LSTM deep learning models to forecast potential collisions at multiple future time horizons. The application provides an interactive dashboard with collision-risk predictions and annotated video output.

<br/>

</div>

## ✨ Features

### Vehicle Detection & Tracking

- Detects vehicles and other road users from uploaded traffic videos.
- Performs multi-object tracking to maintain consistent object identities across video frames.
- Extracts vehicle positions and movement trajectories.
- Analyzes interactions between tracked road users.

### Trajectory & Feature Engineering

- Extracts spatial and temporal movement information from tracked objects.
- Computes trajectory-based and interaction-based features.
- Prepares fixed-rate sequential inputs for deep learning inference.
- Uses 50-frame sequences with 96 input features for collision forecasting.

### Deep Learning-Based Collision Forecasting

- Implements CNN-LSTM architecture for temporal feature learning and collision prediction.
- Uses separate trained models for three prediction horizons.
- Forecasts potential collisions at:
  - 500 milliseconds
  - 1000 milliseconds
  - 1500 milliseconds
- Displays collision-risk predictions directly on the processed video.

### Interactive Streamlit Dashboard

- Provides video upload and processing functionality.
- Displays detected and tracked road users.
- Shows collision predictions across multiple future horizons.
- Generates annotated video output with prediction overlays.
- Presents model evaluation metrics.

### 📈 Model Evaluation

- Evaluates collision forecasting performance using Average Precision (AP) at multiple time-to-event horizons.
- Previously reported test-set results:
  - **AP @ 500 ms:** 68.37%
  - **AP @ 1000 ms:** 79.78%
  - **AP @ 1500 ms:** 74.81%
  - **Overall mAP:** 74.32%

---

## 🏗️ Processing Pipeline

```text
Traffic Video Input
        |
        ▼
YOLOv8 Object Detection
        |
        ▼
ByteTrack Multi-Object Tracking
        |
        ▼
Vehicle Trajectory Extraction
        |
        ▼
Spatial & Temporal Interaction Features
        |
        ▼
50-Frame Sequence Preparation
        |
        ▼
CNN-LSTM Collision Forecasting
        |
        ▼
500 ms | 1000 ms | 1500 ms Predictions
        |
        ▼
Annotated Video Output
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch, CNN-LSTM |
| Object Detection | YOLOv8 |
| Object Tracking | ByteTrack |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Web Application | Streamlit |
| Configuration | YAML |
| Development Tools | Git, GitHub |

---

## 🎯 Key Highlights

- Integrated object detection, multi-object tracking, trajectory extraction, and temporal interaction modeling into a unified collision forecasting pipeline.
- Developed multi-horizon CNN-LSTM models to forecast potential traffic collisions using sequential vehicle movement features.
- Achieved a previously reported **74.32% collision prediction mAP** 


---

## OUTPUT SCREENSHOTS

### DASHBOARD
<img width="1885" height="903" alt="image" src="https://github.com/user-attachments/assets/54c0c17d-73f6-4b5f-a8ec-0287445019a4" />
<img width="1872" height="867" alt="image" src="https://github.com/user-attachments/assets/81be125c-f0b7-4e3f-8dad-4472a36fb387" />

### COLLISION RESULT

https://github.com/user-attachments/assets/1e962d2c-f45c-4583-8117-740a5f15d695

### NON COLLISION RESULT

https://github.com/user-attachments/assets/11589a6d-cb09-4616-94e5-d4a721c6d481




