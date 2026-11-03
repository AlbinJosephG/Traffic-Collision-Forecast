TRAFFIC COLLISION VIDEO APP

1. Copy app.py and requirements.txt into E:\TrafficCollisionProject\
2. Open PowerShell and run:
   cd E:\TrafficCollisionProject
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
3. Confirm these existing project files are present:
   models\yolov8n.pt
   configs\bytetrack_low_conf.yaml
   models\phase_6_3_architecture_comparison\cnn_lstm_500ms.pt
   models\phase_6_3_architecture_comparison\cnn_lstm_1000ms.pt
   models\phase_6_3_architecture_comparison\cnn_lstm_1500ms.pt
   processed\fixed_rate_sequences\sequence_schema.json
   processed\fixed_rate_sequences\preprocessing_config.json
4. Launch:
   streamlit run app.py
5. Open the local address Streamlit prints, upload a video, and click Process video.

NOTES
- The first prediction requires 50 sampled observations (9.8 seconds of history).
- Detection/tracking is run at 5 Hz. The most recent box overlay is held between sampled frames.
- Model outputs are experimental uncalibrated sigmoid scores, not verified collision probabilities.
- This is a first integration implementation. Validate feature parity against a known training video before relying on any output.
