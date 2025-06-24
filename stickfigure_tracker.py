from flask import Flask, request, send_file
import tempfile, os
import cv2
import mediapipe as mp
from datetime import datetime

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(tempfile.gettempdir(), f"processed_{datetime.now().timestamp()}.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(frame)

    cap.release()
    out.release()
    return output_path

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        video.save(temp_in.name)
        result_path = process_video(temp_in.name)
        return send_file(result_path, mimetype='video/mp4', as_attachment=False)

@app.route('/')
def hello():
    return "Stick Figure Tracker API is running."

if __name__ == '__main__':
    app.run()
