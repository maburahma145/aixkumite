from flask import Flask, request, send_file
import os
import tempfile
import cv2
import mediapipe as mp
from datetime import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


#app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), f"processed_{datetime.now().timestamp()}.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # MediaPipe Pose is for single-person — to use it on multiple people,
    # you'd normally detect each person first (e.g. with YOLO), then crop and run pose per crop.
    # This basic example processes the full frame.

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            out.write(frame)

    cap.release()
    out.release()
    return output_path

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        video.save(temp_in.name)
        output_path = process_video(temp_in.name)
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name='result.mp4'
        )

if __name__ == '__main__':
    app.run(debug=True)
