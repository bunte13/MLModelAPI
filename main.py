from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import asyncio

from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.status import HTTP_302_FOUND

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('models/action.h5')
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define constants
ACTIONS = ["hello", "eating", "iloveyou"]
COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define constants for the model
TIME_STEPS = 30
FEATURES = 225
threshold = 0.8

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

async def video_streamer(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    sequence = []
    sentence = []
    predictions = []

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                frame = cv2.flip(frame, 1)
                try:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-TIME_STEPS:]

                    if len(sequence) == TIME_STEPS:
                        # Ensure sequence has the correct shape (batch_size, time_steps, features)
                        sequence_np = np.expand_dims(np.array(sequence), axis=0)  # Add batch dimension

                        # Predict
                        res = model.predict(sequence_np)[0]
                        predictions.append(np.argmax(res))

                        # Determine the current action
                        if np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if ACTIONS[np.argmax(res)] != sentence[-1]:
                                        sentence.append(ACTIONS[np.argmax(res)])
                                else:
                                    sentence.append(ACTIONS[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # Visualize the results
                        image = prob_viz(res, ACTIONS, image, COLORS)

                        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    _, buffer = cv2.imencode('.jpg', image)
                    frame_bytes = buffer.tobytes()

                    # Check if the WebSocket is still open before sending data
                    await websocket.send_bytes(frame_bytes)
                    await asyncio.sleep(0.05)  # Adjust frame rate if necessary

                except WebSocketDisconnect:
                    print("WebSocket disconnected")
                    break

                except Exception as e:
                    print(f"Error in frame processing: {e}")

    except Exception as e:
        print(f"Error in video stream: {e}")

    finally:
        cap.release()
        print("Released video capture")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await video_streamer(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
def test(request: Request):
    return RedirectResponse(url="/static/index.html",status_code=HTTP_302_FOUND)
