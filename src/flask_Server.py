import cv2
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from web_camera import VideoCamera
from statistics import mode
from keras.models import load_model
import numpy as np
import tensorflow as tf
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from flask import send_file
from keras import backend as K

# rtsp://10.50.249.4/live.sdp#http://10.50.197.220:8081/video.mjpg')
app = Flask(__name__, static_folder='', static_url_path='')
# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

frame_window = 10
emotion_offsets = (20, 40)
def modeling():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global face_detection
    face_detection = load_detection_model(detection_model_path)
    global emotion_classifier
    emotion_classifier = load_model(emotion_model_path, compile=False)
    global graph
    graph = tf.get_default_graph()
    global emotion_target_size
    emotion_target_size = emotion_classifier.input_shape[1:3]


# loading models
#face_detection = load_detection_model(detection_model_path)
#emotion_classifier = load_model(emotion_model_path, compile=False)
# getting input model shapes for inference
#emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

video_capture = cv2.VideoCapture(0)


def gen(camera):
    """Video streaming generator function."""
    while True:
        ret, bgr_image = video_capture.read()
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)
        count_angry = 0
        count_sad = 0
        count_happy = 0
        count_surprise = 0
        count = 0
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            print(gray_face)
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            print('Transfer To Gray Face Successful')
            with graph.as_default():
                
                emotion_prediction = emotion_classifier.predict(gray_face)
                print('Emotion Prediction Successful')
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                #print('Probability:' + str(emotion_probability))
                #print('Emotion:' + str(emotion_text))

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                    print('Emotion:' + str(emotion_text))
                    break   
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    print('Emotion:' + str(emotion_text))
                    break                   
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    print('Emotion:' + str(emotion_text))
                    break
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    print('Emotion:' + str(emotion_text))
                    break
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    print('Emotion:' + str(emotion_text))
                    break

                # if emotion_text == 'angry':
                #     count_angry = count_angry + 1
                # elif emotion_text == 'sad':
                #     count_sad = count_sad + 1
                # elif emotion_text == 'happy':
                #     count_happy = count_happy + 1
                # elif emotion_text == 'surprise':
                #     count_surprise = count_surprise + 1
                # else:
                #     count = count + 1
                # if count_angry > 1 :
                #     print('Emotion:' + str(emotion_text))
                #     break
                # if count_sad > 1 :
                #     print('Emotion:' + str(emotion_text))
                #     break
                # if count_happy > 1 :
                #     print('Emotion:' + str(emotion_text))
                #     break
                # if count_surprise > 1 :
                #     print('Emotion:' + str(emotion_text))
                #     break
                # if count > 1 :
                #     print('Emotion:' + str(emotion_text))
                #     break


                color = color.astype(int)
                color = color.tolist()


        ret, bgr_image = cv2.imencode('.jpg', bgr_image)
        bgr_image = bgr_image.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bgr_image + b'\r\n\r\n')


@app.route('/')
def index():
    return Response(
        gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hello/')
#@app.route('/hello/<name>')
def hello(name=None):
    return render_template('index.html')


if __name__ == '__main__':
    modeling()
    app.run(host='0.0.0.0', debug=True, threaded=True, port=8801)
