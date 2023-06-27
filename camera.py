import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras import backend as K

# Скачать данные можно по этой ссылке: https://drive.google.com/drive/folders/18n4XnjsXSi4iNpw6VmEWPHq-_wWOes3R?usp=sharing
class emotion_model():
    def __init__(self, path_to_model):
        super().__init__()
        self.model = load_model(path_to_model)

        self.emotion_dict = {0: 'anger',
                             1: 'contempt',
                             2: 'disgust',
                             3: 'fear',
                             4: 'happy',
                             5: 'neutral',
                             6: 'sad',
                             7: 'surprise',
                             8: 'uncertain'}

    def predict(self, img):
        img = tf.keras.preprocessing.image.smart_resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input_facenet(img)

        return self.emotion_dict[np.argmax(self.model.predict(img))]

    def preprocess_input_facenet(self, x):
        x_temp = np.copy(x)
        data_format = K.image_data_format()

        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912

        return x_temp

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = emotion_model("model.h5")

while(True):
    ret, frame = cam.read()
    faces = face_detector.detectMultiScale(frame)
    if len(faces) >0:
        one_face = faces[0]
        x, y, w, h = one_face
        face_boundingbox_bgr = np.copy(frame)
        face_boundingbox_bgr = face_boundingbox_bgr[y:y + h, x:x + w]
        emotion = model.predict(face_boundingbox_bgr)

        rgb_image_with_boundingbox = cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 3)

        rgb_image_with_boundingbox_and_text = cv2.putText(rgb_image_with_boundingbox, emotion, (y, x - 10),
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("facial emotion recognition", rgb_image_with_boundingbox_and_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break