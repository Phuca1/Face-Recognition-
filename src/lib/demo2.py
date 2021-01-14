import tkinter as tk
from tkinter import filedialog
import cv2
from tensorflow import keras
import numpy as np
import dlib

face_detector = dlib.get_frontal_face_detector()
name_arr = ["Hoa Vinh", "Huan Hoa Hong", "JVevermind", "My Tam", "Son Tung", "Tran Dan"]


def extract_video(video_path, model_path):
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    model = keras.models.load_model(model_path)
    bool_arr = np.array([False] * 6, dtype=bool)
    count = 0
    while success:
        if count % 10 == 0:
            try:
                faces = face_detector(frame, 1)
                for face in faces:
                    face_raw = frame[face.top():face.bottom(), face.left():face.right(), :]
                    img = cv2.resize(face_raw, (224, 224))
                    arr = np.asarray(img)
                    rs = model.predict_classes(arr.reshape(1, 224, 224, 3))
                    for item in rs:
                        bool_arr[item] = True
                    print(rs)
                # cv2.imshow('face',face)
                # cv2.waitKey()
            except Exception:
                print("something went wrong")
        count += 1
        success, frame = video.read()
    for i in range(len(bool_arr)):
        if (bool_arr[i]):
            print(name_arr[i])



if __name__ == "__main__":
    # root = tk.Tk()
    # video_path = filedialog.askopenfilename()

    video_path = "/home/ntp/DeepLearning/face-recognition/src/video/TranDan_SonTung_HoaVinh_2.mp4"
    model_path = '/home/ntp/DeepLearning/face-recognition/src/model/model4_6class_vgg16'

    extract_video(video_path, model_path)
