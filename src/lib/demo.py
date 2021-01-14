from tensorflow import keras
import numpy as np
import cv2
import dlib
model = keras.models.load_model('/src/model/model1')
img_raw = cv2.imread('/home/ntp/DeepLearning/face-recognition/src/image/MyTam1.jpg')
face_detector = dlib.get_frontal_face_detector()
faces = face_detector(img_raw,1)
face = faces[0]
face_raw = img_raw[face.top():face.bottom(), face.left():face.right(),:]
cv2.imshow('face',face_raw)
cv2.waitKey()
img = cv2.resize(face_raw,(224,224))
arr = np.asarray(img)
print(arr.shape)
rs = model.predict_classes(arr.reshape(1,224,224,3))
print(rs)