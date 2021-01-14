import cv2
import dlib
import numpy as np

model_path = '/home/ntp/DeepLearning/face-recognition/src/model/model1_VGG16_2_class_v2'

from tensorflow import keras

model = keras.models.load_model(model_path)

face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
count = 0

while (True):
    bool_arr = np.array([False] * 3, dtype=bool)
    ret, frame = cap.read()
    # textField = ''
    if (count % 20 < 5):
        try:
            faces = face_detector(frame, 1)
            for face in faces:
                face_raw = frame[face.top():face.bottom(), face.left():face.right(), :]
                img = cv2.resize(face_raw, (224, 224))
                arr = np.asarray(img)
                # out = model.predict(arr.reshape(1, 224, 224, 3))
                # rs = np.argmax(out)
                # if(out[rs] < 0.7):
                #     continue
                rs = model.predict_classes(arr.reshape(1, 224, 224, 3))
                # bool_arr[rs] = True
                for item in rs:
                    bool_arr[item] = True
                cv2.line(frame, (face.top() - 5, face.left()), (face.top() - 10, face.right() + 5), (0, 255, 0), 5)
                cv2.line(frame, (face.top() - 5, face.right() + 5), (face.bottom(), face.right() + 5), (0, 255, 0), 5)
                cv2.line(frame, (face.bottom(), face.right() + 5), (face.bottom(), face.left()), (0, 255, 0), 5)
                cv2.line(frame, (face.bottom(), face.left()), (face.top() - 5, face.left()), (0, 255, 0), 5)
        except Exception:
            print("something went wrong")

        for i in range(len(bool_arr)):
            if (bool_arr[i]):
                print(i)
                if (i == 0):
                    cv2.putText(frame, 'Hello Minh', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                if (i == 1):
                    cv2.putText(frame, 'Hello Phuc', (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('face', frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
