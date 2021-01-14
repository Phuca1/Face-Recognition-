import time
import dlib
import cv2

def detection(image):
  hog_face_detector = dlib.get_frontal_face_detector()
  faces = hog_face_detector(image, 1)
  if not faces:
    return None
  face = faces[0]
  return image[face.top():face.bottom(), face.left():face.right(), :]

if __name__ == "__main__":
  img = cv2.imread("/home/ntp/Pictures/Phuc.jpg")
  img = cv2.resize(img, (1400, 1000))
  face = detection(img)
  cv2.imshow("face", face)
  cv2.waitKey()