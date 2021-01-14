import cv2
import numpy as np
import os
import glob

from src.lib.detections.dlib_detections import detection

font = cv2.FONT_HERSHEY_SIMPLEX

def extract_img(video_path, out_dir):
    videos = glob.glob(os.path.join(video_path, "*.*"))
    folder_name = os.path.basename(video_path).split('.')[0]
    result_path = os.path.join(out_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)
    count = 0
    for video_path in videos:
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        while success:
            if count % 4 == 0:
                face = detection(frame)
                if face is None:
                    success, frame = video.read()
                    continue
                try:
                  img = cv2.resize(face, (224,224))
                  cv2.imwrite('{}/{}.jpg'.format(result_path, count // 4), img)
                except Exception:
                  print('some thing wrong')
            count+=1
            success, frame = video.read()

    # video_name = os.path.basename(video_path).split('.')[0]
    # video = cv2.VideoCapture(video_path)
    # success, frame = video.read()
    # count = 0
    # while success:
    #     count += 1
    #     cv2.waitKey()
    #     if count % 1 == 0:
    #         face = detection(frame)
    #         if face is None:
    #             continue
    #         cv2.imwrite('{}/{}.jpg'.format(result_path, count // 15), face)
    #
    #     success, frame = video.read()


if __name__ == "__main__":
    # videos_dir = "/home/ntp/DeepLearning/face-recognition/src/video/"
    image_path = "/home/ntp/DeepLearning/face-recognition/src/image/"
    # list_videos = glob.glob(os.path.join(videos_dir, "*"))
    # for item in list_videos:
    #     extract_img(item, image_path)

    video_dir ="/home/ntp/DeepLearning/face-recognition/src/video/MinhNgo"
    extract_img(video_dir, image_path)

