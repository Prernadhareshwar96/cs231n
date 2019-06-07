import cv2
import os

token = "dribble"
count_file = 0
for filename in os.listdir(token):

    count_file += 1
    filepath ="dribble_jpg/1/" + str(count_file) + "_ "
    vidcap = cv2.VideoCapture(token + "/" + filename)
    success,image = vidcap.read()
    count = 0
    print(count_file , " file done" )
    while success:
          cv2.imwrite(filepath + str(count) + ".jpg", image)       
          success,image = vidcap.read()
          count += 1
