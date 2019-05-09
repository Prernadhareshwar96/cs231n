import numpy as np
import cv2
import os

token = "dribble"
count = 0
for filename in os.listdir(token):
    count += 1
    print(count)
    X = None
    vidcap = cv2.VideoCapture(token + "/" + filename)
    success, image = vidcap.read()
    x = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    X = np.reshape(x, -1)
    while success:
        success, image = vidcap.read()
        if success:
            x = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            X = np.vstack((X, np.reshape(x, -1)))
    print("Done")
    np.savetxt("data/" + token + "/video_"+str(count), X)
    
     
    