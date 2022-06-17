from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
import cv2
import math
import os
from glob import glob
from scipy import stats as s
from predict import analyze

base_model = VGG16(weights='imagenet', include_top=False)

#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

# loading the trained weights
model.load_weights("weight.hdf5")
# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# creating the tags from trainset
train = pd.read_csv('cleaned/train_new.csv')
y = train['class']
y = pd.get_dummies(y)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255,255,255)
  
# Line thickness of 2 px
thickness = 1


# creating two lists to store predicted and actual tags
predict = []
actual = []

# for loop to extract frames from each test video
def realtime_inference(videoFile):
    count = 0
    
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    print(frameRate)
    x=1
    # removing all other files from the temp folder
    files = glob('temp/*')
    for f in files:
        os.remove(f)
    result='INITIATING'
    infer_frames = []
    frameId = 0
    result_B = False
    csvs = glob(f'test/*.json')
    if len(csvs)!=0 and os.path.exists(csvs[0]):
        result_B = analyze(filename=csvs[0])
    while(cap.isOpened()):
        #frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
            # storing the frames of this particular video in temp folder
        # filename ='temp/' + videoFile.split('/')[1]+"_frame%d.jpg" % count;count+=1
        #cv2.imwrite(filename, frame)
        img = cv2.resize(frame,(224,224))
        img = img/255
        infer_frames.append(img)
        #if frameId > (math.floor(frameRate)*10):
        print(frameId,frameRate)
        if frameId > 10 and frameId % (math.floor(frameRate)*5)  == 0:
            # perform inference
            # get result
            result = inference_from_temp(infer_frames)
            if not result_B:
                pass
            elif result_B in result:
                result = result_B
            else:
                pass
            # clear temp
            for i in range(math.floor(frameRate)*5):
                infer_frames.pop(0)
        frame = cv2.putText(frame, result, org, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('VideoTrack',frame)
        cv2.waitKey(5)
        frameId +=1

    cap.release()

def inference_from_temp(infer_frames):
    # reading all the frames from temp folder
    images = glob("temp/*.jpg")
    prediction_images = infer_frames

    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)

    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
    # predicting tags for each array
    # prediction = model.predict_classes(prediction_images)

    predict_x=model.predict(prediction_images) 
    prediction=np.argmax(predict_x,axis=1)
    # appending the mode of predictions in predict list to assign the tag to the video

    return y.columns.values[s.mode(prediction)[0][0]]


realtime_inference('test/0.mp4') # 参数如果是视频，则是视频路径。如果是摄像头，则是摄像头id，比如0.
#realtime_inference(0)
