import cv2
import face_recognition
import numpy as np
import datetime as datetime


vijay_test=face_recognition.load_image_file('vijay_test.jpg')
red_test=cv2.cvtColor(vijay_test, cv2.COLOR_BGR2RGB)
vijay_train=face_recognition.load_image_file('vijay_train.jpg')
red_train=cv2.cvtColor(vijay_train, cv2.COLOR_BGR2RGB)
new_train=face_recognition.load_image_file('test2.jpg')
new_red=cv2.cvtColor(new_train, cv2.COLOR_BGR2RGB)

trainloc=face_recognition.face_locations(red_train)
trainenc=face_recognition.face_encodings(red_train,trainloc)
for location1 in trainloc:
    top, right, bottom, left = location1
rect1= cv2.rectangle(red_train,(left, top), (right, bottom),(0,255,0),2)


newloc=face_recognition.face_locations(new_red)
newenc=face_recognition.face_encodings(new_red,trainloc)
for location2 in newloc:
    top, right, bottom, left = location2
    print(top)
rect2= cv2.rectangle(new_red,(left, top), (right, bottom),(0,255,0),2)




testloc=face_recognition.face_locations(red_test)
testenc=face_recognition.face_encodings(red_test,testloc)
for location3 in testloc:
    top, right, bottom, left = location3
rect3= cv2.rectangle(red_test,(left, top), (right, bottom),(0,255,0),2)


result=face_recognition.compare_faces([trainenc],testenc)



