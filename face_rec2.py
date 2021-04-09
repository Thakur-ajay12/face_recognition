import cv2 as cv
import face_recognition
import numpy as np
from datetime import datetime
import os

path ='G:/datafiles/face_rec/known_ajay'
images=[]
names=[]
mylist= os.listdir(path)

for cl in mylist:
    cl_img=cv.imread(f'{path}/{cl}')
    images.append(cl_img)
    names.append(os.path.splitext(cl)[0])
#print(images)
print(names)

def get_encodings(images):
    encodelist=[]
    for img in images:
        image=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #print(image)
        encod=face_recognition.face_encodings(image)[0]
        encodelist.append(encod)
    return encodelist
known_encodelist= get_encodings(images)

def get_attendence(name):
    with open("attendence.csv","r+", encoding="utf-8") as f :
        datalist= f.readlines()
        namelist=[]
        for line in datalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            ctime = datetime.now()
            stime= ctime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{stime}')



cap = cv.VideoCapture(0)
while True:
    succes, img = cap.read()
    # cv.imshow('video',img)
    # imgs= cv.resize(img,(0,0),None,(0.25,0.25))
    imgs = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #    break

    loc = face_recognition.face_locations(imgs)
    enc = face_recognition.face_encodings(imgs, loc)
    # print(enc)

    for encodeface, face_loc in zip(enc, loc):
        matches = face_recognition.compare_faces(known_encodelist, encodeface)
        result = face_recognition.face_distance(known_encodelist, encodeface)
        print(matches)

        matchIndex = np.argmin(result)
        if matches[matchIndex]:
            name=names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1= face_loc
            cv.rectangle(img, (x1,y1),(x2,y2),(255,255,255),2)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_ITALIC,1,(255,255,255),2)
            get_attendence(name)
    cv.imshow('Webcam', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
       break
    #cv.
