#Importing liabraries

import numpy as np
from keras.models import load_model
import cv2
import tkinter as tk
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tkinter.constants import BOTTOM, LEFT, RIGHT, TOP,X
from tkinter import Button, Image, Label,messagebox
from PIL import Image,ImageTk
from numpy.core.fromnumeric import size
from datetime import datetime
import pandas as pd
import time 

#Stop function
def stop():
    cam.release()
    root.destroy()


#Start function
def start():


    startButton.place_forget()
    Accuracy.place(x=1310,y=100)
    
    # Setting filename to local system time
    today = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    curdate = str(today)
    path = curdate+".xlsx"

    array = list()

    #Setting up camera properties    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    # cam.set(cv2.CAP_PROP_FPS,60)
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier =load_model(r'model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    while True:
        ret,frame = cam.read()
        frame =cv2.flip(frame,1)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
        faces = face_classifier.detectMultiScale(gray)

        #Probability view
        
        
        canvas = np.ones((250,300, 3), dtype="uint8")*255

        i=0
        for (x,y,w,h) in faces:
            #Putting rectangle around face

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                # Setting prediction accuracy in canvas

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                array.append(label)
                label_position = (x,y-10)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
                emotion_probability = np.max(prediction)

                for (i, (emotion, prob)) in enumerate(zip(emotion_labels, prediction)):
                #Probability viewer
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (0, (i * 35) + 5),(w, (i * 35) + 35), (0,255, 0), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
            
            
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            
            #Writing frames
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame1 = ImageTk.PhotoImage(Image.fromarray(img))
            CamFrame['image']= frame1
            frame2=ImageTk.PhotoImage(Image.fromarray(canvas))
            Accuracy['image']=frame2
            root.update()
          
        #Saving data in excel
        df = pd.DataFrame(array)
        df.to_excel(excel_writer = path)
    


cam = cv2.VideoCapture(0)
if cam is None or not cam.isOpened():
        messagebox.showerror('error', 'Camera not found!')
        time.sleep(3)
        stop()

#GUI
root = tk.Tk()
root.iconbitmap("icon.ico")
root.title("Emotica.ai")
root.geometry("1600x900")
root.config(bg='white')
Label(root,text="Emotica.AI",font=("Sans Sherif",15,"bold"),bg="white",fg ="black").place(relwidth=1,relheight=0.05)
#frame = LabelFrame(root,bg="white")
#frame.pack()


CamFrame= Label(root)
Accuracy = Label(root)
#Buttons
startButton = Button(root,height=1, width=20, text ="Start",font=("calibri",20),bg='white',fg='blue',relief='ridge',command = start)
stopButton  = Button(root,height=1, width=20, text ="Stop",font=("calibri",20),bg='white',fg='red',relief='ridge', command = stop)

#Placing buttons
CamFrame.place(height=720,width=1280, x=20, y=100)
startButton.place(x=1320,y=600,height=60,width=230)
stopButton.place(x=1320,y=700,height=60,width=230)

root.mainloop()

cam.release()
cv2.destroyAllWindows()