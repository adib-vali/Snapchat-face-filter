import cv2
import numpy as np
click_event =0
button_num =0
frame_gif_n =-1
dic ={}
#flip frames
flip =lambda frame : cv2.flip(frame, 1)

#add filters icon to buttons
def button(filters_array,background,click):
    global button_num
    icons_space =background
    width,hight,_ =background.shape
    n_icons =len(filters_array)
    scale =int((1-(n_icons/10))*width)
    if scale%2 == 1:scale=scale-1
    print(int((1-(n_icons/10))*width),width)
    button_num =n_icons
    xscale,radius =int(hight/(n_icons+1)),int(scale/2)
    for i in range(1,n_icons+1):
        button_icon =filters_array[i-1]
        button_icon =cv2.resize(button_icon,(scale,scale))
        icons_space[int(width/2)-radius:int(width/2)+radius,xscale*i-radius:xscale*i+radius] =cv2.add(icons_space[int(width/2)-radius:int(width/2)+radius,xscale*i-radius:xscale*i+radius],button_icon)
    if(click != 0):   
        cv2.line(icons_space,(xscale*click-radius,width-10),(xscale*click+radius,width-10),(200,200,200),2)
    return icons_space

#add black effect to buttons area
def gray_filter(img):
    width,hight,_ =img.shape
    filter_color =np.ones((width,hight,3),np.uint8)
    return cv2.addWeighted(img,0.3,filter_color,1,5)

#click event
def click_buttons(event,x,y,flags,param):
    global click_event
    global button_num
    if(y>460 and event==cv2.EVENT_LBUTTONDOWN):
        click_event =int(x/(340/(button_num+1)))+1
        #Find clicked button
        if click_event>button_num : click_event=click_event-1
        # This condition is to prevent the buttons from leaving the screen

#main func 
def _ui_main_func_(filters_address,model,cap):
    app_screen =np.ones((530,340,3),np.uint8)*10
    filters_array =read_filters(filters_address)
    while(cap.isOpened()):
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        cv2.setMouseCallback('app',click_buttons)
        _,frame =cap.read()
        print(frame)
        frame_copy =flip(frame)
        faces =face_detect(frame_copy,model)   
        img_filter =filters_array[click_event-1]
        frame_copy =print_filter(faces,frame_copy,img_filter)
        app_screen[0:480,0:340] =frame_copy[0:480,150:490]
        blur =cv2.blur(app_screen[390:460,0:340],(5,10))
        blur =gray_filter(blur)
        app_screen[460:530,0:340] =button(filters_array,blur,click_event)
        cv2.imshow('app',app_screen)
    cap.release()
    cv2.destroyAllWindows()

#find faces in frame
def face_detect(frame,model):
    face_cascade =cv2.CascadeClassifier(model)
    frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(frame,1.1,4)

#add snapchat filter on fraces
def print_filter(faces,frame,img_filter):
    h_filter,w_filter,_ =img_filter.shape
    if(click_event != 0):
        for(x,y,w,h) in faces:
            if w>10 and h>10 :
                scale =h/h_filter
                h_filter,w_filter =int(h_filter*scale),int(w_filter*scale)
                img_filter =cv2.resize(img_filter,(h_filter,w_filter))
                frame_copy =cv2.add(frame[y-30:y+w_filter-30,x:x+h_filter],img_filter) 
                frame[y-30:y+w_filter-30,x:x+h_filter] =frame_copy
    return frame

#add dark background 
def add_black_background(png):
    mask =cv2.inRange(png,(0,0,0),(254,254,254))
    return cv2.bitwise_and(png,png,mask=mask)

#read defined filters
def read_filters(filters_address):
    png_filters_list =[]
    global dic
    for i in filters_address:
        png =cv2.imread(i)
        png =cv2.resize(png,(400,400))
        png_filters_list.append(list(add_black_background(png)))
    return np.array(png_filters_list)

cap =cv2.VideoCapture(0,cv2.CAP_DSHOW)
filters_address_list =['candy.png','amongus.png','cat.png','dog.png','dog2.png']
#add filters path
_ui_main_func_(filters_address_list,'haarcascade_frontalface_default.xml',cap)
#add model path
