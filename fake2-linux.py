import os
import cv2
import numpy as np
import requests
import pyfakewebcam
import logging
import threading

gmask = ""
i=7
gframe = ""

def thread_function(name):
    bodypix_url='http://172.17.0.2:9000'
    print("Thread %s: starting", name)
    global gmask, i, gframe
    while True:
        if i==1:
            # print("updateMask")
            frame = gframe
            _, data = cv2.imencode(".jpg", frame)
            r = requests.post(
                url=bodypix_url,
                data=data.tobytes(),
                headers={'Content-Type': 'application/octet-stream'})
            mask = np.frombuffer(r.content, dtype=np.uint8)
            mask = mask.reshape((frame.shape[0], frame.shape[1]))
            gmask = mask
            i=0
        else:
            i=i+1

    time.sleep(2)
    print("Thread %s: finishing", name)



def get_mask(frame, bodypix_url='http://172.17.0.2:9000'):
    global gmask, i, gframe
    gframe = frame.copy()
    if i==7:
        _, data = cv2.imencode(".jpg", frame)
        r = requests.post(
            url=bodypix_url,
            data=data.tobytes(),
            headers={'Content-Type': 'application/octet-stream'})
        mask = np.frombuffer(r.content, dtype=np.uint8)
        mask = mask.reshape((frame.shape[0], frame.shape[1]))
        gmask = mask
        i=0
        thread = threading.Thread(target=thread_function, args=(1,))
        thread.start()
        return mask
    else:
        return gmask

def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10,10), np.uint8) , iterations=1)
    mask = cv2.blur(mask.astype(float), (30,30))
    return mask

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def get_frame(cap, background_scaled):
    _, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    #print(frame)
    # cv2.imshow("pic2",frame)
    # fetch the mask with retries (the app needs to warmup and we're lazy)
    # e v e n t u a l l y c o n s i s t e n t
    mask = None
    while mask is None:
    #    try:
            mask = get_mask(frame)
    #    except:
    #        print("mask request failed, retrying")
    # post-process mask and frame
    mask = post_process_mask(mask)
    # frame = hologram_effect(frame)
    # composite the foreground and background
    inv_mask = 1-mask
    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c]*mask + background_scaled[:,:,c]*inv_mask
    return frame

# setup access to the *real* webcam
cap = cv2.VideoCapture(0)
#height, width = 720, 1280
height,width=240,320
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#cap.set(cv2.CAP_PROP_FPS, 60)

# setup the fake camera
fake = pyfakewebcam.FakeWebcam('/dev/video2', width, height)

# load the virtual background
#background = cv2.imread("background.jpg")
_, background = cap.read()
#print(background)
background_scaled = cv2.resize(background, (width, height))
# frames forever
while True:
    cv2.waitKey(10)
    frame = get_frame(cap, background_scaled)
    # try:
    bgframe = cv2.copyMakeBorder(gframe,0,0,0,0,cv2.BORDER_REPLICATE)
    background_scaled = cv2.blur(bgframe,(40,40), cv2.BORDER_DEFAULT).copy()
    # except:
        # pass
    # fake webcam expects RGB
    cv2.imshow("greenscreen",frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)
