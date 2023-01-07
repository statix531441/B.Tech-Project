OBJ_FILT = False

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
shutil.rmtree('dmaps')
os.mkdir('dmaps')

device = 'cuda'
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


vid = cv2.VideoCapture('videos/badminton.mp4')
ret, img = vid.read()
frame = 0
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


input_batch = transform(img).to(device)

with torch.no_grad():
    #depth estimation
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    #object detection
    df = model(img).pandas().xyxy[0]

df['xmin'] = df['xmin'].astype(int)
df['ymin'] = df['ymin'].astype(int)
df['xmax'] = df['xmax'].astype(int)
df['ymax'] = df['ymax'].astype(int)
df = df.to_numpy()

dmap = prediction.cpu().numpy()
# dmap = dmap.max()+dmap.min()-dmap
dmap = dmap - dmap.min()
dmap = dmap/dmap.max()
dmap = dmap*255

#object filtering
skim = np.zeros(dmap.shape)
for i in df:
    xmin,ymin,xmax,ymax = i[:4]
    skim[ymin:ymax, xmin:xmax] = dmap[ymin:ymax, xmin:xmax]

# plt.imshow(dmap, cmap='gray')
# plt.show()

def f(a):
    pass
cv2.namedWindow('sliders')
cv2.resizeWindow('sliders',640,480)
cv2.createTrackbar('min','sliders',0,dmap.max(),f)
cv2.createTrackbar('max','sliders',dmap.max(),dmap.max(),f)

while True:
    if False: #OBJ_FILT:
        i = np.copy(skim)
    else:
        i = np.copy(dmap)
    minD = cv2.getTrackbarPos('min','sliders')
    maxD = cv2.getTrackbarPos('max','sliders')
    # i[dmap<minD] = 0
    # i[dmap>maxD] = 0

    #Minimum
    i = i-minD
    i = i*(dmap.max()/i.max())
    i[i<0] = 0

    #Maximum
    i = i*(dmap.max()/maxD)
    i[i>dmap.max()] = 0


    #i = i-i[i>i.min()].min()+10
    i = i/i.max()

    cv2.imshow('test', i)

    key = cv2.waitKey(32) & 0xFF
    if key == 32:
        break

cv2.destroyAllWindows()

plt.imsave(f'dmaps/frame_{frame:0>4}.png', i, cmap='gray')

exit()

while True:

    ret, img = vid.read()
    frame += 1

    if not ret:
        print("done ig")
        break

    if frame % int(1): continue
    if frame > 50: break

    #object detection
    df = model(img).pandas().xyxy[0]
    df['xmin'] = df['xmin'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    df = df.to_numpy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    dmap = prediction.cpu().numpy()
    # dmap = dmap.max()+dmap.min()-dmap
    dmap = dmap - dmap.min()
    dmap = dmap/dmap.max()
    dmap = dmap*255

    #Range filtering
    i = np.copy(dmap)
    i[dmap<minD] = 0
    i[dmap>maxD] = 0
    i = i/i.max()

    #object filtering
    skim = np.zeros(i.shape)
    for k in df:
        xmin,ymin,xmax,ymax = k[:4]
        skim[ymin:ymax, xmin:xmax] = i[ymin:ymax, xmin:xmax]

    if OBJ_FILT:
        plt.imsave(f'dmaps/frame_{frame:0>4}.png', skim, cmap='gray')
    else:
        plt.imsave(f'dmaps/frame_{frame:0>4}.png', i, cmap='gray')

vid.release()



#WRITE TO VIDEO --------------------------------------------------------------------
out = cv2.VideoWriter('/outputs/project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280,720), 0)

for imgname in os.listdir('dmaps'):
    kek = cv2.imread(f'dmaps/{imgname}',0)

    out.write(kek)

out.release()








