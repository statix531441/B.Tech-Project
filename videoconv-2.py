import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

shutil.rmtree('dmaps')
os.mkdir('dmaps')

device = 'cuda'
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)


vid = cv2.VideoCapture('videos/badminton.mp4')
ret, img = vid.read()
frame = 0
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


input_batch = transform(img).to(device)

with torch.no_grad():
    #Depth Map Generation
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    #object Detection
    df = model(img).pandas().xyxy[0]

df['xmin'] = df['xmin'].astype(int)
df['ymin'] = df['ymin'].astype(int)
df['xmax'] = df['xmax'].astype(int)
df['ymax'] = df['ymax'].astype(int)
df = df.to_numpy()

dmap = prediction.cpu().numpy()
dmap = dmap - dmap.min()
dmap = dmap/dmap.max()
dmap = dmap*255

def f(a):
    pass

cv2.namedWindow('Options')
cv2.resizeWindow('Options',640,480)
cv2.createTrackbar('Max','Options',dmap.max(),dmap.max(),f)
cv2.createTrackbar('Min','Options',0,dmap.max(),f)
cv2.createTrackbar('ObjFilter','Options',0,1,f)
cv2.createTrackbar('ConfThres','Options',0,100,f)

while True:

    OBJ_FILT = cv2.getTrackbarPos('ObjFilter', 'Options')
    CONF_THRES = cv2.getTrackbarPos('ConfThres', 'Options')
    minD = 255-cv2.getTrackbarPos('Max','Options')
    maxD = 255-cv2.getTrackbarPos('Min','Options')

    range_filtered = np.copy(dmap)

    #Minimum Range
    range_filtered = range_filtered-minD
    range_filtered = range_filtered*(dmap.max()/range_filtered.max())
    range_filtered[range_filtered<0] = 0

    #Maximum Range
    range_filtered = range_filtered*(dmap.max()/maxD)
    range_filtered[range_filtered>dmap.max()] = 0
    range_filtered = range_filtered/range_filtered.max()

    #Object filtering
    skim = np.zeros(range_filtered.shape)
    for objs in df:
        xmin,ymin,xmax,ymax,conf,objclass = objs[:6]
        if objclass==0 and conf*100 > CONF_THRES:
            skim[ymin:ymax, xmin:xmax] = range_filtered[ymin:ymax, xmin:xmax]
    
    if OBJ_FILT:
        output = skim
    else:
        output = range_filtered

    cv2.imshow('test', output)

    key = cv2.waitKey(32) & 0xFF
    if key == 32:
        break

cv2.destroyAllWindows()

plt.imsave(f'dmaps/frame_{frame:0>4}.png', output, cmap='gray')
h,w = output.shape[1],output.shape[0]

#exit()
total = 180

if total > vid.get(cv2.CAP_PROP_FRAME_COUNT):
    total = vid.get(cv2.CAP_PROP_FRAME_COUNT)

print("Running ...")

for i in tqdm(range(total)):

    ret, img = vid.read()
    frame += 1

    if not ret or frame>total-1:
        break

    #Skip frames: 1 => use every frame
    if frame % int(1): continue

    #Object Detection
    df = model(img).pandas().xyxy[0]
    df['xmin'] = df['xmin'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    df = df.to_numpy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Depth Map Generation
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

    range_filtered = np.copy(dmap)

    #Minimum Range
    range_filtered = range_filtered-minD
    range_filtered = range_filtered*(dmap.max()/range_filtered.max())
    range_filtered[range_filtered<0] = 0

    #Maximum Range
    range_filtered = range_filtered*(dmap.max()/maxD)
    range_filtered[range_filtered>dmap.max()] = 0
    range_filtered = range_filtered/range_filtered.max()

    #Object filtering
    if OBJ_FILT:
        skim = np.zeros(range_filtered.shape)
        for objs in df:
            xmin,ymin,xmax,ymax,conf,objclass = objs[:6]
            if objclass==0 and conf*100 > CONF_THRES:
                skim[ymin:ymax, xmin:xmax] = range_filtered[ymin:ymax, xmin:xmax]
        output = skim

    else:
        output = range_filtered

    plt.imsave(f'dmaps/frame_{frame:0>4}.png', output, cmap='gray')
    
vid.release()



#WRITE TO VIDEO --------------------------------------------------------------------
out = cv2.VideoWriter(f'live/{minD}-{maxD}-{OBJ_FILT}-{CONF_THRES}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (h,w), 0)

images = os.listdir('dmaps')
print("Compiling Video")
for imgname in tqdm(images):
    kek = cv2.imread(f'dmaps/{imgname}',0)

    out.write(kek)

out.release()
print("Finished!")







