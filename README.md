# DCT-PRB: A Dynamic Conflict Tracker Model for Drone-Based Traffic Hotspot Detection

## Abstract
we combined DCT-PRB with SMILEtrack for real-time, accurate target tracking. We designed the Dynamic Conflict Tracker (DCT), which includes three constraints: pedestrian location, vehicle speed, and DCT overlap ratio, to determine potential conflicts and enhance detection reliability. We also implemented an interactive lane contour drawing interface to help users quickly delineate observation areas. After detection, the model can immediately generate potential conflict hotspot maps based on the results. This makes it the fastest and most versatile accident hotspot detection model, suitable for roads in any country or region.

<div align="center">
    <a href="./">
        <img src="heatmap_overlay.jpg" width="80%"/>
    </a>
</div>

## Inference (Detection ＆ tracking)
Inference weight ＆ Test video can be downloaded ([here](https://drive.google.com/drive/folders/18JZ7gxwDMHOE4I0XSTMWlkf0Md4e_oL6?usp=sharing)).

On video:
``` shell
 python DCT-PRB.py --weights weights/best.pt --source video/dowload/<video path (*.mp4)> --conf 0.5 --save-txt --img-size 1280 --trace --view-img --draw --classes 0 3 4 5 6 7 8 9
```

You will get the submmision file in 'runs/detect/exp*'

## Reference 
Detection code is based on PRB-FPN [here](https://github.com/pingyang1117/PRBNet_PyTorch)

Tracking code is based on SMILETrack(multi-object-tracking-on-mot20 2024-SOTA) [here](https://github.com/NYCU-AICVLab/AICITY_2023_Track5)

version 1.1.1 by LI,GUAN-YI 2024/07/30
