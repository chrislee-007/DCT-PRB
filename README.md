# DCT-PRB: A Dynamic Conflict Tracker Model for Drone-Based Traffic Hotspot Detection

<div align="center">
    <a href="./">
        <img src="heatmap_overlay.jpg" width="80%"/>
    </a>
</div>

## Abstract
We combined DCT-PRB with SMILEtrack for real-time, accurate target tracking. We designed the Dynamic Conflict Tracker (DCT), which includes three constraints: pedestrian location, vehicle speed, and DCT overlap ratio, to determine potential conflicts and enhance detection reliability. We also implemented an interactive lane contour drawing interface to help users quickly delineate observation areas. After detection, the model can immediately generate potential conflict hotspot maps based on the results. This makes it the fastest and most versatile accident hotspot detection model, suitable for roads in any country or region.

## How to use (Detection ＆ tracking)
Dataset & Inference weight ＆ Test video can be downloaded ([here](https://drive.google.com/drive/folders/18JZ7gxwDMHOE4I0XSTMWlkf0Md4e_oL6?usp=sharing)).

1.Run interactive_polygon.py to set the ROI (viewing area).
``` shell
 python interactive_polygon.py 
```

2.Run DCT-PRB.py for potential conflict detection. 
``` shell
 python DCT-PRB.py --weights weights/best.pt --source video/dowload/<video path (*.mp4)> --conf 0.5 --save-txt --img-size 1280 --trace --view-img --draw --classes 0 3 4 5 6 7 8 9
```

You will get the submmision file in 'runs/detect/exp*'

## Troubleshooting
Q-> Path error problem.

A-> Set your folder_path and video_path in interactive_polygon.py＆ DCT-PRB.py.
``` shell
＃sample 1
if __name__ == "__main__":
    folder_path = '/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/polygon_counter'
    clear_folder(folder_path)  
    video_path = '/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/video/dowload/053101.mp4'  
    draw_polygon(video_path)
```

Q-> The video is too large for your screen.

A-> you can change the scale variable to adjust the size of the input video.
``` shell
 scale = 0.7
```

## Reference 
Detection code is based on PRB-FPN [here](https://github.com/pingyang1117/PRBNet_PyTorch)

Tracking code is based on SMILETrack(Multi-Object-Tracking 2024-SOTA) [here](https://github.com/NYCU-AICVLab/AICITY_2023_Track5)

version 1.1.1 by chris-lee 2024/07/30
