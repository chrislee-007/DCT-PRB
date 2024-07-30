# This project code base on AICITY 2023 Challenge ([Track5](https://www.aicitychallenge.org/)) -- Team : NYCU-Road Beast

## System workflow

<div align="center">
    <a href="./">
        <img src="./figure/system workflow.png" width="80%"/>
    </a>
</div>

## Inference (tracking)
On video:
``` shell
 python DCT-PRB.py --weights weights/best.pt --source video/dowload/<video path (*.mp4)> --conf 0.5 --save-txt --img-size 1280 --trace --view-img --draw --classes 0 3 4 5 6 7 8 9
```

You will get the submmision file in 'runs/detect/exp*'

## Reference 
This code is based on [PRBNet_Pytorch](https://github.com/pingyang1117/PRBNet_PyTorch)
