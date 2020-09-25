# tf-pose-recognition-blur-live

Uses OpenCV, CUDA 10.0, tensorflow, and bodypix to blur a background of a live camera stream. Use OBS Studio virtual camera on Windows to loopback video input or v4l2-loopback on linux.

```
npm install
node app.js &
python3 fake2.py
```

Based off https://github.com/jwd83/Open-Source-Virtual-Background
