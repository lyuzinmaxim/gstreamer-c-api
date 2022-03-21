subdirectory "socket" is the main code now


Working version is in "root" without video output, works great

Cache starts working only since "I" (Intracoded) frame was coming


build
```
sudo make
```

Start stream (in another terminal) with MIPI camera connected
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 bufapi-version=1 ! "video/x-raw(memory:NVMM),width=1920,height=1080,format=(string)NV12,framerate=30/1" ! nvvideoconvert flip-method=2 ! nvv4l2h264enc  bitrate=6000000  maxperf-enable=1 control-rate=1 ! h264parse config-interval=-1 ! rtph264pay ! queue ! udpsink host=127.0.0.1 port=5000 sync=False
```
