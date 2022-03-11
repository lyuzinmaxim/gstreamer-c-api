build
```
sudo make
```

Start stream (in another terminal)
```
cd ~/gst-rtsp-server/examples && ./test-launch "( videotestsrc ! nvvideoconvert ! nvv4l2h264enc ! rtph264pay name=pay0 pt=96 )"

```
Start SmartRecord
```
sudo cd /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/smart_record && ./deepstream-testsr-app
```

Passing 0 is for ignoring, passing 1 is for make start/stop



Local transmit
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 bufapi-version=1 ! "video/x-raw(memory:NVMM),width=1920,height=1080,format=(string)NV12,framerate=30/1" ! nvvideoconvert flip-method=2 ! nvv4l2h264enc  bitrate=6000000  maxperf-enable=1 control-rate=1 ! h264parse config-interval=-1 ! rtph264pay ! queue ! udpsink host=127.0.0.1 port=5000 sync=False
```

Local receive
```
gst-launch-1.0 -e udpsrc port=5000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! nvv4l2decoder ! autovideosink sync=False
```
