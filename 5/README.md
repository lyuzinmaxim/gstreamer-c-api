This code is based on *deepstream_test1*, published by NVIDIA

Not only Jetson board, but host PC too needed to run this code. Make sure there is Ethernet connections between them, IP adress of host should be 192.168.0.1, port 5000 should be open. Jetson should be in the same subnet, with IP 192.168.0.0. Host commands are from precompiled binaries, not using C API.

!Before starting, receiving code (on host) should be ran


```
gst-launch-1.0 udpsrc port=5000 caps = "application/x-rtp, media=(string)video,clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" \
! rtph264depay ! decodebin ! videoconvert ! fpsdisplaysink sync=False
```
C-code makes the same as:

![GStreamer send&infere from file](https://github.com/lyuzinmaxim/gstreamer-c-api/blob/6df9162ed91c13e4fc6318a622599cbdfa34bcba/docs/GStreamer%20send&infere%20from%20file.drawio.png)

```
ifconfig eth0 192.168.0.0 && \
gst-launch-1.0 filesrc location = /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 ! h264parse  ! nvv4l2decoder !  tee name=t \
t. ! queue ! m.sink_0 nvstreammux name = m \
batch-size=1 width=1920 height=1080 ! nvinfer config-file-path=dstest1_pgie_config.txt ! nvvideoconvert ! nvdsosd ! nvegltransform ! nveglglessink \
t. ! nvvideoconvert ! nvv4l2h264enc  bitrate=4000000 ! rtph264pay ! udpsink host=192.168.0.1 port=5000 sync=True
```

Because it's based on NVidia DeepStream example apps, compile it using Makefile
```
sudo make
```
to run (notice that there Jetson's IP-adress will be changed)

```
sudo ifconfig eth0 192.168.0.0 && ./deepstream-test1-app
```
