# 3. **udp_encoded_stream.c**

Not only Jetson board, but host PC too needed to run this code. Make sure there is Ethernet connections between them, IP adress of host should be 192.168.0.1, port 5000 should be open. Jetson should be in the same subnet, with IP 192.168.0.0. Host commands are from precompiled binaries, not using C API.

! Before starting, receiving code (on host) should be ran

```
gst-launch-1.0 udpsrc port=5000 caps = "application/x-rtp, media=(string)video,clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" \
! rtph264depay ! decodebin ! videoconvert ! fpsdisplaysink sync=False
```


C-code  makes the same as:

``` ifconfig eth0 192.168.0.0 && \
gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080, framerate=(fraction)30/1' ! \
! queue ! nvvideoconvert !  nvv4l2h264enc  bitrate=1000000 ! rtph264pay ! udpsink host=192.168.0.1 port=5000
```

to compile

```gcc udp_encoded_stream.c -o udp_encoded_stream `pkg-config --cflags --libs gstreamer-1.0` ```

to run (notice that there Jetson's IP-adress will be changed)

```sudo ifconfig eth0 192.168.0.0 && ./udp_encoded_stream```
