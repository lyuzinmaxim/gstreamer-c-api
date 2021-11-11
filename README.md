# gstreamer-c-api
some code examples for deepstream C API on Jetson Nano with DS 6.0

Dependencies:

* GStreamer 1.14.5
* DeepStreamSDK 6.0.0
* JetPack 4.6 (SD-card boot)
* gcc (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) 7.5.0


1. **h264_gstreamer.c**

```gst-launch-1.0 filesrc location = sample_720p.h264 ! h264parse ! nvv4l2decoder !  autovideosink sync=0```
 
to compile

```gcc h264_gstreamer.c -o h264_gstreamer `pkg-config --cflags --libs gstreamer-1.0` ```
