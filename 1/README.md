C-code  makes the same as:

```gst-launch-1.0 filesrc location = sample_720p.h264 ! h264parse ! nvv4l2decoder !  autovideosink sync=0```

**IMPORTANT: specify path to .h264 video**
 
to compile

```gcc h264_gstreamer.c -o h264_gstreamer `pkg-config --cflags --libs gstreamer-1.0` ```

to run

```./h264_gstreamer.c```
