C-code  makes the same as:

```gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080, framerate=(fraction)30/1' ! queue ! autovideosink sync=false```

**IMPORTANT: specify video resolution**

to compile

```gcc test_video.c -o test_video `pkg-config --cflags --libs gstreamer-1.0` ```

to run

```./test_video.c```
