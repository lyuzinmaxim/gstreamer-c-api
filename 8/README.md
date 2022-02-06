in directory "backup1" is code, that is generated due to deepstream-testsr, but not saving video (theit size is 0)

https://askubuntu.com/questions/1095521/cant-build-gst-rtsp-server

```gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test ! rtph264depay ! tee ! nvv4l2decoder ! autovideosink```

```~/gst-rtsp-server/examples$ ./test-launch "( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )"```

https://stackoverflow.com/questions/31455979/how-to-specify-libraries-paths-in-gcc
