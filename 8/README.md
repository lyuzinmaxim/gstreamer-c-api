in directory "backup1" is code, that is generated due to deepstream-testsr, but not saving video (theit size is 0)

https://askubuntu.com/questions/1095521/cant-build-gst-rtsp-server

```gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test ! rtph264depay ! tee ! nvv4l2decoder ! autovideosink```
