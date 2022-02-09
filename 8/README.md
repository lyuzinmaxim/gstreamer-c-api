in directory "backup1" is code, that is generated due to deepstream-testsr, but not saving video (theit size is 0)

https://askubuntu.com/questions/1095521/cant-build-gst-rtsp-server

```gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test ! rtph264depay ! tee ! nvv4l2decoder ! autovideosink```

```cd ~/gst-rtsp-server/examples && ./test-launch "( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )"```

https://stackoverflow.com/questions/31455979/how-to-specify-libraries-paths-in-gcc


```
#sudo nvpmodel -m 0 && \
#echo -e "10 W mode is ON \tn" && \
#sleep 3 && \
#echo -e "It was 3 seconds driver initializatior" && \
#sudo ifconfig eth0 10.0.100.10/16  && \
#sudo ip route add 10.0.111.10 via 10.0.100.230 dev eth0 && \
#echo -e "IP adress and route were set" && \
#gst-launch-1.0 nvarguscamerasrc sensor-id=0 bufapi-version=1 ! "video/x-raw(memory:NVMM),width=1920,height=1080,format=(string)NV12,framerate=30/1" ! nvvideoconvert flip-method=2 ! nvv4l2h265enc  bitrate=6000000  maxperf-enable=1 control-rate=1 ! h265parse config-interval=-1 ! rtph265pay ! tee name = t \
#t. ! queue ! udpsink host=10.0.111.10 port=31990 sync=False \
#t. ! queue ! udpsink host=127.0.0.1 port=3445 sync=False

#exit 0 
```
