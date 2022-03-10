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
cd /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/smart_record && ./deepstream-testsr-app
```

Passing 0 is for ignoring, passing 1 is for make start/stop
