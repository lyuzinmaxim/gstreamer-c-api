It doesn't still work...

build
```
sudo make
```

Start stream (in another terminal)
```
cd ~/gst-rtsp-server/examples && ./test-launch "( videotestsrc ! nvvideoconvert ! nvv4l2h264enc ! rtph264pay name=pay0 pt=96 )"

```
