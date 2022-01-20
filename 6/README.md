This example is based on *deepstream_test4* sample applecation published by NVIDIA

Here I use Raspberry Camera HQ with sensor IMX477 via CSI connection. 

Before start working, run as superuser python script in ``/opt/nvidia/jetson_io/``` directory to configure drivers.

```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 bufapi-version=1 ! "video/x-raw(memory:NVMM),width=1920,height=1080,format=(string)NV12,framerate=60/1" ! tee name=t \
t. ! queue ! m.sink_0 nvstreammux name = m \
batch-size=1 width=1920 height=1080  batched-push-timeout=33 num-surfaces-per-frame=1 ! nvinfer config-file-path=dstest1_pgie_config.txt ! nvvideoconvert ! nvdsosd ! nvegltransform ! nveglglessink sync=False \
t. ! nvvideoconvert ! nvv4l2h264enc  bitrate=4000000 ! rtph264pay ! udpsink host=192.168.0.1 port=5000 sync=True
```

![Deepstream camera src and infere](https://github.com/lyuzinmaxim/gstreamer-c-api/blob/bb0d47e5b89d2969c3f9a99a92d2bf99464b6faf/docs/gstreamer_camera_src_infere.png)
