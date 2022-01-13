#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

//ifconfig eth0 192.168.0.0 && \
//gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080, framerate=(fraction)30/1' ! tee name=t \
//t. ! queue ! nvvideoconvert !  nvv4l2h264enc  bitrate=1000000 ! rtph264pay ! udpsink host=192.168.0.1 port=5000 \
//t. ! queue !  autovideosink

int
main (int argc, char *argv[])
{
  GstElement *pipeline, *source, *filter, *queue_local, *queue_enet, *tee, *converter, *encoder, *payer, *videosink, *enetsink;
  GstBus *bus;
  GstMessage *msg;
  GstCaps *filtercaps;
  GstStateChangeReturn ret;
  GstPad *tee_local_pad, *tee_enet_pad;
  GstPad *queue_local_pad, *queue_enet_pad;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Create the elements */
  source = gst_element_factory_make ("videotestsrc", "source");
  filter = gst_element_factory_make ("capsfilter","filter");
  tee = gst_element_factory_make("tee","tee");
  queue_local = gst_element_factory_make ("queue","queue_local");
  queue_enet = gst_element_factory_make ("queue","queue_enet");
  
  converter = gst_element_factory_make ("nvvideoconvert","converter");
 
  encoder = gst_element_factory_make ("nvv4l2h264enc","encoder");
  payer = gst_element_factory_make ("rtph264pay","payer");

  enetsink = gst_element_factory_make("udpsink","enetsink");
  videosink = gst_element_factory_make ("autovideosink", "videosink");

  /* Create the empty pipeline */
  pipeline = gst_pipeline_new ("maxim-pipeline");

  if (!pipeline || !source || !filter || !queue_local || !queue_enet || !tee || !converter || !encoder || !payer || !enetsink|| !videosink ) {
    g_printerr ("Not all elements could be created.\n");
    return -1;
  }

  /* Build the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), source, filter, tee, queue_local, queue_enet, converter, encoder, payer, enetsink, videosink, NULL);
  
  if (!gst_element_link_many(source,filter,tee,NULL)){
		  g_printerr ("Source->filter->tee(sink) problem\n");
		  gst_object_unref (pipeline);
		  return -1;
  }
  
  if (!gst_element_link_many(queue_enet,converter,encoder,payer,enetsink,NULL)){
                  g_printerr ("Queue->convert->encode->pay->udpsink problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }

  if (!gst_element_link_many(queue_local,videosink,NULL)){
                  g_printerr ("Queue->videosink problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }


  /* Modify the source's properties */
  g_object_set (source, "pattern", 18, NULL);
  g_object_set (source, "flip", 1, NULL);
  g_object_set (videosink, "sync", "FALSE", NULL);
  
  filtercaps = gst_caps_new_simple ("video/x-raw",
	  "format",G_TYPE_STRING,"I420",
          "width", G_TYPE_INT, 1920,
          "height", G_TYPE_INT, 1080,
          "framerate",GST_TYPE_FRACTION,30,1,
 	  NULL);
  
  g_object_set (filter, "caps", filtercaps, NULL);
  gst_caps_unref (filtercaps);
 
  g_object_set (encoder, "bitrate", 2000000, NULL);
  g_object_set (encoder, "maxperf-enable", 1, NULL); //not sure if causes small latency 
  g_object_set (encoder, "preset-level", 4, NULL); //not sure too
  g_object_set (encoder, "profile", 2, NULL); //really makes latency small
  g_object_set (encoder, "ratecontrol-enable", 1, NULL); //not sure

  //  g_object_set (encoder, "MeasureEncoderLatency", 1, NULL);
  
  /* Info from gst-inspect-1.0 about nvv4l2h264enc */
  /*
    preset-level        : HW preset level for encoder
                        flags: readable, writable, changeable only in NULL or READY state
                        Enum "GstV4L2VideoEncHwPreset" Default: 1, "UltraFastPreset"
                           (0): DisablePreset    - Disable HW-Preset
                           (1): UltraFastPreset  - UltraFastPreset for high perf
                           (2): FastPreset       - FastPreset
                           (3): MediumPreset     - MediumPreset
                           (4): SlowPreset       - SlowPreset
  qp-range            : Qunatization range for P, I and B frame,
                         Use string with values of Qunatization Range 
                         in MinQpP-MaxQpP:MinQpI-MaxQpI:MinQpB-MaxQpB order, to set the property.
                        flags: readable, writable
                        String. Default: null
  MeasureEncoderLatency: Enable Measure Encoder latency Per Frame
                        flags: readable, writable, changeable only in NULL or READY state
                        Boolean. Default: false
  ratecontrol-enable  : Enable or Disable rate control mode
                        flags: readable, writable, changeable only in NULL or READY state
                        Boolean. Default: true
  maxperf-enable      : Enable or Disable Max Performance mode
                        flags: readable, writable, changeable only in NULL or READY state
                        Boolean. Default: false
  profile             : Set profile for v4l2 encode
                        flags: readable, writable, changeable only in NULL or READY state
                        Enum "GstV4l2VideoEncProfileType" Default: 0, "Baseline"
                           (0): Baseline         - GST_V4L2_H264_VIDENC_BASELINE_PROFILE
                           (2): Main             - GST_V4L2_H264_VIDENC_MAIN_PROFILE
                           (4): High             - GST_V4L2_H264_VIDENC_HIGH_PROFILE
                           (7): High444          - GST_V4L2_H264_VIDENC_HIGH_444_PREDICT
    */

  g_object_set (enetsink, "host", "192.168.0.1", NULL);
  g_object_set (enetsink, "port", 5000, NULL);
  g_object_set (enetsink, "sync", 0, NULL);

  g_object_set (videosink, "sync", 0, NULL);


  /* Requesting src (output) pads of tee, bc they are "Presence - request"*/

  tee_local_pad = gst_element_get_request_pad (tee,"src_%u");
  if (!tee_local_pad) {
    g_printerr ("Tee local pad request failed. Exiting.\n");
    return -1;
  }
  queue_local_pad = gst_element_get_static_pad (queue_local,"sink");
  if (!queue_local_pad) {
    g_printerr ("Queue local pad request failed. Exiting.\n");
    return -1;
  }

  tee_enet_pad = gst_element_get_request_pad (tee,"src_%u");
  if (!tee_enet_pad) {
    g_printerr ("Tee enet pad request failed. Exiting.\n");
    return -1;
  }

  queue_enet_pad = gst_element_get_static_pad (queue_enet,"sink");
  if (!queue_enet_pad) {
    g_printerr ("Tee enet pad request failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (tee_local_pad, queue_local_pad) != GST_PAD_LINK_OK ||
      gst_pad_link (tee_enet_pad, queue_enet_pad) != GST_PAD_LINK_OK){
   
    g_printerr ("Tee goes wrong\n");
    gst_object_unref (pipeline);
    return -1;
  }

  gst_object_unref (queue_local_pad);
  gst_object_unref (queue_enet_pad);

  /* Start playing */
  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr ("Unable to set the pipeline to the playing state.\n");
    gst_object_unref (pipeline);
    return -1;
  }

  /* Wait until error or EOS */
  bus = gst_element_get_bus (pipeline);
  msg =
      gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
      GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

  gst_element_release_request_pad (tee, tee_local_pad);
  gst_element_release_request_pad (tee, tee_enet_pad);
  gst_object_unref (tee_local_pad);
  gst_object_unref (tee_enet_pad);

  /* Parse message */
  if (msg != NULL) {
    GError *err;
    gchar *debug_info;

    switch (GST_MESSAGE_TYPE (msg)) {
      case GST_MESSAGE_ERROR:
        gst_message_parse_error (msg, &err, &debug_info);
        g_printerr ("Error received from element %s: %s\n",
            GST_OBJECT_NAME (msg->src), err->message);
        g_printerr ("Debugging information: %s\n",
            debug_info ? debug_info : "none");
        g_clear_error (&err);
        g_free (debug_info);
        break;
      case GST_MESSAGE_EOS:
        g_print ("End-Of-Stream reached.\n");
        break;
      default:
        /* We should not reach here because we only asked for ERRORs and EOS */
        g_printerr ("Unexpected message received.\n");
        break;
    }
    gst_message_unref (msg);
  }

  /* Free resources */
  gst_object_unref (bus);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return 0;
}

