#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

// gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080, framerate=(fraction)30/1' ! \
// ! queue ! nvvideoconvert !  nvv4l2h264enc  bitrate=1000000 ! rtph264pay ! udpsink host=192.168.0.1 port=5000

int
main (int argc, char *argv[])
{
  GstElement *pipeline, *source, *filter, *queue, *converter, *encoder, *payer, *sink;
  GstBus *bus;
  GstMessage *msg;
  GstCaps *filtercaps;
  GstStateChangeReturn ret;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Create the elements */
  source = gst_element_factory_make ("videotestsrc", "source");
  filter = gst_element_factory_make ("capsfilter","filter");
  queue = gst_element_factory_make ("queue","queue");
  converter = gst_element_factory_make ("nvvideoconvert","converter");
  encoder = gst_element_factory_make ("nvv4l2h264enc","encoder");
  payer = gst_element_factory_make ("rtph264pay","payer");
  sink = gst_element_factory_make ("udpsink", "sink");
 
  /* Create the empty pipeline */
  pipeline = gst_pipeline_new ("maxim-pipeline");

  if (!pipeline || !source || !filter || !queue || !converter || !encoder || !payer || !sink) {
    g_printerr ("Not all elements could be created.\n");
    return -1;
  }

  /* Build the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), source, filter, queue, converter, encoder, payer, sink, NULL);
  
  if (!gst_element_link_many (source,filter,queue,NULL)){
	  	  g_printerr ("Source->filter->queue problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }
  
  if (!gst_element_link_many (queue,converter,encoder,NULL)){
                  g_printerr ("Queue->converter->encoder problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }


  if (!gst_element_link_many(encoder,payer,sink,NULL)){
                  g_printerr ("Encoder->payer->sink problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }

  /* Modify the properties */
  g_object_set (encoder, "bitrate", 2000000, NULL);
  
  g_object_set (sink, "host","192.168.0.1", NULL);
  g_object_set (sink, "port",5000, NULL);
 
  g_object_set (sink, "sync", "FALSE", NULL);
  
  filtercaps = gst_caps_new_simple ("video/x-raw",
	  "format",G_TYPE_STRING,"I420",
          "width", G_TYPE_INT, 1920,
          "height", G_TYPE_INT, 1080,
          "framerate",GST_TYPE_FRACTION,30,1,
 	  NULL);
  
  g_object_set (filter, "caps", filtercaps, NULL);
  gst_caps_unref (filtercaps);


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

