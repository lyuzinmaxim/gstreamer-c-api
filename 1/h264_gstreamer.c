#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

// gst-launch-1.0 filesrc location = sample_720p.h264 ! h264parse ! nvv4l2decoder !  autovideosink sync=0

int
main (int argc, char *argv[])
{
  GstElement *pipeline, *source, *parser, *decoder, *sink;
  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Create the elements */
  source = gst_element_factory_make ("filesrc", "source");
  parser = gst_element_factory_make ("h264parse","parser");
  decoder = gst_element_factory_make ("nvv4l2decoder","decoder");
  sink = gst_element_factory_make ("autovideosink", "sink");

  
  /* Create the empty pipeline */
  pipeline = gst_pipeline_new ("maxim-pipeline");

  if (!pipeline || !source || !parser || !decoder || !sink) {
    g_printerr ("Not all elements could be created.\n");
    return -1;
  }

  /* Build the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), source, parser, decoder, sink, NULL);
  
  if (!gst_element_link(source,parser)){
		  g_printerr ("Source->parser problem\n");
		  gst_object_unref (pipeline);
		  return -1;
  }
  
  if (!gst_element_link(parser,decoder)){
                  g_printerr ("Parser->decoder problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }

  if (!gst_element_link(decoder,sink)){
                  g_printerr ("Decoder->sink problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }


  /* Modify the source's properties */
  g_object_set (source, "location","/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264", NULL);
  g_object_set (sink, "sync", "FALSE", NULL);
  
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

