#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

// gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080, framerate=(fraction)30/1' ! queue ! autovideosink sync=false 


/* 
1. Add GstElement
2. Add factory
3. add to bin
4. set properties
5. link
6. request pads (if needed)
*/

int
main (int argc, char *argv[])
{
  GstElement *pipeline, *source, *parser, *decoder, *tee;
  GstElement *queue_local, *muxer, *infer, *converter_local, *nvosd, *transform, *videosink;
  GstElement *queue_enet, *converter_enet, *encoder, *payer, *enetsink;
  
  GMainLoop *loop = NULL;

  GstPad *tee_local_pad, *tee_enet_pad;
  GstPad *queue_local_pad, *queue_enet_pad;
  GstPad *sinkpad, *srcpad;
  GstPad *osd_sink_pad;

  guint bus_watch_id;

  GstBus *bus;
  GstMessage *msg;
  GstCaps *filtercaps;
  GstStateChangeReturn ret;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Create the elements */
  
  source = gst_element_factory_make ("filesrc", "source");
  parser = gst_element_factory_make ("h264parse", "parser");
  decoder = gst_element_factory_make ("nvv4l2decoder", "decoder");
  tee = gst_element_factory_make ("tee", "tee");


  queue_local = gst_element_factory_make ("queue","queue_local");
  muxer = gst_element_factory_make ("nvstreammux", "muxer");
  infer = gst_element_factory_make ("nvinfer", "infer");
  converter_local = gst_element_factory_make ("nvvideoconvert", "converter_local");
  nvosd = gst_element_factory_make ("nvdsosd", "nvosd");
  if(prop.integrated) {
     transform = gst_element_factory_make ("nvegltransform", "transform");
  }
  videosink = gst_element_factory_make ("nveglglessink", "videosink");


  queue_enet = gst_element_factory_make ("queue","queue_enet");  
  converter_enet = gst_element_factory_make ("nvvideoconvert", "converter_enet"); 
  encoder = gst_element_factory_make ("nvv4l2h264enc","encoder");
  payer = gst_element_factory_make ("rtph264pay","payer");
  enetsink = gst_element_factory_make("udpsink","enetsink");

  /* Create the empty pipeline */
  pipeline = gst_pipeline_new ("maxim-pipeline");

  if (!pipeline || !source || !parser || !decoder || !tee ||
      !queue_local || !muxer || !infer || !converter_local || !nvosd || !videosink ||
      !queue_enet || !converter_enet || !encoder || !payer || !enetsink) {
    g_printerr ("Not all elements could be created.\n");
    return -1;
  }
   

  if(prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline),
        source, parser, decoder, tee,
        queue_local, muxer, infer, converter_local, nvosd, transform, videosink,
        queue_enet, converter_enet, encoder, payer, enetsink, NULL);
  }
  else {
    gst_bin_add_many (GST_BIN (pipeline),
        source, parser, decoder, tee,
        queue_local, muxer, infer, converter_local, nvosd, videosink,
        queue_enet, converter_enet, encoder, payer, enetsink, NULL);
  }
  /* Build the pipeline */

  if (!gst_element_link_many (source, parser, decoder, tee, NULL)){
		  g_printerr ("Source->parser->->decoder->tee(sink) problem\n");
		  gst_object_unref (pipeline);
		  return -1;
  }
  
/*  if(prop.integrated) {
    if (!gst_element_link_many (queue_local, muxer, infer, converter_local, nvosd, transform, enetsink, NULL)) {
      g_printerr ("Local branch cannot be created.\n");
      return -1;
    }
  }
  else {
    if (!gst_element_link_many (queue_local, muxer, infer, converter_local, nvosd, videosink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  }*/

  if (!gst_element_link_many(queue_enet, converter_enet, encoder, payer, enetsink,NULL)){
                  g_printerr ("Queue->convert->encode->pay->udpsink problem\n");
                  gst_object_unref (pipeline);
                  return -1;
  }
  /* Modify properties */

  g_object_set (source, "location", "/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264", NULL);

  g_object_set (muxer, "batch-size", 1, NULL);
  g_object_set (muxer, "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_set (infer, "config-file-path", "dstest1_pgie_config.txt", NULL);

  g_object_set (encoder, "bitrate", 2000000, NULL);
  g_object_set (encoder, "maxperf-enable", 1, NULL); //not sure if causes small latency 
  g_object_set (encoder, "preset-level", 4, NULL); //not sure too
  g_object_set (encoder, "profile", 2, NULL); //really makes latency small
  g_object_set (encoder, "ratecontrol-enable", 1, NULL); //not sure
  
  g_object_set (enetsink, "host", "192.168.0.1", NULL);
  g_object_set (enetsink, "port", 5000, NULL);
  g_object_set (enetsink, "sync", 0, NULL);

    
  /* Request and link pads */

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
  gst_object_unref (tee_local_pad);
  gst_object_unref (tee_enet_pad);

  sinkpad = gst_element_get_request_pad (muxer, "sink_0");
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }
  srcpad = gst_element_get_static_pad (queue_local, "src");
  if (!srcpad) {
    g_printerr ("Queue request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link queue to stream muxer. Exiting.\n");
      return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);
  

  /* Some things */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Start playing */
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}

