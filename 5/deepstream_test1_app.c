/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline, *source, *parser, *decoder, *tee;
  GstElement *queue_local, *muxer, *infer, *converter_local, *nvosd, *videosink;
  GstElement *queue_enet, *converter_enet, *encoder, *payer, *enetsink;

  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *sinkpad, *srcpad;
  GstPad *tee_enet_pad, *queue_enet_pad;
  
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  /* Check input arguments */
  /*if (argc != 2) {
    g_printerr ("Usage: %s <H264 filename>\n", argv[0]);
    return -1;
  }*/

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest1-pipeline");

  source = gst_element_factory_make ("filesrc", "source");
  parser = gst_element_factory_make ("h264parse", "h264-parser");
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
  tee = gst_element_factory_make ("tee", "tee");

  queue_local = gst_element_factory_make ("queue","queue_local");
  muxer = gst_element_factory_make ("nvstreammux", "stream-muxer");
  if (!pipeline || !muxer) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  infer = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  converter_local = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  }
  videosink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!source || !parser || !decoder || !tee || !infer
      || !converter_local || !nvosd || !videosink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if(!transform && prop.integrated) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  queue_enet = gst_element_factory_make ("queue","queue_enet");  
  converter_enet = gst_element_factory_make ("nvvideoconvert", "converter_enet"); 
  encoder = gst_element_factory_make ("nvv4l2h264enc","encoder");
  payer = gst_element_factory_make ("rtph264pay","payer");
  enetsink = gst_element_factory_make("udpsink","enetsink");
  
/* Set elements properties */

  g_object_set (source, "location", "/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264",NULL);
  g_object_set (muxer, "batch-size", 1, NULL);
  g_object_set (muxer, "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_set (infer,
      "config-file-path", "dstest1_pgie_config.txt", NULL);

  g_object_set (encoder, "bitrate", 2000000, NULL);
  g_object_set (encoder, "maxperf-enable", 1, NULL); //not sure if causes small latency 
  g_object_set (encoder, "preset-level", 4, NULL); //not sure too
  g_object_set (encoder, "profile", 2, NULL); //really makes latency small
  g_object_set (encoder, "ratecontrol-enable", 1, NULL); //not sure
  
  g_object_set (enetsink, "host", "192.168.0.1", NULL);
  g_object_set (enetsink, "port", 5000, NULL);
  g_object_set (enetsink, "sync", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if(prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline),
        source, parser, decoder,tee,
        queue_local, muxer, infer, converter_local, nvosd, transform, videosink,
        queue_enet, converter_enet, encoder, payer, enetsink, NULL);
  }
  else {
    gst_bin_add_many (GST_BIN (pipeline),
        source, parser, decoder,tee,
        queue_local, muxer, infer, converter_local, nvosd, videosink,
        queue_enet, converter_enet, encoder, payer, enetsink, NULL);
  }

  /* Link elements */

  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (muxer, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_request_pad (tee, "src_%u");
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
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

  if (gst_pad_link (tee_enet_pad, queue_enet_pad) != GST_PAD_LINK_OK){
    g_printerr ("Tee goes wrong\n");
    gst_object_unref (pipeline);
    return -1;
  }

  gst_object_unref (tee_enet_pad);
  gst_object_unref (queue_enet_pad);
  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, parser, decoder, tee, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if(prop.integrated) {
    if (!gst_element_link_many (muxer, infer,
        converter_local, nvosd, transform, videosink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  }
  else {
    if (!gst_element_link_many (muxer, infer,
        converter_local, nvosd, videosink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  }
  
  if (!gst_element_link_many (queue_enet, converter_enet, encoder, payer, enetsink, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }
  
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
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
