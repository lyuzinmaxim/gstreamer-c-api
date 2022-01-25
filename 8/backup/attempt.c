#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "/usr/local/cuda-10.2/targets/aarch64-linux/include/cuda_runtime_api.h"
#include "/opt/nvidia/deepstream/deepstream-6.0/sources/includes/gstnvdsmeta.h"
#include <string.h>
#include "gst-nvdssr.h"
#define MAX_DISPLAY_LEN 64

#define SMART_REC_CONTAINER 0
#define CACHE_SIZE_SEC 15
#define SMART_REC_DEFAULT_DURATION 10
#define START_TIME 2
#define SMART_REC_DURATION 7
#define SMART_REC_INTERVAL 7

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


static GMainLoop *loop = NULL;
static GstElement *tee = NULL;
static GstElement *converter = NULL;
static GstElement *encoder = NULL;
static GstElement *pipeline = NULL;
static NvDsSRContext *nvdssrCtx = NULL;

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

static gpointer
smart_record_callback (NvDsSRRecordingInfo * info, gpointer userData)
{
  static GMutex mutex;
  FILE *logfile = NULL;
  g_return_val_if_fail (info, NULL);

  g_mutex_lock (&mutex);
  logfile = fopen ("smart_record.log", "a");
  if (logfile) {
    fprintf (logfile, "%d:%s:%d:%d:%s:%d channel(s):%d Hz:%ldms:%s:%s\n",
        info->sessionId, info->containsVideo ? "video" : "no-video",
        info->width, info->height, info->containsAudio ? "audio" : "no-audio",
        info->channels, info->samplingRate, info->duration,
        info->dirpath, info->filename);
    fclose (logfile);
  } else {
    g_print ("Error in opeing smart record log file\n");
  }
  g_mutex_unlock (&mutex);

  return NULL;
}

static gboolean
smart_record_event_generator (gpointer data)
{
  NvDsSRSessionId sessId = 0;
  NvDsSRContext *ctx = (NvDsSRContext *) data;
  guint startTime = START_TIME;
  guint duration = SMART_REC_DURATION;

  if (ctx->recordOn) {
    g_print ("Recording done.\n");
    if (NvDsSRStop (ctx, 0) != NVDSSR_STATUS_OK)
      g_printerr ("Unable to stop recording\n");
  } else {
    g_print ("Recording started..\n");
    if (NvDsSRStart (ctx, &sessId, startTime, duration,
            NULL) != NVDSSR_STATUS_OK)
      g_printerr ("Unable to start recording\n");
  }
  return TRUE;
}

static void
cb_newpad (GstElement * element, GstPad * element_src_pad, gpointer data)
{

  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (element_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  GstElement *depay_elem = (GstElement *) data;

  const gchar *media = gst_structure_get_string (str, "media");
  gboolean is_video = (!g_strcmp0 (media, "video"));
  gboolean is_audio = (!g_strcmp0 (media, "audio"));

  GstPad *sinkpad = gst_element_get_static_pad (depay_elem, "sink");
  if (gst_pad_link (element_src_pad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link depay loader to rtsp src");
  }
  gst_object_unref (sinkpad);

  //GstElement *converter = gst_element_factory_make ("nvvideoconvert", "parser-pre-recordbin");
  GstElement *encoder = gst_element_factory_make ("nvv4l2h264enc", "encoder");

  g_object_set (encoder, "bitrate", 6000000, NULL);


  //gst_bin_add_many (GST_BIN (pipeline), parser_pre_recordbin, NULL);

  if (!gst_element_link_many (tee, encoder, nvdssrCtx->recordbin, NULL)) {
    g_print ("Elements not linked. Exiting. \n");
    g_main_loop_quit(loop);
  }
  gst_element_sync_state_with_parent(encoder);
  gst_caps_unref (caps);
}


int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline, *source, *filter, *converter;
  GstElement *videosink;
  GstCaps *filtercaps;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *sinkpad, *srcpad;
  GstPad *tee_enet_pad, *queue_enet_pad;
  
  NvDsSRInitParams params = { 0 };

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

  source = gst_element_factory_make ("videotestsrc", "source");
  filter = gst_element_factory_make ("capsfilter","filter");
  converter = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  tee = gst_element_factory_make ("tee", "tee");
  videosink = gst_element_factory_make ("autovideosink", "sink");

  if (!source || !filter || !converter || !tee || !videosink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  

/* Set elements properties */
  g_object_set (source, "pattern", "ball", NULL);
  filtercaps = gst_caps_new_simple ("video/x-raw(memory:NVMM)",
	  "format",G_TYPE_STRING,"I420",
          "width", G_TYPE_INT, 20,
          "height", G_TYPE_INT, 108,
          "framerate",GST_TYPE_FRACTION,30,1,
 	  NULL);

  g_object_set (filter, "caps", filtercaps, NULL);
  gst_caps_unref (filtercaps);
  g_object_set (videosink, "sync", 0, NULL);
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  
  gst_bin_add_many (GST_BIN (pipeline),
        source, filter, converter, tee, videosink, NULL);
  

  /* Link elements */
  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */
  g_signal_connect (G_OBJECT (source), "pad-added",
      G_CALLBACK (cb_newpad),converter);

  if (!gst_element_link_many (source, filter, converter, tee, videosink, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  params.containerType = SMART_REC_CONTAINER;
  params.cacheSize = CACHE_SIZE_SEC;
  params.defaultDuration = SMART_REC_DEFAULT_DURATION;
  params.callback = smart_record_callback;
  params.fileNamePrefix = "testing";
 
  if (NvDsSRCreate (&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
    g_printerr ("Failed to create smart record bin");
    return -1;
  }

  gst_bin_add_many (GST_BIN (pipeline), nvdssrCtx->recordbin, NULL);

  
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  /* Set the pipeline to "playing" state */
  if (nvdssrCtx) {
    g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        nvdssrCtx);
  }
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
