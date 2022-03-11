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
#define SMART_REC_DEFAULT_DURATION 100
#define START_TIME 2
#define SMART_REC_DURATION 100
#define SMART_REC_INTERVAL 3

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
static GstElement *parser = NULL;
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
  guint run = 0;
  if (run==0) {
    printf("Enter 0 to do nothing, 1 for start/stop! \n");
    scanf("%d", &run);
	if (run==1) {
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
	} else {
	    printf("ZERO was passed\n");
	}
	
  run = 0;
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
  GstElement *parser = gst_element_factory_make ("h264parse", "parser");
  gst_bin_add_many (GST_BIN (pipeline), parser, NULL);

  if (!gst_element_link_many (tee, parser, nvdssrCtx->recordbin, NULL)) {
    g_print ("Elements not linked. Exiting. \n");
    g_main_loop_quit(loop);
  }
  gst_element_sync_state_with_parent(parser);
  gst_caps_unref (caps);
}


int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *source, *filter, *depayer, *decoder, *parser;
  GstElement *videosink;
  GstCaps *filtercaps;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *tee_output_pad, *tee_save_pad;
  GstPad *queue_local_pad, *queue_enet_pad;
  
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

  source = gst_element_factory_make ("udpsrc", "rtsp-source");
  filter = gst_element_factory_make ("capsfilter","filter");
  depayer = gst_element_factory_make ("rtph264depay", "depay");
  parser = gst_element_factory_make ("h264parse", "parser");
  tee = gst_element_factory_make ("tee", "tee");
  decoder = gst_element_factory_make ("nvv4l2decoder", "decoder");
  videosink = gst_element_factory_make ("autovideosink", "sink");

  if (!source || !filter || !depayer || !parser || !decoder || !tee || !videosink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  

/* Set elements properties */
  //g_object_set (source, "location", "rtsp://127.0.0.1:8554/test", NULL);
  g_object_set (source, "port", 5000, NULL);

  filtercaps = gst_caps_new_simple ("application/x-rtp",
	  "media",G_TYPE_STRING,"video",
          "clock-rate", G_TYPE_INT, 90000,
          "encoding-name", G_TYPE_STRING, "H264",
          "payload",G_TYPE_INT,96, NULL);

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
        source, filter, depayer, parser, decoder, tee, videosink, NULL);
  

  /* Link elements */
  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */
//  g_signal_connect (G_OBJECT (source), "pad-added",
//      G_CALLBACK (cb_newpad), depayer);

  //g_signal_connect (G_OBJECT (source), "pad-added",
  //    G_CALLBACK (cb_newpad), depayer);

  /*if (!gst_element_link_many (source, filter, depayer, parser, tee, decoder, videosink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }*/
  
  if (!gst_element_link_many (source, filter, depayer, parser, tee, decoder, videosink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
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
  
  /*if (nvdssrCtx) {
    g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        nvdssrCtx);
  }*/

  /*g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        nvdssrCtx);*/
  
  /*char mode[20];
  scanf("%s",mode);
  printf("Mode is: %s",mode);*/
  
  /*if (true) {
    smart_record_event_generator(nvdssrCtx);
  };*/

  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  if (pipeline && nvdssrCtx) {
    if(NvDsSRDestroy (nvdssrCtx) != NVDSSR_STATUS_OK)
    g_printerr ("Unable to destroy recording instance\n");
  }
  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
