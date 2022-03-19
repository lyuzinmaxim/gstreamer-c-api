#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "/usr/local/cuda-10.2/targets/aarch64-linux/include/cuda_runtime_api.h"
#include "/opt/nvidia/deepstream/deepstream-6.0/sources/includes/gstnvdsmeta.h"
#include <string.h>
#include "gst-nvdssr.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <fcntl.h>
#include <poll.h>

#define SMART_REC_CONTAINER 0
#define CACHE_SIZE_SEC 3
#define SMART_REC_DEFAULT_DURATION 100
#define START_TIME 0
#define SMART_REC_DURATION 100
#define SMART_REC_INTERVAL 0.5

#define PORT     8080
#define CLIENT "127.0.0.1"

static GMainLoop *loop = NULL;
static GstElement *tee = NULL;
static GstElement *converter = NULL;
static GstElement *parser = NULL;
static GstElement *pipeline = NULL;
static NvDsSRContext *nvdssrCtx = NULL;

struct Connecting {
    int sockfd;
    int foo;
    struct sockaddr_in servaddr;
    struct sockaddr_in cliaddr;
}; 

struct Data {
    NvDsSRContext* nvdssrctx;
    struct Connecting connect;
};


char* receive_payload(struct Connecting * structure){
    
	struct pollfd pfds[1]; // More if you want to monitor more
    pfds[0].fd = structure->sockfd;          // Standard input
    pfds[0].events = POLLIN; 
		
    int len, n;
    static char msg[512];
	
	printf("Hit RETURN or wait 2.5 seconds for timeout\n");
    int num_events = poll(pfds, 1, 10000); // 2.5 second timeout
	
	if (num_events == 0) {
		printf("Poll timed out!\n");
	} else {
		int pollin_happened = pfds[0].revents & POLLIN;
         
		if (pollin_happened) {
			printf("File descriptor %d is ready to read\n", pfds[0].fd);
		} else {
			printf("Unexpected event occurred: %d\n", pfds[0].revents);
		}
			
		read(pfds[0].fd, &msg, 18);

	}
	//printf("\n that is size %d\n",sizeof(msg));
	
	return msg;
}

struct Connecting establish_connection(){
	
	int sockfd;
	struct sockaddr_in servaddr, cliaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
	
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { ///socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0)
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
	
	servaddr.sin_family    = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = inet_addr(CLIENT);
    servaddr.sin_port = htons(PORT);
	
	if ( bind(sockfd, (const struct sockaddr *)&servaddr, 
            sizeof(servaddr)) < 0 )
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
	
	struct Connecting connect;
    connect.sockfd = sockfd;
    connect.servaddr.sin_family = servaddr.sin_family;
    connect.servaddr.sin_port = servaddr.sin_port;
    connect.servaddr.sin_addr.s_addr = servaddr.sin_addr.s_addr;
    connect.cliaddr = cliaddr;
	
	//fcntl(sockfd, F_SETFL, O_NONBLOCK);
	//printf("\naaaaa\n");
	
	return connect;
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
smart_record_event_generator (struct Data *  data)
{
  NvDsSRSessionId sessId = 0;
  NvDsSRContext *ctx = (NvDsSRContext *) data->nvdssrctx;
  guint startTime = START_TIME;
  guint duration = SMART_REC_DURATION;
  
  char* msg;
  msg = receive_payload(&data->connect);
  guint run = 0;
  
  
  if (run==0) {
	  
	if (msg[0] == 0xAA){
			run = 1;
	} else {
			run = 0;
	}
	
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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *source, *filter, *depayer, *parser;
  GstCaps *filtercaps;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  NvDsSRInitParams params = { 0 };

  params.containerType = SMART_REC_CONTAINER;
  params.cacheSize = CACHE_SIZE_SEC;
  params.defaultDuration = SMART_REC_DEFAULT_DURATION;
  params.callback = smart_record_callback;
  params.fileNamePrefix = "testing";
  params.dirpath = "/home/maxim/Videos";

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);


  pipeline = gst_pipeline_new ("dstest1-pipeline");

  source = gst_element_factory_make ("udpsrc", "udp-source");
  filter = gst_element_factory_make ("capsfilter","filter");
  depayer = gst_element_factory_make ("rtph264depay", "depay");
  parser = gst_element_factory_make ("h264parse", "parser");
  if (NvDsSRCreate (&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
    g_printerr ("Failed to create smart record bin");
    return -1;
  }

  if (!source || !filter || !depayer || !parser ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  filtercaps = gst_caps_new_simple ("application/x-rtp",
	  "media",G_TYPE_STRING,"video",
          "clock-rate", G_TYPE_INT, 90000,
          "encoding-name", G_TYPE_STRING, "H264",
          "payload",G_TYPE_INT,96,NULL);
  g_object_set (filter, "caps", filtercaps, NULL);
  gst_caps_unref (filtercaps);

  /* Set elements properties */
  g_object_set (source, "port", 5000, NULL);
  g_object_set (parser, "config-interval", -1, NULL);

  /* Adding to bin */
  gst_bin_add_many (GST_BIN (pipeline),
      source, filter, depayer, parser, NULL);
  gst_bin_add_many (GST_BIN (pipeline), nvdssrCtx->recordbin, NULL);

  /* Bus and linking */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Link all elements together:
	  src -> filter -> depayer ->  parser -> SmartRecordBin
  */ 

  if (!gst_element_link_many (source, filter, depayer, parser, nvdssrCtx->recordbin, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }
  //gst_element_sync_state_with_parent(parser2);
  
  ///**********************///
  //struct Connecting  *connect = g_new0(struct Connecting, 1);
  //struct Args *args = g_new0(struct Args, 1);
  //connect->sockfd = 10;

  //printf("\nIt's:%d\n",connect->sockfd);
  
  struct Connecting connect;
  connect = establish_connection();
  
  struct Data *data = g_new0(struct Data, 1);
  data->connect = connect;
  data->nvdssrctx = nvdssrCtx;
  
  /*
  struct Data data;
  data.nvdssrctx = nvdssrCtx;
  data.connect = connect;
  */
 
  //calling(&connect);
  
 
  ///**********************///
  
  /*
  if (nvdssrCtx) {
    g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        nvdssrCtx);
  }
  */
  if (nvdssrCtx) {
    g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        data);
  }

  g_print ("Now playing video. \n");
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
