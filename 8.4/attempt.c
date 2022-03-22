#include <gst/gst.h>
#include <glib.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <sys/timeb.h>
			
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <byteswap.h>
#include <pthread.h>
#include <stdint.h>
#include "thpool.h"

#include <fcntl.h>
#include <poll.h>

#include "gst-nvdssr.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#define PGIE_CLASS_ID 0
#define PGIE_CONFIG_FILE  "infer_config.txt"
#define MSCONV_CONFIG_FILE "dstest4_msgconv_config.txt"
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define SMART_REC_CONTAINER 0
#define CACHE_SIZE_SEC 3
#define SMART_REC_DEFAULT_DURATION 100
#define START_TIME 0
#define SMART_REC_DURATION 100
#define SMART_REC_INTERVAL 0.5
			
#define HOST_ENET 		"192.168.0.107"
#define HOST_RECORD 	8080
#define HOST_PORT_VIDEO 31990
#define HOST_PORT_MSG 	52000

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33000

static gchar *cfg_file = "cfg_amqp.txt";
static gchar *topic = NULL;
static gchar *conn_str = NULL;
static gchar *proto_lib = "libnvds_amqp_proto.so";
static gint schema_type = 0;
static gint msg2p_meta = 0;
static gint frame_interval = 1;
static gboolean display_off = FALSE;

gint frame_number = 0;

struct Coords {
    int frame;
    int top;
    int left;
    int width;
    int height;
    float conf;
	GstClockTime time_buf;
    int sockfd;
    struct sockaddr_in servaddr;
	struct sockaddr_in cliaddr;
};

struct Data {
    NvDsSRContext* nvdssrctx;
    struct Coords coords;
};

char* receive_payload(struct Coords * structure){
    
	struct pollfd pfds[1]; // More if you want to monitor more
    pfds[0].fd = structure->sockfd;          // Standard input
    pfds[0].events = POLLIN; 
		
    int len, n;
    static char msg[512];
	
	printf("Hit RETURN or wait 2.5 seconds for timeout\n");
    int num_events = poll(pfds, 1, -1); // inf timeout
	
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

struct Coords establish_connection
(const char *client,unsigned short int *port, int server){
	
	int sockfd;
	struct sockaddr_in servaddr, cliaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
	
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { ///socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0)
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
	
	servaddr.sin_family    = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = inet_addr(client);
    servaddr.sin_port = htons(*port);
	
	if (server==1){
		if ( bind(sockfd, (const struct sockaddr *)&servaddr, 
				sizeof(servaddr)) < 0 )
		{
			perror("bind failed");
			exit(EXIT_FAILURE);
		}
	}
	
	struct Coords coord;
    coord.sockfd = sockfd;
    coord.servaddr.sin_family = servaddr.sin_family;
    coord.servaddr.sin_port = servaddr.sin_port;
    coord.servaddr.sin_addr.s_addr = servaddr.sin_addr.s_addr;
    coord.cliaddr = cliaddr;
	
	//fcntl(sockfd, F_SETFL, O_NONBLOCK);
	//printf("\naaaaa\n");
	
	return coord;
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
  msg = receive_payload(&data->coords);
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

void send_bytes(struct Coords coord){
    
    int sockfd_ = coord.sockfd;
    uint16_t frame_ = coord.frame;
    uint16_t top_ = coord.top;
    uint16_t left_ = coord.left;
    uint16_t width_ = coord.width;
    uint16_t height_ = coord.height;
    float conf_ = coord.conf;

    unsigned char msg[18];
    
    msg[0] = 0xAA;
    msg[1] = 0xAA;
    msg[2] = (frame_ >> 8) & 0xFF;
    msg[3] = (frame_ >> 0) & 0xFF;
    msg[4] = 0x00;
    msg[5] = 0x01;
	msg[6] = (left_ >> 8) & 0xFF;
    msg[7] = (left_ >> 0) & 0xFF;
    msg[8] = (top_ >> 8) & 0xFF;
    msg[9] = (top_ >> 0) & 0xFF;
    msg[10] = (width_ >> 8) & 0xFF;
    msg[11] = (width_ >> 0) & 0xFF;
    msg[12] = (height_ >> 8) & 0xFF;
    msg[13] = (height_ >> 0) & 0xFF;

    unsigned char conf__[sizeof (float)];
    memcpy (conf__, &conf_, sizeof (conf_));

    msg[14] = conf__[3];
    msg[15] = conf__[2];
    msg[16] = conf__[1];
    msg[17] = conf__[0];

/*    for (int i = 0; i < 18; i++)
    { 
     printf("%d: %02X ",i, msg[i]);
    }*/
      
    sendto( coord.sockfd, 
			msg, 
			sizeof(msg),
			MSG_CONFIRM, 
			(const struct sockaddr *) &coord.servaddr,
			sizeof(coord.servaddr));
}



/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    guint object_count = 0;
    gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj;
	
	GstClockTime t1 = GST_CLOCK_TIME_NONE;
	GstClockTime t2 = GST_CLOCK_TIME_NONE;
	
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
    // No batch meta attached.
		g_print("\nEMPTY\n");
		return GST_PAD_PROBE_OK;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
		
		frame_meta = (NvDsFrameMeta *) l_frame->data;

		if (frame_meta == NULL) {
		  // Ignore Null frame meta.
		  //g_print("\nEMPTY\n");
		  continue;
		}
				
		for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
			
			NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

			if (obj_meta == NULL) {
			// Ignore Null object.
			continue;
			}

			if (obj_meta->class_id == PGIE_CLASS_ID)
				object_count++;
				
				if (!(frame_number % frame_interval)) {

				struct Coords *coord1 = u_data;
				
				t1 = coord1->time_buf;
				t2 = g_get_monotonic_time();
				float result = 1/((t2 - t1)*0.000001);
				coord1->time_buf = t2;
				
				coord1->frame = frame_number;
				coord1->top = (int)obj_meta->rect_params.top;
				coord1->left = (int)obj_meta->rect_params.left;
				coord1->width = (int)obj_meta->rect_params.width;
				coord1->height = (int)obj_meta->rect_params.height;
				g_print("FPS %f top %d left %d width %d height %d\n",
					result,
					coord1->top, 
					coord1->left, 
					coord1->width, 
					coord1->height);

				send_bytes(*coord1);
				}
		  }
    }
  
    /*g_print ("%d frame, %d objects\n",
			frame_number, object_count);*/
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

void dummy(){
	printf("\n\nG_TIMEOUT_SUCCESSFUL\n\n");
}

int
main (int argc, char *argv[])
{
	GMainLoop *loop = NULL;
	GstElement *pipeline = NULL, *source = NULL, *filter = NULL, *tee = NULL;
	GstElement *nvstreammux = NULL, *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL, *fakesink = NULL;
	GstElement *queue = NULL, *nvvidconv_enet = NULL,  *encoder = NULL, *payer = NULL, *enetsink = NULL;
	GstElement *transform = NULL;
	
	GstElement *tee2 = NULL, *parser = NULL;
	NvDsSRContext *nvdssrCtx = NULL;

	
	GstBus *bus = NULL;
	GstCaps *filtercaps;
	guint bus_watch_id;
	GstPad *muxer_sink_pad = NULL;
	GstPad *osd_sink_pad = NULL;
	GstPad *tee_msg_pad = NULL;
	GstPad *tee_enet_pad = NULL;
	GstPad *sink_pad = NULL;
	GstPad *src_pad = NULL;
	GOptionContext *ctx = NULL;
	GOptionGroup *group = NULL;
	GError *error = NULL;
	
	NvDsSRInitParams params = { 0 };

	params.containerType = SMART_REC_CONTAINER;
	params.cacheSize = CACHE_SIZE_SEC;
	params.defaultDuration = SMART_REC_DEFAULT_DURATION;
	params.callback = smart_record_callback;
	params.fileNamePrefix = "testing";
	params.dirpath = "/home/maxim/Videos";
		
	gst_init (&argc, &argv);

	loop = g_main_loop_new (NULL, FALSE);

	/* Create gstreamer elements */
	pipeline = gst_pipeline_new ("custom-pipeline");

	source = gst_element_factory_make ("nvarguscamerasrc", "source");
	filter = gst_element_factory_make ("capsfilter","filter");
	tee = gst_element_factory_make ("tee", "nvsink-tee");

	queue = gst_element_factory_make ("queue", "nvtee-que1");
	nvvidconv_enet = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter-enet");
	encoder = gst_element_factory_make ("nvv4l2h264enc","encoder");
	payer = gst_element_factory_make ("rtph264pay","payer");
	enetsink = gst_element_factory_make("udpsink","enetsink");

	nvstreammux = gst_element_factory_make ("nvstreammux", "nvstreammux");
	pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
	nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");//from NV12 to RGBA
	nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
	fakesink = gst_element_factory_make ("fakesink", "fakesink");

	tee2 = gst_element_factory_make ("tee", "nvsink-tee2");
	parser = gst_element_factory_make ("h264parse", "parser");
	if (NvDsSRCreate (&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
		g_printerr ("Failed to create smart record bin");
		return -1;
	}

	if ( !pipeline || !source || !filter || !tee  
		|| !nvstreammux || !pgie || !nvvidconv || !nvosd || !fakesink 
		|| !queue || !nvvidconv_enet || !encoder || !payer || !enetsink) {
		g_printerr ("One element could not be created. Exiting.\n");
		return -1;
	}

	if ( !tee2 || !parser ) {
		g_printerr ("New elements could not be created. Exiting.\n");
		return -1;
	}

	/* we set the input filename to the source element */
	g_object_set (source, "sensor-id", 0, NULL);
	g_object_set (source, "bufapi-version", 1, NULL);
	filtercaps = gst_caps_new_simple ("video/x-raw(memory:NVMM)",
			"format",G_TYPE_STRING,"NV12",
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

	g_object_set (payer, "config-interval", 1, NULL); //not surepayer

	g_object_set (enetsink, "host", HOST_ENET, NULL);
	g_object_set (enetsink, "port", HOST_PORT_VIDEO, NULL);
	g_object_set (enetsink, "sync", FALSE, NULL);

	g_object_set (G_OBJECT (nvstreammux), "batch-size", 1, NULL);
	g_object_set (  G_OBJECT (nvstreammux), 
					"width", MUXER_OUTPUT_WIDTH, 
					"height", MUXER_OUTPUT_HEIGHT,
					"batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
	g_object_set (G_OBJECT (pgie),
		"config-file-path", PGIE_CONFIG_FILE, NULL);


	/* we add a message handler */
	bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
	bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
	gst_object_unref (bus);

	/* Set up the pipeline */
	/* we add all elements into the pipeline */
	gst_bin_add_many (GST_BIN (pipeline),
		source, filter, tee,
		nvstreammux, pgie, nvvidconv, nvosd, fakesink, 
		queue, nvvidconv_enet, encoder, payer, enetsink,  NULL);

	gst_bin_add_many (GST_BIN (pipeline), tee2, parser, nvdssrCtx->recordbin, NULL);
/************************************************************************/

	tee_msg_pad = gst_element_get_request_pad (tee, "src_%u");
	if (!tee_msg_pad) {
		g_printerr ("Unable to get request pads\n");
		return -1;
	}

	muxer_sink_pad = gst_element_get_request_pad (nvstreammux, "sink_0");
	if (!muxer_sink_pad) {
		g_printerr ("Streammux request sink pad failed. Exiting.\n");
		return -1;
	}

	if (gst_pad_link (tee_msg_pad, muxer_sink_pad) != GST_PAD_LINK_OK) {
		g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
		return -1;
	}

	gst_object_unref (tee_msg_pad);
	gst_object_unref (muxer_sink_pad); 

/************************************************************************/

	sink_pad = gst_element_get_static_pad (queue, "sink");
	tee_enet_pad = gst_element_get_request_pad (tee, "src_%u");
	if (!sink_pad || !tee_enet_pad) {
		g_printerr ("Unable to get request pads\n");
		return -1;
	}

	if (gst_pad_link (tee_enet_pad, sink_pad) != GST_PAD_LINK_OK) {
		g_printerr ("Unable to link tee and message converter\n");
		gst_object_unref (sink_pad);
		return -1;
	}

	gst_object_unref (sink_pad);
	gst_object_unref (tee_enet_pad);

/************************************************************************/
	GstPad *tee_msg_pad2 = NULL;
	GstPad *muxer_sink_pad2 = NULL;
	

	tee_msg_pad2 = gst_element_get_request_pad (tee2, "src_%u");
	if (!tee_msg_pad2) {
		g_printerr ("Unable to get request pads\n");
		return -1;
	}

	muxer_sink_pad2 = gst_element_get_static_pad (payer, "sink");
	if (!muxer_sink_pad2) {
		g_printerr ("Streammux request sink pad failed. Exiting.\n");
		return -1;
	}

	if (gst_pad_link (tee_msg_pad2, muxer_sink_pad2) != GST_PAD_LINK_OK) {
		g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
		return -1;
	}

	gst_object_unref (tee_msg_pad2);
	gst_object_unref (muxer_sink_pad2); 

/************************************************************************/

	GstPad *tee_rec_pad2 = NULL;
	GstPad *muxer_rec_pad2 = NULL;
	
	tee_rec_pad2 = gst_element_get_request_pad (tee2, "src_%u");
	if (!tee_rec_pad2) {
		g_printerr ("Unable to get request pads\n");
		return -1;
	}

	muxer_rec_pad2 = gst_element_get_static_pad (parser, "sink");
	if (!muxer_rec_pad2) {
		g_printerr ("Streammux request sink pad failed. Exiting.\n");
		return -1;
	}

	if (gst_pad_link (tee_rec_pad2, muxer_rec_pad2) != GST_PAD_LINK_OK) {
		g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
		return -1;
	}

	gst_object_unref (tee_rec_pad2);
	gst_object_unref (muxer_rec_pad2);

/************************************************************************/

	if (!gst_element_link_many (source, filter, tee, NULL)) {
		g_printerr ("Elements could not be linked1. Exiting.\n");
		return -1;
	}

	if (!gst_element_link_many (nvstreammux, pgie, nvvidconv, nvosd, fakesink, NULL)) {
		g_printerr ("Elements could not be linked2. Exiting.\n");
		return -1;
	}
  
	if (!gst_element_link_many (queue, nvvidconv_enet, encoder, tee2, NULL)) {
		g_printerr ("Elements could not be linked3. Exiting.\n");
		return -1;
	}
	
	if (!gst_element_link_many (payer, enetsink, NULL)) {
		g_printerr ("Elements could not be linked4. Exiting.\n");
		return -1;
	}
	
	if (!gst_element_link_many (parser, nvdssrCtx->recordbin, NULL)) {
		g_printerr ("Elements could not be linked5. Exiting.\n");
		return -1;
	}

/************************************************************************/

	const char client [64] = HOST_ENET;
	unsigned short int port_msg = HOST_PORT_MSG;
	unsigned short int port_record = HOST_RECORD;
	
	struct Coords send_messages;
	send_messages = establish_connection(client,&port_msg, 0);	
	send_messages.time_buf = g_get_monotonic_time();
	
	struct Coords recording;
	recording = establish_connection("127.0.0.1",&port_record, 1);
	struct Data *data = g_new0(struct Data, 1);
	data->coords = recording;
	data->nvdssrctx = nvdssrCtx;
	
/************************************************************************/

	osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
	if (!osd_sink_pad)
	g_print ("Unable to get sink pad\n");
	else {
		if(msg2p_meta == 0) //generate payload using eventMsgMeta
			gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
				osd_sink_pad_buffer_probe, &send_messages, NULL);
    }
	gst_object_unref (osd_sink_pad);

/************************************************************************/
	
	//g_timeout_add (1 * 1000, dummy, NULL);
	
	if (nvdssrCtx) {
		g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator, data);
	}
	
  /* Set the pipeline to "playing" state */
	g_print ("Now playing...\n");
	gst_element_set_state (pipeline, GST_STATE_PLAYING);
	
  /* Wait till pipeline encounters an error or EOS */
	g_print ("Running...\n");
	g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
	g_print ("Returned, stopping playback\n");

	g_free (cfg_file);
	g_free (topic);
	g_free (conn_str);
	g_free (proto_lib);

  /* Release the request pads from the tee, and unref them */
	gst_element_release_request_pad (tee, tee_msg_pad);
	gst_element_release_request_pad (tee, tee_enet_pad);
	gst_object_unref (tee_msg_pad);
	gst_object_unref (tee_enet_pad);

	gst_element_set_state (pipeline, GST_STATE_NULL);
	g_print ("Deleting pipeline\n");
	gst_object_unref (GST_OBJECT (pipeline));
	g_source_remove (bus_watch_id);
	g_main_loop_unref (loop);
	return 0;
}
