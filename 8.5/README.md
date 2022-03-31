There are used 3 structures types: 
- Coords (metadata handling & socket info)
- Data (handling deepstream smart record and socket receiving)
- Eth_uart (incapsulate two coords structures to send to two different hosts at the same time)


Every Coords structure is initialized using ```struct Coords establish_connection
(const char *client,unsigned short int *port, int server)``` function

# How are structures used:
- send_messages - sending metadata to host PC (macroses HOST_ENET, HOST_PORT_MSG)
- uart - sending metadata only for most confident object (macros UART_PORT) (localhost should be used)
- recording - receiving message to start/stop video recording (macroses HOST_ENET,HOST_RECORD)

# Following functions are used:
## receive_payload 
function, that waits (inf time) on socket for reading possibility	and if smth is ready reads the data.

args: struct Coords * structure (in particular, int sockfd)
returns: pointer to msg - incoming byte array
  
## establish_connection 
function, that creates an UDP socket & binds it

args: 
		*client, pointer to const char - IPv4 adress, from
				which come datagrams
		*port, pointer to unsigned short int - port to "listen"
		
returns: instance of struct Coords with filled fields 
		sockfd, servaddr, cliaddr
    
## smart_record_callback 
function used in NvDsSRContext initialization parameters
## send_bytes 
void function that generates a UDP data packet and sends them to host

args: struct Coords coods - filled strcuture with IP&port to send to

## osd_sink_pad_buffer_probe 
function that takes the metadata from gstreamer (deepstream) buffer and calls ```send_bytes```

## bus_call 
default gstreamer function
