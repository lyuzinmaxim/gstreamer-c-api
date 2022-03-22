This structure is pure

                            |-> streammux -> pgie -> converter -> osd -> fakesink
                            |
                            |
                            |
    source -> filter -> tee                                          |-> payer -> enetsink
                            |                                        |
                            |-> queue -> converter -> encoder -> tee  
                                                                     |
                                                                     |-> parser -> nvdssrCtx->recordbin 

Macroses:

- HOST_ENET - IPv4 adress of video/metadata receiver
- HOST_PORT_VIDEO - port to send video to
- HOST_PORT_MSG - port to send UDP packets with metadata attached 
- HOST_RECORD - port to listen for getting record start/stop commands

Using sockets:
- for metadata sending (server=0)
- for receiving video start/stop messages (server=1) - INADDR_ANY

