This structure is pure

                            |-> streammux -> pgie -> converter -> osd -> fakesink
    
    source -> filter -> tee 

                            |-> queue -> converter -> encoder -> payer -> enetsink


Macroses:

- HOST_ENET - IPv4 adress of video/metadata receiver
- HOST_PORT_VIDEO - port to send video to
- HOST_PORT_MSG - port to send UDP packets with metadata attached 
- HOST_RECORD - port to listen for getting record start/stop commands

