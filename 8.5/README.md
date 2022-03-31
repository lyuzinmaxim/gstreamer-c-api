There are used 3 structures types: 
- Coords (metadata handling & socket info)
- Data (handling deepstream smart record and socket receiving)
- Eth_uart (incapsulate two coords structures to send to two different hosts at the same time)


Every Coords structure is initialized using ```struct Coords establish_connection
(const char *client,unsigned short int *port, int server)``` function

How are structures used:
- send_messages - sending metadata to host PC (macroses HOST_ENET, HOST_PORT_MSG)
- uart - sending metadata only for most confident object (macros UART_PORT) (localhost should be used)
- recording - receiving message to start/stop video recording (macroses HOST_ENET,HOST_RECORD)

