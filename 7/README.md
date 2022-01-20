This example is based on *deepstream_test4* sample applecation published by NVIDIA

I configure AMQP server on host PC, using RabbitMQ (Erlang based) and GUI (according to simple instructions on official site)

Deepstream pipeline looks like:

![Deepstream metadata sending](https://github.com/lyuzinmaxim/gstreamer-c-api/blob/1e407025151cf51dd5de3324e86c25481904caca/docs/Metadata_sending.png)

Different configuring files are used there:
- **cfg_amqp.txt** - broker settings
- **dstest1_pgie_config.txt** - inference settings
- **dstest4_msgconv_config.txt** - metadata settings

Main connection schema looks like:

![Connections](https://github.com/lyuzinmaxim/gstreamer-c-api/blob/38180de0734dfeb68062fe0645513e8b6878fe0f/docs/RabbitMQ.drawio.png)

RabbitMQ settings:
- create queue named **modem_queue** with durable=YES and autodelete=FALSE
- link that queue to new exchange named **modem_exchange** with routing key **modem**
- wait for connecting :)
- P.S. You must can ping jetson from host PC and must be able to send cimple python commands from jetson to host (using PIKA library)

I use SimpleRabbitmqClient (C++) over rammitmq-c (C) as consumer.
