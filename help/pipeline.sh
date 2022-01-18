#Kafka

#./deepstream-test4-app \
#        -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
#        -p libnvds_kafka_proto.so \
#        --conn-str="10.0.111.10;9092;modem" \
#        -s 0



#./deepstream-test4-app \
#	-i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
#	-p libnvds_kafka_proto.so \
#	--conn-str="192.168.0.203;9092;connection" \
#	-s 0

#/deepstream-test4-app \
#       -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
#       -p libnvds_kafka_proto.so \
#       --conn-str="localhost;9092;test" \
#       --conn-str="192.168.0.203;9092;connection" \
#       --topic=connection \
#       --frame-interval 1 \
#       -s 0

#AMQP
./deepstream-test4-app \
        -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
        -p libnvds_amqp_proto.so \
	--cfg-file cfg_amqp.txt \
	-s 0

#./deepstream-test4-app \
#        -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
#        -p libnvds_amqp_proto.so \
#        --conn-str "10.0.111.10;5672;jetson;1111" \
#        -s 0


#Redis

#./deepstream-test4-app \
#       -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.h264 \
#       -p libnvds_redis_proto.so \
#       --conn-str="localhost;6379" \
#       --topic=topic1 \
#      --msg2p-meta 1 \
#      --frame-interval=1 \
#      -s 0

