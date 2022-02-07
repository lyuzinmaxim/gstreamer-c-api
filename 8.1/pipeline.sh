#!/bin/sh
#gcc -c -I/home/maxim/rabbitmq-c/librabbitmq -I/home/maxim/rabbitmq-c/examples -L/home/maxim/rabbitmq-c/build/librabbitmq producer.c -o test -lrabbitmq

gcc -I/home/maxim/rabbitmq-c/examples -I/home/maxim/rabbitmq-c/librabbitmq -L/home/maxim/rabbitmq-c/build/librabbitmq -o test producer.c -lrabbitmq
