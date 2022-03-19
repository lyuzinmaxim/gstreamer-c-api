#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <byteswap.h>


//#include <pthread.h>
#include <stdint.h>
//#include "thpool.h"
   

#define PORT     8080
#define SERVER "127.0.0.1"


struct Coords {
    int sockfd;
    int frame;
    int top;
    int left;
    int width;
    int height;
    float conf;
    struct sockaddr_in servaddr;
}; 

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

    for (int i = 0; i < 2; i++)
    { 
     printf("%d: %02X ",i, msg[i]);
    }
       
    int sockfd;      
    sendto(coord.sockfd, 
	  msg, 
	  sizeof(msg),
          MSG_CONFIRM, 
          (const struct sockaddr *) &coord.servaddr,
          sizeof(coord.servaddr));

    printf("\npayload sent.\n");

    /* Confirming message (receiving) */
    /*int n, len;
    char confirm_msg[10];
    n = recvfrom(coord.sockfd, (char *)confirm_msg, 512, 
                MSG_WAITALL, (struct sockaddr *) &coord.servaddr,
                &len);
    confirm_msg[n] = '\0';
    printf("I received confirm message: %s\n", confirm_msg);*/
    /*________________________________*/
}

void get_data_send(struct Coords * structure){
    
    uint16_t n = 253;
    
    struct Coords *coord1 = structure;
    coord1->frame = n;

    send_bytes(*coord1);
}

int main() {

    int sockfd;
    struct sockaddr_in     servaddr;
    if ( (sockfd = socket(AF_INET,  SOCK_DGRAM, 0)) < 0 ) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    memset(&servaddr, 0, sizeof(servaddr));       
    // Filling server information
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    //servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_addr.s_addr = inet_addr(SERVER); //"10.0.100.11"      

    struct Coords coord;
    coord.sockfd = sockfd;
    coord.servaddr.sin_family = servaddr.sin_family;
    coord.servaddr.sin_port = servaddr.sin_port;
    coord.servaddr.sin_addr.s_addr = servaddr.sin_addr.s_addr;
    get_data_send(&coord);
    //send_bytes(coord);
    /*threadpool thpool = thpool_init(1);
    thpool_add_work(thpool, (void*)send_bytes, &coord);

    thpool_wait(thpool);
    thpool_destroy(thpool);*/
    
    //close(sockfd);
    //printf("Socket closed.\n");

    return 0;
}
 
