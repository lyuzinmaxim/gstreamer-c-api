#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
   
#include <pthread.h>
#include "thpool.h"
#include <fcntl.h>
#include <poll.h>

//#include <gst/gst.h>



#define PORT     8080
//#define CLIENT "127.0.0.1"
#define CLIENT "192.168.0.107"

struct Connecting {
    int sockfd;
    struct sockaddr_in servaddr;
    struct sockaddr_in cliaddr;
};


char* receive_payload(struct Connecting * structure){
    
	struct pollfd pfds[1]; // More if you want to monitor more
    pfds[0].fd = structure->sockfd;          // Standard input
    pfds[0].events = POLLIN; 
		
    int len, n;
    static char msg[512];
	
	printf("Hit RETURN or wait 2.5 seconds for timeout\n");
    int num_events = poll(pfds, 1, -1); // 2.5 second timeout
	
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

void calling(struct Connecting * structure){
	char* msg;
    msg = receive_payload(structure);
	
	int run;
	if (msg[0] == 0xAA){
		run = 1;
	} else {
		run = 0;
	}
	
	printf("\n run is %d\n",run);
	/*
	for (int i = 0; i < 18; i++){ 
		 printf("%d: %02X ",i, msg[i]);
	}
	*/
	
	//printf("\n");
	//printf("\n%02X\n",msg[0]);
	//printf("\n%02X of size \n", *msg);
}

struct Connecting establish_connection
(const char *client,unsigned short int *port){
	
	int sockfd;
	struct sockaddr_in servaddr, cliaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
	
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { ///socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0)
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
	
	servaddr.sin_family    = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY; //inet_addr(client);
    servaddr.sin_port = htons(*port);
	
	
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
	
	fcntl(sockfd, F_SETFL, O_NONBLOCK);
	//printf("\naaaaa\n");
	
	return connect;
}

int main() {
    
	const char client [64] = CLIENT;
	unsigned short int port = PORT;
	
	struct Connecting connect;
    connect = establish_connection(client,&port);
	calling(&connect);
	
	//receive_payload(&connect);

    /* Confirming message (sending)*/
    /*const char *confirm_msg = "OK";
    sendto(sockfd, (const char *)confirm_msg, strlen(confirm_msg), 
        MSG_CONFIRM, (const struct sockaddr *) &cliaddr,
            len);
    printf("I sent confirm message!\n"); */
    //close(sockfd);
    //printf("I sent confirm message!\n"); 
	
    return 0;
}
