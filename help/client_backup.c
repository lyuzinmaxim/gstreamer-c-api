#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <byteswap.h>
#include <pthread.h>
#include <stdint.h>
#include "thpool.h"
   
#define PORT     8080
#define MAXLINE 1024

#define SERVER "10.0.100.11"

int open_socket(struct sockaddr_in servaddr){
    char buffer[MAXLINE];
    char *hello = "Hello from client";
    
    int n, len;
    
    int sockfd = 3;
    
    sendto(sockfd, (const char *)hello, strlen(hello),
        MSG_CONFIRM, (const struct sockaddr *) &servaddr, 
            sizeof(servaddr));
    printf("Hello message sent.\n");
    printf("Socket: %d servaddr.sin_family: %d servaddr.sin_port: %d servaddr.sin_addr.s_addr: %d\n", sockfd, servaddr.sin_family,servaddr.sin_port, servaddr.sin_addr.s_addr);
    close(sockfd);
    printf("Socket closed.\n");  
    
    return sockfd;
}


// Driver code
int main() {



    int sockfd;
    
    struct sockaddr_in     servaddr;
   
    // Creating socket file descriptor
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
   
    memset(&servaddr, 0, sizeof(servaddr));
       
    // Filling server information
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    //servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    /*threadpool thpool = thpool_init(1);
    thpool_add_work(thpool, open_socket, (void*)(struct sockaddr_in)servaddr);

    thpool_wait(thpool);
    thpool_destroy(thpool);*/

    open_socket (servaddr);
    return 0;
}
 
