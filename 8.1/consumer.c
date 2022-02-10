#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <amqp.h>
#include <amqp_tcp_socket.h>

#include <assert.h>

#include "utils.h"

int main(int argc, char const *const *argv) {
  char const *hostname;
  int port, status;
  char const *exchange;
  char const *bindingkey;
  amqp_socket_t *socket = NULL;
  amqp_connection_state_t conn;

  amqp_bytes_t queuename;

  hostname = "localhost";
  port = 5672;
  exchange = "amq.direct";
  bindingkey = "test";

  conn = amqp_new_connection();

  socket = amqp_tcp_socket_new(conn);
  if (!socket) {
    printf("socket error\n");
  }

  status = amqp_socket_open(socket, hostname, port);
  if (status) {
    printf("opening error\n");
  }

  amqp_login(conn, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN,
                               "guest", "guest");
  amqp_channel_open(conn, 1);
  amqp_get_rpc_reply(conn);

  {
    amqp_queue_declare_ok_t *r = amqp_queue_declare(
        conn, 1, amqp_empty_bytes, 0, 0, 0, 1, amqp_empty_table);
    amqp_get_rpc_reply(conn);
    queuename = amqp_bytes_malloc_dup(r->queue);
    if (queuename.bytes == NULL) {
      fprintf(stderr, "Out of memory while copying queue name");
      return 1;
    }
  }

  amqp_queue_bind(conn, 1, queuename, amqp_cstring_bytes(exchange),
                  amqp_cstring_bytes(bindingkey), amqp_empty_table);
  amqp_get_rpc_reply(conn);

  amqp_basic_consume(conn, 1, queuename, amqp_empty_bytes, 0, 1, 0,
                     amqp_empty_table);
  amqp_get_rpc_reply(conn);

  {
    for (;;) {
      amqp_rpc_reply_t res;
      amqp_envelope_t envelope;

      amqp_maybe_release_buffers(conn);

      res = amqp_consume_message(conn, &envelope, NULL, 0);

      if (AMQP_RESPONSE_NORMAL != res.reply_type) {
        break;
      }

      printf("Received message: %.*s\n",
	     (int)envelope.message.body.len,
             (char *)envelope.message.body.bytes);
      printf("----\n");

      amqp_destroy_envelope(&envelope);
    }
  }

/*  amqp_bytes_free(queuename);

  amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS);
  amqp_connection_close(conn, AMQP_REPLY_SUCCESS);
  amqp_destroy_connection(conn);*/

  return 0;
}
