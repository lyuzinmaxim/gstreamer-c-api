Client is sending messages (byte array) to host for starting/finishing video recording.

Use client 
```
cd host && gcc video_client.c -o client

./client
```

Use of server (smartrecord)
```
sudo make

sudo ./deepstream-testsr-app 
```

