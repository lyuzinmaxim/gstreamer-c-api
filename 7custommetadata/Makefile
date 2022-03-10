CUDA_VER?=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream_custom_app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=6.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
  CFLAGS+= -DTHPOOL_DEBUG
endif

SRCS:= $(wildcard *.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0
#PKGS+= thread

OBJS:= $(SRCS:.c=.o)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-6.0/sources/includes \
		-I /usr/local/cuda-$(CUDA_VER)/include \
		-I /home/maxim/C-Thread-Pool

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lpthread \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


