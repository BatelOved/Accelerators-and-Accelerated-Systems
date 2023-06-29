DEBUG=0

CFLAGS+=-Xcompiler=-Wall -maxrregcount=32 -arch=sm_75
CFLAGS+=`pkg-config libibverbs --cflags --libs`

RANDOMIZE_IMAGES_CFLAGS:=$(CFLAGS)
RANDOMIZE_IMAGES_CFLAGS+=-O3 -lineinfo

ifneq ($(DEBUG), 0)
CFLAGS+=-O0 -g -G -DDEBUG=1
else
CFLAGS+=-O3 -g -lineinfo
endif

FILES=server client

all: $(FILES)

server: ex3.o server.o common.o
	nvcc --link $(CFLAGS) $^ -o $@
client: ex3.o client.o common.o ex3-cpu.o randomize_images.o
	nvcc --link $(CFLAGS) $^ -o $@

server.o: server.cu ex3.h
client.o: client.cu ex3.h
common.o: common.cu ex3.h
ex3.o: ex3.cu ex3.h ex2.cu ex2.h
ex3-cpu.o: ex3-cpu.cu ex3-cpu.h ex2.h ex3.h
randomize_images.o: randomize_images.cu randomize_images.h ex3.h
	nvcc --compile $(CPPFLAGS) $< $(RANDOMIZE_IMAGES_CFLAGS) -o $@

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
