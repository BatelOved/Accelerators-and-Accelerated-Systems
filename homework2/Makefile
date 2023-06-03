DEBUG=0

CFLAGS=-Xcompiler=-Wall -maxrregcount=32 -arch=sm_75
CFLAGS+=`pkg-config opencv --cflags --libs`
# Use to find out shared memory size
# CFLAGS+=--ptxas-options=-v 
CFLAGS+=--nvlink-options --verbose

RANDOMIZE_IMAGES_CFLAGS:=$(CFLAGS)
RANDOMIZE_IMAGES_CFLAGS+=-O3 -lineinfo

ifneq ($(DEBUG), 0)
CFLAGS+=-O0 -g -G
else
CFLAGS+=-O3 -lineinfo
endif


FILES=ex2 hello-shmem

all: $(FILES)

ex2: ex2.o main.o ex2-cpu.o randomize_images.o
	nvcc --link $(CFLAGS) $^ -o $@
hello-shmem: hello-shmem.o
	nvcc --link $(CFLAGS) $^ -o $@

ex2.o: ex2.cu ex2.h ex2-cpu.h
main.o: main.cu ex2.h randomize_images.h
ex2-cpu.o: ex2-cpu.h ex2-cpu.cu
randomize_images.o: randomize_images.cu randomize_images.h
	nvcc --compile $(CPPFLAGS) $< $(RANDOMIZE_IMAGES_CFLAGS) -o $@

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
