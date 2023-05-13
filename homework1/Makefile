DEBUG=0
TESTS=1
PROFILE=0

CFLAGS+="-std=c++14"
CFLAGS+=-Xcompiler=-Wall -arch=sm_75
CFLAGS+=`pkg-config opencv --cflags`

CPPFLAGS=

ifneq ($(TESTS), 0)
CPPFLAGS+="-DUNIT_TEST"
endif

RANDOMIZE_IMAGES_CFLAGS:=$(CFLAGS)
RANDOMIZE_IMAGES_CFLAGS+=-O3 -lineinfo

ifneq ($(DEBUG), 0)
CFLAGS+=-O0 -g -G
else
CFLAGS+=-O3 -lineinfo
endif

ifneq ($(PROFILE), 0)
CFLAGS+=-g -pg
endif

FILES=ex1 image

ifeq ($(TESTS), 1)
FILES+=tests
endif

all: $(FILES)

ex1: ex1.o main.o ex1-cpu.o randomize_images.o
	nvcc --link $(CFLAGS) $^  -o $@

image: ex1.o ex1-cpu.o image.o
	nvcc --link $(CFLAGS) `pkg-config opencv --libs` $^ -o $@

tests: ex1.o ex1-cpu.o tests.o randomize_images.o
	nvcc --link $(CFLAGS) -lgtest $^ -o $@

ex1.o: ex1.cu ex1.h
main.o: main.cu ex1.h randomize_images.h
image.o: image.cu ex1.h
tests.o: tests.cu ex1.h randomize_images.h
randomize_images.o: randomize_images.cu randomize_images.h
	nvcc --compile $(CPPFLAGS) $< $(RANDOMIZE_IMAGES_CFLAGS) -o $@

%.o: %.cu
	nvcc --compile $(CPPFLAGS) $< $(CFLAGS) -o $@

clean::
	rm -f *.o image ex1 tests
