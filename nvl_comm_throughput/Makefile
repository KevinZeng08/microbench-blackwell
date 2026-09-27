NVCC ?= nvcc
ARCH ?= sm_100

all: libnvl_comm.so

libnvl_comm.so: kernels.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo --shared -Xcompiler=-fPIC $< -o $@ -lcuda

clean:
	rm -f libnvl_comm.so

.PHONY: all clean
