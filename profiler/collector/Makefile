# profiler/collector — CUPTI Activity 커널 트레이서 (Exp_19 검증본의 승격 사본)
# 빌드 명령은 kernel_tracer.c 헤더 주석의 검증된 라인 그대로.
CUDA ?= /usr/local/cuda

kernel_tracer.so: kernel_tracer.c
	gcc -shared -fPIC -O2 kernel_tracer.c -o kernel_tracer.so \
	  -I$(CUDA)/extras/CUPTI/include -I$(CUDA)/include \
	  -L$(CUDA)/extras/CUPTI/lib64 -lcupti

clean:
	rm -f kernel_tracer.so

.PHONY: clean
