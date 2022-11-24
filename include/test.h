#ifndef TEST_H
#define TEST_H

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <pthread.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <signal.h>
#include <memory>
#include <stdlib.h>
#include <c10/cuda/CUDAFunctions.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sched.h>
#include <signal.h>
#include <nvToolsExt.h>

#include "thpool.h"
#include "alex.h"
#include "vgg.h"
#include "resnet.h"
#include "densenet.h"
#include "squeeze.h"
#include "mobile.h"
#include "mnasnet.h"
#include "inception.h"
#include "shuffle.h"
#include "efficient.h"
#include "regnet.h"

#define n_streamPerPool 32
#define n_Branch 3  //only use for branch

extern threadpool thpool; 
extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern c10::DeviceIndex GPU_NUM;
extern int *cond_i;
extern std::vector<at::cuda::CUDAStream> streams;
extern double *start_min,*end_max;
double what_time_is_it_now();

#endif
