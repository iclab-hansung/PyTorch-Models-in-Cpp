#include "test.h"
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
#include <cuda_profiler_api.h>

// #define n_dense 1
// #define n_res 0
// #define n_alex 0
// #define n_vgg 0
// #define n_wide 0
// #define n_squeeze 0
// #define n_mobile 0
// #define n_mnasnet 0
// #define n_inception 0
// #define n_shuffle 0
// #define n_resX 0

#define n_threads 4
#define WARMING 4

// index of flatten or gap
#define DENSE_FLATTEN 1
#define RES_FLATTEN 1   //WideResNet, ResNext
#define ALEX_FLATTEN 5
#define VGG_FLATTEN 5
#define SQUEEZE_FLATTEN 1
#define MOBILE_FLATTEN 1
#define MNAS_GAP 1
#define INCEPTION_FLATTEN 1
#define SHUFFLE_GAP 1
#define EFFICIENT_FLATTEN 1
#define REG_FLATTEN 1

// #define decompose


extern void *predict_alexnet(Net *input);
extern void *predict_vgg(Net *input);
extern void *predict_resnet(Net *input);
extern void *predict_densenet(Net *input);
extern void *predict_squeeze(Net *input);
extern void *predict_mobilenet(Net *input);
extern void *predict_MNASNet(Net *input);
extern void *predict_inception(Net *input);
extern void *predict_shuffle(Net *input);
extern void *predict_efficientnet(Net *input);
extern void *predict_regnet(Net *input);

namespace F = torch::nn::functional;
using namespace std;

threadpool thpool;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;
std::vector<at::cuda::CUDAStream> streams;

c10::DeviceIndex GPU_NUM=0;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, const char* argv[]) {
  GPU_NUM=atoi(argv[1]);
  c10::cuda::set_device(GPU_NUM);
  torch::Device device = {at::kCUDA,GPU_NUM};

  // std::string filename = argv[2];

  int n_dense=atoi(argv[2]);
  int n_res=atoi(argv[3]);
  int n_alex=atoi(argv[4]);
  int n_vgg=atoi(argv[5]);
  int n_wide=atoi(argv[6]);
  int n_squeeze=atoi(argv[7]);
  int n_mobile=atoi(argv[8]);
  int n_mnasnet=atoi(argv[9]);
  int n_inception=atoi(argv[10]);
  int n_shuffle=atoi(argv[11]);
  int n_resX=atoi(argv[12]);
  int n_efficient=atoi(argv[13]);
  int n_reg=atoi(argv[14]);
  
  int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX + n_efficient + n_reg;

  static int stream_index_H = 0;
  static int branch_index_H = 31;


  for(int i=0; i<n_streamPerPool; i++){
    streams.push_back(at::cuda::getStreamFromPool(true,GPU_NUM));
  }

  thpool = thpool_init(n_threads);
  torch::jit::script::Module denseModule;
  torch::jit::script::Module denseModule2;
  torch::jit::script::Module resModule;
  torch::jit::script::Module alexModule;
  torch::jit::script::Module vggModule;
  torch::jit::script::Module DecomposeVggModule;
  torch::jit::script::Module wideModule;
  torch::jit::script::Module squeezeModule;
  torch::jit::script::Module mobileModule;
  torch::jit::script::Module mnasModule;
  torch::jit::script::Module inceptionModule;
  torch::jit::script::Module shuffleModule;
  torch::jit::script::Module resXModule;
  torch::jit::script::Module efficientModule;
  torch::jit::script::Module regModule;
  try {
    	denseModule = torch::jit::load("../densenet201_model.pt");
      denseModule.to(device);

    	resModule = torch::jit::load("../resnet152_model.pt");
      resModule.to(device);

    	alexModule = torch::jit::load("../alexnet_model.pt");
      alexModule.to(device);

    	vggModule = torch::jit::load("../vgg_model.pt");
      vggModule.to(device);

    	wideModule = torch::jit::load("../wideresnet_model.pt");
      wideModule.to(device);

    	squeezeModule = torch::jit::load("../squeeze_model.pt");
      squeezeModule.to(device);

    	mobileModule = torch::jit::load("../mobilenet_model.pt");
      mobileModule.to(device);

    	mnasModule = torch::jit::load("../mnasnet_model.pt");
      mnasModule.to(device);

    	inceptionModule = torch::jit::load("../inception_model.pt");
      inceptionModule.to(device);

    	shuffleModule = torch::jit::load("../shuffle_model.pt");
      shuffleModule.to(device);

    	resXModule = torch::jit::load("../resnext_model.pt");
      resXModule.to(device);

      efficientModule = torch::jit::load("../efficient_b3_model.pt",device);
      efficientModule.to(device);

      regModule = torch::jit::load("../regnet_y_32gf_model.pt");
      regModule.to(device);
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";



  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);


  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }


  vector<torch::jit::IValue> inputs;
  vector<torch::jit::IValue> inputs2;
  vector<torch::jit::IValue> inputs3;

  torch::Tensor x = torch::ones({1, 3, 224, 224}).to(device);
  inputs.push_back(x);

  torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(device);
  inputs3.push_back(x3);

  at::Tensor out;

  if(n_inception){
    torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(device);

    auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
      
    x_ch0.to(device);
    x_ch1.to(device);
    x_ch2.to(device);

    auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(device);
    inputs2.push_back(x_cat);
  }
  
  Net net_input_dense[n_dense];
  Net net_input_res[n_res];
  Net net_input_alex[n_alex];
  Net net_input_vgg[n_vgg];
  Net net_input_wide[n_wide];
  Net net_input_squeeze[n_squeeze];
  Net net_input_mobile[n_mobile];
  Net net_input_mnasnet[n_mnasnet];
  Net net_input_inception[n_inception];
  Net net_input_shuffle[n_shuffle];
  Net net_input_resX[n_resX];
  Net net_input_efficient[n_efficient];
  Net net_input_reg[n_reg];

  pthread_t networkArray_dense[n_dense];
  pthread_t networkArray_res[n_res];
  pthread_t networkArray_alex[n_alex];
  pthread_t networkArray_vgg[n_vgg];
  pthread_t networkArray_wide[n_wide];
  pthread_t networkArray_squeeze[n_squeeze];
  pthread_t networkArray_mobile[n_mobile];
  pthread_t networkArray_mnasnet[n_mnasnet];
  pthread_t networkArray_inception[n_inception];
  pthread_t networkArray_shuffle[n_shuffle];
  pthread_t networkArray_resX[n_resX];
  pthread_t networkArray_efficient[n_efficient];
  pthread_t networkArray_reg[n_reg];

  for(int i=0;i<n_dense;i++){
    //std::cout<<"device num = "<<device.index()<<"\n";
    get_submodule_densenet(denseModule, net_input_dense[i]);
    std::cout << "End get submodule_densenet "<< i << "\n";
    net_input_dense[i].input = inputs;
    net_input_dense[i].name = "DenseNet";
    net_input_dense[i].flatten = net_input_dense[i].layers.size() - DENSE_FLATTEN;
    net_input_dense[i].index_n = i;
    net_input_dense[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    // for(int l=0;l<net_input_dense[i].layers.size();l++){
    //     std::cout<<"index : "<<l<<" , name : "<<net_input_dense[i].layers[l].name<<std::endl;
    // }
    // while(1){}
    for(int j=0;j<WARMING;j++){
      predict_densenet(&net_input_dense[i]);
      net_input_dense[i].input = inputs;
      
    }

    /*=============FILE===============*/
    //net_input_dense[i].fp = fopen((filename+"-"+"D"+".txt").c_str(),"a"); 
  }

  for(int i=0;i<n_res;i++){
	  get_submodule_resnet(resModule, net_input_res[i]);
    std::cout << "End get submodule_resnet "<< i << "\n";
    net_input_res[i].name = "ResNet";
    net_input_res[i].flatten = net_input_res[i].layers.size() - RES_FLATTEN;
	  net_input_res[i].input = inputs;
    net_input_res[i].index_n = i+n_dense;
    net_input_res[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_res[i]);
      net_input_res[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_res[i].fp = fopen((filename+"-"+"R"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_alex;i++){
    net_input_alex[i].index_n = i+ n_res + n_dense;
    get_submodule_alexnet(alexModule, net_input_alex[i]);
    std::cout << "End get submodule_alexnet " << i <<"\n";
	  net_input_alex[i].input = inputs;
    net_input_alex[i].name = "AlexNet";
    net_input_alex[i].flatten = net_input_alex[i].layers.size() - ALEX_FLATTEN;
    net_input_alex[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_alexnet(&net_input_alex[i]);
      net_input_alex[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_alex[i].fp = fopen((filename+"-"+"A"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_vgg;i++){
    #ifdef decompose
      get_submodule_vgg(DecomposeVggModule, net_input_vgg[i]);
    #endif
	  #ifndef decompose
      get_submodule_vgg(vggModule, net_input_vgg[i]);
    #endif
    std::cout << "End get submodule_vgg " << i << "\n";
	  net_input_vgg[i].input = inputs;
    net_input_vgg[i].name = "VGG";
    net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
    net_input_vgg[i].index_n = i + n_alex + n_res + n_dense;
    net_input_vgg[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_vgg(&net_input_vgg[i]);
      net_input_vgg[i].input = inputs;
      
    }

    /*=============FILE===============*/
    //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_wide;i++){
	  get_submodule_resnet(wideModule, net_input_wide[i]);
    std::cout << "End get submodule_widenet "<< i << "\n";
	  net_input_wide[i].input = inputs;
    net_input_wide[i].name = "WideResNet";
    net_input_wide[i].flatten = net_input_wide[i].layers.size() - RES_FLATTEN;
    net_input_wide[i].index_n = i+n_alex + n_res + n_dense + n_vgg;
    net_input_wide[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_wide[i]);
      net_input_wide[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_wide[i].fp = fopen((filename+"-"+"W"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_squeeze;i++){
	  get_submodule_squeeze(squeezeModule, net_input_squeeze[i]);
    std::cout << "End get submodule_squeezenet "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_squeeze[i].record.push_back(event_temp);
    }
    net_input_squeeze[i].name = "SqueezeNet";
    net_input_squeeze[i].flatten = net_input_squeeze[i].layers.size() - SQUEEZE_FLATTEN;
    net_input_squeeze[i].n_all = n_all;
	  net_input_squeeze[i].input = inputs;
    net_input_squeeze[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide;
    net_input_squeeze[i].stream_id = {stream_index_H%n_streamPerPool, abs(branch_index_H)%n_streamPerPool};

    stream_index_H+=1;
    branch_index_H-=1;

    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_squeeze(&net_input_squeeze[i]);
      net_input_squeeze[i].input = inputs;
      for(int n=0;n<net_input_squeeze[i].layers.size();n++){
        net_input_squeeze[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_squeeze[i].fp = fopen((filename+"-"+"SQ"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_mobile;i++){
	  get_submodule_mobilenet(mobileModule, net_input_mobile[i]);
    std::cout << "End get submodule_mobilenet "<< i << "\n";
	  net_input_mobile[i].input = inputs;
    net_input_mobile[i].name = "Mobile";
    net_input_mobile[i].flatten = net_input_mobile[i].layers.size() - MOBILE_FLATTEN;
    net_input_mobile[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze;
    net_input_mobile[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_mobilenet(&net_input_mobile[i]);
      net_input_mobile[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_mobile[i].fp = fopen((filename+"-"+"M"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_mnasnet;i++){
	  get_submodule_MNASNet(mnasModule, net_input_mnasnet[i]);
    std::cout << "End get submodule_mnasnet "<< i << "\n";
	  net_input_mnasnet[i].input = inputs;
    net_input_mnasnet[i].name = "MNASNet";
    net_input_mnasnet[i].gap = net_input_mnasnet[i].layers.size() - MNAS_GAP;
    net_input_mnasnet[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile;
    net_input_mnasnet[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_MNASNet(&net_input_mnasnet[i]);
      net_input_mnasnet[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_mnasnet[i].fp = fopen((filename+"-"+"N"+".txt").c_str(),"a");
  }
  for(int i=0;i<n_inception;i++){
	  get_submodule_inception(inceptionModule, net_input_inception[i]);
    std::cout << "End get submodule_inception "<< i << "\n";
    for(int j=0;j<4;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_inception[i].record.push_back(event_temp);
    }
    net_input_inception[i].n_all = n_all;
	  net_input_inception[i].input = inputs2;
    net_input_inception[i].name = "Inception_v3";
    net_input_inception[i].flatten = net_input_inception[i].layers.size() - INCEPTION_FLATTEN;
    net_input_inception[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet;
    net_input_inception[i].stream_id = {stream_index_H%n_streamPerPool, abs(branch_index_H)%n_streamPerPool, abs(branch_index_H-1)%n_streamPerPool, abs(branch_index_H-2)%n_streamPerPool};

    stream_index_H+=1;
    branch_index_H-=1;

    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_inception(&net_input_inception[i]);
      net_input_inception[i].input = inputs2;
      for(int n=0;n<net_input_inception[i].layers.size();n++){
        net_input_inception[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_inception[i].fp = fopen((filename+"-"+"I"+".txt").c_str(),"a");
  }
  for(int i=0;i<n_shuffle;i++){
	  get_submodule_shuffle(shuffleModule, net_input_shuffle[i]);
    std::cout << "End get submodule_shuffle "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_shuffle[i].record.push_back(event_temp);
    }
    net_input_shuffle[i].n_all = n_all;
	  net_input_shuffle[i].input = inputs;
    net_input_shuffle[i].name = "ShuffleNet";
    net_input_shuffle[i].gap = net_input_shuffle[i].layers.size() - SHUFFLE_GAP;
    net_input_shuffle[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception;
    net_input_shuffle[i].stream_id = {stream_index_H%n_streamPerPool, abs(branch_index_H)%n_streamPerPool};

    stream_index_H+=1;
    branch_index_H-=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_shuffle(&net_input_shuffle[i]);
      net_input_shuffle[i].input = inputs;
      for(int n=0;n<net_input_shuffle[i].layers.size();n++){
        net_input_shuffle[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_shuffle[i].fp = fopen((filename+"-"+"SH"+".txt").c_str(),"a");
  }
  for(int i=0;i<n_resX;i++){
	  get_submodule_resnet(resXModule, net_input_resX[i]);
    std::cout << "End get submodule_resnext "<< i << "\n";
	  net_input_resX[i].input = inputs;
    net_input_resX[i].name = "ResNext";
    net_input_resX[i].flatten = net_input_resX[i].layers.size() - RES_FLATTEN;
    net_input_resX[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle;
    net_input_resX[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_resX[i]);
      net_input_resX[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_resX[i].fp = fopen((filename+"-"+"X"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_efficient;i++){
	  get_submodule_efficientnet(efficientModule, net_input_efficient[i]);
    std::cout << "End get submodule_efficientnet "<< i << "\n";
	  net_input_efficient[i].input = inputs3;
    net_input_efficient[i].name = "EfficientNet";
    net_input_efficient[i].flatten = net_input_efficient[i].layers.size() - EFFICIENT_FLATTEN;
    net_input_efficient[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX;
    net_input_efficient[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_efficientnet(&net_input_efficient[i]);
      net_input_efficient[i].input = inputs3;
      
    }
    /*=============FILE===============*/
    //net_input_efficient[i].fp = fopen((filename+"-"+"E"+".txt").c_str(),"a");
  }

  for(int i=0;i<n_reg;i++){
	  get_submodule_regnet(regModule, net_input_reg[i]);
    std::cout << "End get submodule_regnet "<< i << "\n";
	  net_input_reg[i].input = inputs;
    net_input_reg[i].name = "RegNet";
    net_input_reg[i].flatten = net_input_reg[i].layers.size() - REG_FLATTEN;
    net_input_reg[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX + n_efficient;
    net_input_reg[i].stream_id = {stream_index_H%n_streamPerPool};
    stream_index_H+=1;
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_regnet(&net_input_reg[i]);
      net_input_reg[i].input = inputs;
      
    }
    /*=============FILE===============*/
    //net_input_reg[i].fp = fopen((filename+"-"+"E"+".txt").c_str(),"a");
  }

  std::cout<<"\n==================WARM UP END==================\n";
  cudaDeviceSynchronize();
  
  cudaProfilerStart();

  cudaEvent_t t_start, t_end;
  float t_time;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  
  //double time1 = what_time_is_it_now();
  
  for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &net_input_dense[i]) < 0){
      perror("thread error");
      exit(0);
    }//std::cout<<"dense device num = "<<c10::cuda::current_device()<<"\n";
  }
  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &net_input_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_alex;i++){
    if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_vgg;i++){
	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_wide;i++){
    if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))predict_resnet, &net_input_wide[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_squeeze;i++){
    if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mobile;i++){
    if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mnasnet;i++){
    if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))predict_MNASNet, &net_input_mnasnet[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_inception;i++){
    if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_shuffle;i++){
    if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))predict_shuffle, &net_input_shuffle[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_resX;i++){
    if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))predict_resnet, &net_input_resX[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_efficient;i++){
    if (pthread_create(&networkArray_efficient[i], NULL, (void *(*)(void*))predict_efficientnet, &net_input_efficient[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_reg;i++){
    if (pthread_create(&networkArray_reg[i], NULL, (void *(*)(void*))predict_regnet, &net_input_reg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  
  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL);
  }
  for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }
  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }
  for (int i = 0; i < n_vgg; i++){
    pthread_join(networkArray_vgg[i], NULL);
  }
  for (int i = 0; i < n_wide; i++){
    pthread_join(networkArray_wide[i], NULL);
  }
  for (int i = 0; i < n_squeeze; i++){
    pthread_join(networkArray_squeeze[i], NULL);
  }
  for (int i = 0; i < n_mobile; i++){
    pthread_join(networkArray_mobile[i], NULL);
  }
  for (int i = 0; i < n_mnasnet; i++){
    pthread_join(networkArray_mnasnet[i], NULL);
  }
  for (int i = 0; i < n_inception; i++){
    pthread_join(networkArray_inception[i], NULL);
  }
  for (int i = 0; i < n_shuffle; i++){
    pthread_join(networkArray_shuffle[i], NULL);
  }
  for (int i = 0; i < n_resX; i++){
    pthread_join(networkArray_resX[i], NULL);
  }
  for (int i = 0; i < n_efficient; i++){
    pthread_join(networkArray_efficient[i], NULL);
  }
  for (int i = 0; i < n_reg; i++){
    pthread_join(networkArray_reg[i], NULL);
  }
  //cudaDeviceSynchronize();
  //double time2 = what_time_is_it_now();

  cudaDeviceSynchronize();
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&t_time, t_start, t_end);

	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
  cudaProfilerStop();
  free(cond_t);
  free(mutex_t);
  free(cond_i);

  // for (int i = 0; i < n_dense; i++){
  //   fclose(net_input_dense[i].fp);
  // }
  // for (int i = 0; i < n_res; i++){
  //   fclose(net_input_res[i].fp);
  // }
  // for (int i = 0; i < n_alex; i++){
  //   fclose(net_input_alex[i].fp);
  // }
  // for (int i = 0; i < n_vgg; i++){
  //   fclose(net_input_vgg[i].fp);
  // }
  // for(int i=0;i<n_all;i++){
  //   std::cout<<end_max[i] - start_min[i]<<"\n";
  // }

  // double tmp;
  // double max_value = 0.0;
  
  // for(int i=0;i<n_all;i++){
  //   for(int j=0;j<n_all;j++){
  //     tmp = end_max[i]-start_min[j];
  //     max_value = max(max_value,tmp);
  //   }
  // }
  //  std::cout<<max_value<<"\n";
}
