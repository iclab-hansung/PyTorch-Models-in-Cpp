
#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "vgg.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_vgg(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	for(auto child : module.named_children()){
		if(child.value.children().size()==0){	//avgpool
			t_layer.layer = child.value;
			t_layer.name = "avgpool";
			net.layers.push_back(t_layer);
		}
		else{	//feature , classifier
			for(auto ch : child.value.named_children()){
				if(child.name == "features"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "2" || ch.name == "5" || ch.name == "7" || ch.name == "10" || ch.name == "12" || ch.name == "14" ||
						ch.name == "17" || ch.name == "19" || ch.name == "21" || ch.name == "24" || ch.name == "26" || ch.name == "28"){
							t_layer.name = "conv";
					}
					else if(ch.name == "1" || ch.name == "3" || ch.name == "6" || ch.name == "8" || ch.name == "11" || ch.name == "13" || ch.name == "15" ||
							ch.name == "18" || ch.name == "20" || ch.name == "22" || ch.name == "25" || ch.name == "27" || ch.name == "29"){
								t_layer.name = "relu";
					}
					else if(ch.name == "4" || ch.name == "9" || ch.name == "16" || ch.name == "23" || ch.name == "30"){
						t_layer.name = "maxpool";
					}
					net.layers.push_back(t_layer);
				}
				else if(child.name == "classifier"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" || ch.name == "6"){
						t_layer.name = "linear";
					}
					else if(ch.name == "1" || ch.name == "4"){
						t_layer.name = "relu";
					}
					else if(ch.name == "2" || ch.name == "5" ){	//dropout
						continue;
					}
					net.layers.push_back(t_layer);
				}
			}

		}
	}
}

void *predict_vgg(Net *vgg){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		//double time1 = what_time_is_it_now();
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);

		for(i=0;i<vgg->layers.size();i++){
			pthread_mutex_lock(&mutex_t[vgg->index_n]);
			cond_i[vgg->index_n] = 1;

			netlayer nl;
			nl.net = vgg;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;
			// if(i == 0){
			// 	for(auto param : vgg->layers[i].layer.named_parameters(false)){
			// 		if(param.name == "weight"){
			// 			std::cout<<"name: "<<param.name<<" , param : "<<param.value[100,2,2,63]<<std::endl;
			// 		}
			// 		else	//param.name == "bias"
			// 			std::cout<<"name: "<<param.name<<std::endl;
			// 	}
			// }
		
			thpool_add_work(thpool,(void(*)(void *))forward_vgg,&th);

			while (cond_i[vgg->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[vgg->index_n], &mutex_t[vgg->index_n]);
			}
			i = nl.net->index;
			vgg->input.clear();
			vgg->input.push_back(vgg->layers[i].output);
			pthread_mutex_unlock(&mutex_t[vgg->index_n]);
		}
		cudaStreamSynchronize(streams[vgg->stream_id[0]]);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<vgg->name<<" result  "<<time/1000<<"s ***** \n";
		std::cout << (vgg->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	
	}
}

void forward_vgg(th_arg *th){
	{
		at::cuda::CUDAStreamGuard guard(streams[(th->arg->net->index_n)%(n_streamPerPool)]);
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs = nl->net->input;
		int k = nl->net->index;
		// char str[30];
		// sprintf(str, "VGG layer - %d", k);
		// nvtxRangeId_t id1 = nvtxRangeStartA(str);
		at::Tensor out;
		cudaEvent_t start, end;
		float l_time;
		// cudaEventCreate(&start);
		// cudaEventCreate(&end);
		// cudaEventRecord(start);
		//at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
		// std::cout<<k<<", layer: "<<nl->net->layers[k].name<<std::endl;
		if(k == nl->net->flatten){
			out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
			inputs.clear();
			inputs.push_back(out);
		}
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "relu"){
			nl->net->layers[k].output = out;
			k++;
			inputs.clear();
			inputs.push_back(out);
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
		cudaStreamSynchronize(streams[th->arg->net->stream_id[0]]);
		nl->net->layers[k].output = out;
		nl->net->index = k;
		cond_i[nl->net->index_n]=0;
		// nvtxRangeEnd(id1);
		// cudaEventRecord(end);
		// cudaEventSynchronize(end);
		// cudaEventElapsedTime(&l_time, start, end);
		//fprintf((nl->net->fp),"%d,%lf\n",nl->net->index,l_time/1000);
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
	}
}
