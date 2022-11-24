# PyTorch-Models-in-Cpp

#### 전체 코드 실행을 위한 실행파일 생성과정
1. CMakeLists.txt가 들어있는 디렉토리에 build 디렉토리 생성
2. build 디렉토리 내에서 cmake.. 명령어 실행
3. 2번 과정 완료 후 make 명령어 실행
4. build 디렉토리 내에 실행파일 생성 완료

#### model_pt 디렉토리를 생성하여 각 모델의 pt파일을 넣어줘야 정상적인 실행 가능
#### - 각 DNN 모델의 version  (model pt파일의 파일 명) 
- alexnet       (alexnet_model.pt)
- densenet201   (densenet201_model.pt)
- efficient_b3 (efficient_b3_model.pt)
- inception_v3  (inception_model.pt)
- mnasnet1_0    (mnasnet_model.pt)
- mobilenet_v2  (mobile_model.pt)
- regnet_y_32gf (regnet_y_32gf_model.pt)
- resnet152     (resnet_model.pt)
- resnext50     (resnext_model.pt)
- shufflenet_v2_1_0   (shuffle_model.pt)
- squeezenet1_0       (squeeze_model.pt)
- vgg16         (vgg_model.pt)
- wideresnet50  (wideresnet_model.pt)
