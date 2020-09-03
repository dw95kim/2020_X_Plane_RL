# 2020_X_Plane_RL

## Env
```
python 3.8.3
Nvidia Driver 436.30
Cuda 10.0
Cudnn 7.6.5
Pytorch 1.5.1
X Plane 11.41
```

## 실행을 위한 환경 설정
```
1. Aircraft 의 F-16 기체를 X Plane 11 폴더의 Aircraft 에 추가한다.

2. Situation 의 폴더를 X Plane 11 폴더의 Output/Situation 폴더에 추가한다.

3. Resource/plugins 의 64를 X Plane 11 폴더의 Resource/Plugins 에 추가한다.

4. Resource/plugins 의 X_Plane_connect 가 강화학습과 통신에 관한 코드이다.
```
## 실행 방법
```
무한정 실행해야 하므로 bat 파일을 이용해 시뮬레이션이 꺼지더라도 계속 실행할 수 있도록 설정

1. 두개의 terminal을 킨다. (A와 B의 터미널이라고 하자)

2. A 터미널에서는 call open_x_plane.bat 을 실행

3. 동시에 B 터미널에서 call open_python.bat 을 실행
```

## RL code
```
argparse에 --best_model , --load_model, --use_bat 이 있다.

1. --best_model을 True 로 설정할 경우 지금까지의 최고의 결과 model 을 불러온다. (수동으로 가장 좋은 model path 설정)

2. --load_model을 True 로 설정할 경우 가장 최근 실행된 결과 중 가장 좋은 model을 불러온다. (가장 최근 폴더만 검사)

3. --use_bat을 True 로 설정할 경우 Simulation 시간을 고려하여 시작시 100초간 sleep 후 동작한다.
```
