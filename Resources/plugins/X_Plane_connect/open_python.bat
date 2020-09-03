:START
C:\Users\DWKim\Anaconda3\envs\torch\python C:\Users\DWKim\Desktop\X-Plane_11\Resources\plugins\XPlaneConnect-1.3-rc6\Python3\src\ppo_lstm_discrete.py --load_model True --use_bat True
taskkill /f /im X-Plane.exe
timeout 10 > NUL
@GOTO START