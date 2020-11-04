# Communication
import socket

# Environment
import numpy as np
import utm
import math
import time
import os
import codecs

# Reinforcement Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import struct
#############################################################
# 신경망 형태에 관한 부분입니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 신경망의 Input / Output / Hidden Layer
input_state = 12
output_state = 6
start_hidden = 128
second_hidden = 64

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(input_state, start_hidden)
        self.lstm = nn.LSTM(start_hidden, second_hidden)
        self.fc_pi = nn.Linear(second_hidden, output_state)
        self.fc_v = nn.Linear(second_hidden, 1)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, start_hidden)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, start_hidden)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v


#############################################################
# Helper Function
d2r = math.pi / 180
r2d = 180 / math.pi


def eular2dcm(r, p, y):
    c1 = math.cos(d2r * r)
    c2 = math.cos(d2r * p)
    c3 = math.cos(d2r * y)

    s1 = math.sin(d2r * r)
    s2 = math.sin(d2r * p)
    s3 = math.sin(d2r * y)

    # yaw
    yaw_matrix = [
        [c3, 0, s3],
        [0, 1, 0],
        [-s3, 0, c3]
    ]

    # pitch
    pitch_matrix = [
        [1, 0, 0],
        [0, c2, s2],
        [0, -s2, c2]
    ]

    # roll
    roll_matrix = [
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ]

    temp1 = np.matmul(np.array(roll_matrix), np.array(pitch_matrix))
    temp2 = np.matmul(temp1, np.array(yaw_matrix))
    return temp2


def clip(min_val, max_val, val):
    return min(max_val, max(min_val, val))


def get_norm(x, y, z):
    mag = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    if (mag == 0):
        return [0, 0, 0]
    return [x / mag, y / mag, z / mag]


def get_norm_2d(x, y):
    mag = math.sqrt(x ** 2 + y ** 2)
    if (mag == 0):
        return [0, 0, 0]
    return [x / mag, y / mag]


def make_deg_bun_sec(x):
    deg = int(x)
    bun = int((x - deg) * 60)
    sec = int((((x - deg) * 60) - bun) * 1000000 * 60 / 10000) / 100.0
    return [deg, bun, sec]


def make_distance(Ownship_GPS, Enemy_GPS):
    # Ownship_GPS = [lat, lon, alt]
    # Enemy_GPS = [lat, lon, alt]
    diff_lat = Ownship_GPS[0] - Enemy_GPS[0]
    diff_lot = Ownship_GPS[1] - Enemy_GPS[1]
    diff_alt = Ownship_GPS[2] - Enemy_GPS[2]

    lat_dbs = make_deg_bun_sec(diff_lat)
    lot_dbs = make_deg_bun_sec(diff_lot)

    C = math.cos((Ownship_GPS[0] + Enemy_GPS[0]) / 2 * d2r) * 111.3194
    D = 111.3194

    X = lat_dbs[0] * D + lat_dbs[1] * D / 60 + lat_dbs[2] * D / 3600
    Y = lot_dbs[0] * C + lot_dbs[1] * C / 60 + lot_dbs[2] * C / 3600

    place_distance = math.sqrt(X ** 2 + Y ** 2)
    real_distance = math.sqrt(place_distance ** 2 + (diff_alt / 1000) ** 2)
    return real_distance


#############################################################
class Combat():
    def __init__(self):
        self.model = PPO()
        self.model.to(device)

        self.first_time = True
        self.combat_flag = 1

        best_model_path = 'model/ckpt_9553.pth'
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Load Model Success")

        self.fight_or_comback_paramter = 0.7

        # debug part
        self.check_Hz = 0
        self.prev_time = 0
        self.step_cnt = 0
        self.check_input = 0

    # LLA makes Distance, Angle Error, Angle, Height
    def process(self, recv_data):
        # Parsing Data
        Our_craft = recv_data[0:3]
        Npc_craft = recv_data[3:6]

        AoA = recv_data[6]
        SS = recv_data[7]
        Total_Vel = recv_data[8]

        Angle = recv_data[9:12]
        r, p, y = Angle[0] * 180, Angle[1] * 180, Angle[2] * 180
        Stick_cmd = recv_data[12:]

        # make Angle Error from LLA
        ################################################################################
        # X Plane 과 통합 프로그램의 Simulation과의 좌표계가 다를 것이라 생각됩니다..
        # 아래 Coordinate를 드리겠습니다. 일단 확인해보시는걸 추천드립니다.
        #
        # X Plane Coordinate (MAP 기준 right, up, back)
        # x : right, y : up, z : back
        #
        # utm coordinate (MAP 기준 right, front, up)
        # x : right, y : front, z : up
        #
        # A/C coordinate (비행기 기준 front, left, up)
        # x : front, y : left, z : up
        #
        # 위쪽의 eular2dcm 도 한번 확인 부탁드립니다.
        # 목적은 통합 프로그램에서 eular 를 가지고 dcm(Direction Cosine Matrix) 을 만들 수 있으면 됩니다.
        ################################################################################
        our_utm = utm.from_latlon(Our_craft[0], Our_craft[1])
        npc_utm = utm.from_latlon(Npc_craft[0], Npc_craft[1])

        # Coordinate chagne from utm to X Plane
        diff_x = npc_utm[0] - our_utm[0]
        diff_y = Npc_craft[2] - Our_craft[2]
        diff_z = our_utm[1] - npc_utm[1]

        # Coordinate chagne from X Plane to A/C
        output_data = np.matmul(eular2dcm(r, p, y), np.array([diff_x, diff_y, diff_z]))
        Our_to_NPC_vec = [-output_data[2] / 1000, -output_data[0] / 1000, output_data[1] / 1000]

        plane_xy = get_norm_2d(Our_to_NPC_vec[0], Our_to_NPC_vec[1])
        plane_xz = get_norm_2d(Our_to_NPC_vec[0], Our_to_NPC_vec[2] + 0.008)

        if (plane_xz[1] > 0):
            pitch_error = math.acos(plane_xz[0]) / math.pi
        else:
            pitch_error = -math.acos(plane_xz[0]) / math.pi

        if (plane_xy[1] > 0):
            roll_error = -math.acos(plane_xy[0]) / math.pi
        else:
            roll_error = math.acos(plane_xy[0]) / math.pi

        cliped_roll_error = clip(-0.6, 0.6, 6 * roll_error)
        cliped_pitch_error = clip(-0.6, 0.6, 6 * pitch_error)

        # Height 의 경우 X Plane 에서는 5000m에서 시작합니다.
        # 최대한 비슷한 환경에서 진행 부탁드립니다.
        # 어쩔수 없이 다른 곳에서 시작한다면 아래의 식 앞쪽 2500 의 값을 바꾸면 됩니다.
        Height = (Our_craft[2] - 5000) / 2500

        Our_NPC_distance = make_distance(Our_craft, Npc_craft)
        if (Our_NPC_distance > 2):
            Our_NPC_distance = 2

        input_list = [Height, Our_NPC_distance] + [cliped_roll_error, cliped_pitch_error] + \
                     [AoA / (math.pi / 2), SS / (math.pi / 2), Total_Vel] + \
                     Angle[:2] + Stick_cmd

        return input_list

    def check_fight_or_comeback(self, state):
        s = torch.tensor(state, dtype=torch.float).to(device)
        hidden_parameter = (torch.zeros([1, 1, second_hidden], dtype=torch.float, device=device),
                            torch.zeros([1, 1, second_hidden], dtype=torch.float, device=device))
        value = self.model.v(s, hidden_parameter).squeeze(1)
        return value[0].item()
        
    def run(self):
        print("Start RL")
        h_out = (torch.zeros([1, 1, second_hidden], dtype=torch.float, device=device),
                 torch.zeros([1, 1, second_hidden], dtype=torch.float, device=device))

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("Socket Creation Success")

        send_ip = '127.0.0.1'
        send_port = 9090
        send_addr = (send_ip, send_port)

        recv_sock.bind(('localhost', 8080))
        print("Socket Binding Success")

        while True:
            self.step_cnt += 1

            if (self.check_Hz == 1):
                now = time.time()
                print(1 / (now - self.prev_time))
                self.prev_time = now

            ################################################################################
            # Input State 는 꼭 다음 사항에 맞게 주어야 합니다. (순서도 중요합니다.)
            # [     Ownship_Lat, Ownship_Lon, Ownship_Alt, Enemy_Lat, Enemy_Lon, Enemy_Alt, \
            #       Ownship_Angle of Attack, Ownship_Side_Slip, Ownship_Total Vel, \
            #       Ownship_Angle(roll, pitch, yaw), Stick Command(elev, ailrn, trottle)     ]
            #
            # Ownship_Angle_of_Attack [rad]
            # Ownship_Side_Slip [rad]
            # Ownship_Total_Vel [100m/s] ; total vel 값이 3.7 이면 370m/s
            # Ownship_Angle [deg/180] ; Roll Angle 값이 1.1 이면 198deg
            # Stick Command [-1, 1]
            #
            # 1 x 15 Array
            ################################################################################

            recv_data = [0 for i in range(15)]
            received_data, addr = recv_sock.recvfrom(128)

            for i in range(0, 15):
                x = struct.unpack('d', received_data[0 + 8 * i : 8 + 8 * i])[0]
                recv_data[i] = float(x)

            # Preprocessing data 의 형태는 [Height, distance, Angle Error, AoA, SS, Total Vel, Angle, Stick Cmd] 입니다.
            # 1 x 12 Array
            preprocessing_data = self.process(recv_data)
            cur_stick_cmd = preprocessing_data[9:]

            if (self.check_input == 1 and self.step_cnt % 20 == 0):
                os.system('cls')
                print("height, distance : " + str(preprocessing_data[:2]))
                print("Angle Error : " + str(preprocessing_data[2:4]))
                print("AOA / SS / Vel : " + str(preprocessing_data[4:7]))
                print("Our Angle : " + str(preprocessing_data[7:9]))
                print(" ")

            value = self.check_fight_or_comeback(preprocessing_data)

            if (self.first_time == True):
                if (value > self.fight_or_comback_paramter):
                    self.combat_flag = 1
                else:
                    self.combat_flag = 0
                self.first_time = False

            h_in = h_out
            temp_s = torch.from_numpy(np.array(preprocessing_data)).float()
            prob, h_out = self.model.pi(temp_s.to(device), h_in)
            prob = prob.view(-1)
            m = Categorical(prob)
            action = m.sample().item()

            if (action == 0):
                cur_stick_cmd[0] -= 0.02
            elif (action == 1):
                cur_stick_cmd[0] += 0.02
            elif (action == 2):
                cur_stick_cmd[1] -= 0.02
            elif (action == 3):
                cur_stick_cmd[1] += 0.02
            elif (action == 4):
                cur_stick_cmd[2] -= 0.02
            else:
                cur_stick_cmd[2] += 0.02

            cur_stick_cmd[0] = clip(-1.0, 1.0, cur_stick_cmd[0])
            cur_stick_cmd[1] = clip(-1.0, 1.0, cur_stick_cmd[1])
            cur_stick_cmd[2] = clip(0.2, 0.7, cur_stick_cmd[2])

            ################################################################################
            # Output State 는 RL으로 통해 생성된 [Elev, Ailrn, Throttle] 과 combat flag 가 묶여 4개의 list 로 전달됩니다.
            # combat flag 는 현재 상태의 Score 를 판단하여, 0 일 경우 후퇴 / 1 일 경우 교전 유지 입니다.
            # 0 일 경우 귀환에 대한 부분은 아직 짜여지지 않았고, 1세부에서 할 수 있다고 들어 놔뒀습니다.
            ################################################################################
            send_data = cur_stick_cmd + [self.combat_flag, value]

            ba = bytearray(struct.pack("d", send_data[0]))
            for i in range(1, 5):
                ba += bytearray(struct.pack("d", send_data[i]))
            send_sock.sendto(ba, send_addr)

            time.sleep(0.03)

################################################################################
# main function

alg = Combat()
alg.run()
