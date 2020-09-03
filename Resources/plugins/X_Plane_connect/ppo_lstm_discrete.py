# Env
from time import sleep
import xpc
import time
import numpy as np
import math
import utm
import datetime
import random
import os
import argparse

# Train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--best_model', type=bool, default=False)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--use_bat', type=bool, default=False)
args = parser.parse_args()

################################################################################################
# RL Part
################################################################################################
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epoch       = 20
T_horizon     = 2000000

#X Env
input_state = 12
output_state = 6
start_hidden = 128
second_hidden = 64

device = torch.device("cuda")

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1        = nn.Linear(input_state, start_hidden)
        self.lstm       = nn.LSTM(start_hidden, second_hidden)
        self.fc_pi      = nn.Linear(second_hidden, output_state)
        self.fc_v       = nn.Linear(second_hidden, 1)
        self.optimizer  = optim.Adam(self.parameters(), lr=learning_rate)

        self.epi_loss   = 0

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
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float).to('cuda'), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float).to('cuda'), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r.to('cuda') + gamma * v_prime * done_mask.to('cuda')
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.to('cpu')
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to('cuda')

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a.to('cuda'))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.to('cuda')))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            # td_target = td_target.to('cpu')
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

            self.epi_loss = loss.mean().data


################################################################################################
# Env Part
################################################################################################
d2r = math.pi/180
r2d = 180/math.pi

def eular2dcm(r, p, y):
    c1 = math.cos(d2r*r)
    c2 = math.cos(d2r*p)
    c3 = math.cos(d2r*y)

    s1 = math.sin(d2r*r)
    s2 = math.sin(d2r*p)
    s3 = math.sin(d2r*y)

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
    mag = math.sqrt(x**2 + y**2 + z**2)
    if(mag == 0):
        return [0, 0, 0]
    return [x/mag, y/mag, z/mag]

def get_norm_2d(x, y):
    mag = math.sqrt(x**2 + y**2)
    if(mag == 0):
        return [0, 0, 0]
    return [x/mag, y/mag]

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

    C = math.cos((Ownship_GPS[0] + Enemy_GPS[0])/2 * d2r) * 111.3194
    D = 111.3194
    
    X = lat_dbs[0] * D + lat_dbs[1] * D / 60 + lat_dbs[2] * D / 3600
    Y = lot_dbs[0] * C + lot_dbs[1] * C / 60 + lot_dbs[2] * C / 3600

    place_distance = math.sqrt(X**2 + Y**2)
    real_distance = math.sqrt(place_distance ** 2 + (diff_alt/1000) ** 2)
    return real_distance

class X_Env():
    def __init__(self):
        self.client = xpc.XPlaneConnect()
        self.time = time.time()
        self.start_time = 0

        self.Our_craft = []
        self.Npc_craft = []
        self.ctrl = []

        # for attack
        self.is_attacking = False
        self.attacking_time = 0
        self.start_attack_time = 0
        self.prepare_re_arm_time = 0
        self.prev_toggle = 0
        self.attack_success = False
        self.need_arm = False

        # for NPC random posi
        self.x = 0
        self.y = 0
        self.z = 0

        # for input
        self.prev_Our_position = []
        self.prev_NPC_position = []
        self.prev_Our_orientation = []
        self.prev_position_time = 0
        self.first_input = True
        self.step_cnt = 0

        # for checking step FPS
        self.prev_time = 0

        # for Enemy
        self.enemy_heading = 30
        self.noise1 = 0
        self.noise2 = 0
        self.stability = 0
        self.random_number = random.random()

        # Arming Gun
        for i in range(4):
            command = "sim/weapons/weapon_select_down"
            self.client.sendCOMM(command)
            sleep(1)
        
        command = "sim/weapons/weapon_select_up"
        self.client.sendCOMM(command)

        # input parameter
        self.Our_to_NPC_vec = []
        self.Our_vel_from_body_frame = []
        self.NPC_vel_from_body_frame = []
        self.angle_rate = []
        self.angle = []
        self.Our_NPC_distance = 0
        self.NPC_to_Our_vec = 0

        # for reward
        self.angle_of_attack = 0
        self.side_slip = 0
        self.first_reward = True

        # Debug
        self.check_Hz = 0
        self.Fighting_Mode = 1
        self.Terminal_roll = 170
        self.Terminal_pitch = 80
        self.Terminal_alt = 2500
        self.Enemy_Mode = 3 # 1: Easy, 2: Normal, 3: Hard
        self.use_enemy_heading = False
        self.print_input = False

    def reset(self):
        # for input
        self.first_input = True
        self.prev_Our_position = []
        self.prev_NPC_position = []
        self.prev_Our_orientation = []
        self.prev_position_time = 0
        self.step_cnt = 0

        # for attack
        self.is_attacking = False
        self.attacking_time = 0
        self.start_attack_time = 0
        self.prepare_re_arm_time = 0
        self.prev_toggle = 0
        self.attack_success = False
        self.need_arm = False
        self.toggle_wait_time = 3

        # for reward
        self.angle_of_attack = 0
        self.side_slip = 0
        self.first_reward = True

        # for enemy
        self.enemy_heading = 30
        if(self.Enemy_Mode == 1):
            self.noise1 = (random.random()-0.5) * 60
            self.noise2 = (random.random()-0.5) * 60
        elif(self.Enemy_Mode == 2):
            self.noise1 = (random.random()-0.5) * 200
            self.noise2 = (random.random()-0.5) * 200
        self.prev_enemy_pitch = 0
        self.prev_enemy_roll = 0
        self.prev_enemy_yaw = 0
        self.stability = 0
        self.random_number = random.random()
        
        # re-generate opponent aircraft
        command = "sim/operation/load_situation_1"
        self.client.sendCOMM(command)

        # wait
        sleep(7)

        # current time
        self.time = time.time()
        
        # view change
        self.client.sendVIEW(xpc.ViewType.Chase)

        # set time
        self.client.sendDREF("sim/time/zulu_time_sec", 3000.0)

        try:
            self.client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # Set position of the player aircraft
        #       Lat     Lon         Alt   Pitch Roll Yaw Gear
        posi = [37.524, -122.06899, 5000, 0,    0,  0,  0]
        self.client.sendPOSI(posi)

        # Set position of a non-player aircraft
        #       Lat       Lon      Alt   Pitch Roll Yaw Gear
        if(self.Enemy_Mode < 4):
            self.x = ( random.random() - 0.5 ) * 0.01     # -0.005 ~ 0.005
            self.y = ( random.random() - 0.5 ) * 0.01     # -0.005 ~ 0.005
            posi = [37.55 + self.x, -122.06899 + self.y, 5000, 0,    0,   0,  0]
        else:
            self.x = ( random.random() - 0.5 ) * 0.05     # -0.025 ~ 0.025
            self.y = ( random.random() - 0.5 ) * 0.05     # -0.025 ~ 0.025
            self.z = (random.random() - 0.5) * 500       # -250 ~ 250
            posi = [37.55 + self.x, -122.06899 + self.y, 5000 + self.z, 0,    0,   0,  0]
        self.client.sendPOSI(posi, 1)

        # re-arm weapon
        command = "sim/weapons/re_arm_aircraft"
        self.client.sendCOMM(command)

        # Set angle of attack, velocity, and orientation using the DATA command
        data = [
            # angle of attack, sideslip, &path
            # number / angle of attack / angle of sideslip
            [18,   0, -998,   0, -998, -998, -998, -998, -998],

            # Speed
            # number / indicated airspeed / equivalent airspeed / true airspeed / ground speed
            [ 3, 130,  130, 130,  130, -998, -998, -998, -998],

            # angular velocity
            # number / pitch rate / roll rate / yaw rat
            [16,   0,    0,   0, -998, -998, -998, -998, -998],

            # linear velocity
            # number / x / y / z / vx / vy / vz
            # [21, -1000.6, 6000.5, -15000.8, -100, 0, 140, 3580.0, 0.5893]
            [21, -1000.6, 6000.5, -15000.8,   0,  0,  -100, 3580.0, 0.5893]
        ]
        self.client.sendDATA(data)

        ###################################
        # Set control surfaces and throttle of the player aircraft using sendCTRL
        # Latitudinal Stick [-1,1]
        # Longitudinal Stick [-1,1]
        # Rudder Pedals [-1, 1]
        # Throttle [-1/4, 1]
        # Gear (0=up, 1=down)
        # Flaps [0, 1]

        ctrl = [0.0, 0.0, 0.0, 0.7, 0, 0]
        self.client.sendCTRL(ctrl)

        ctrl = [0.0, 0.0, 0.0, 0.7, 0, 0]
        self.client.sendCTRL(ctrl, 1)
        ##################################

        self.ctrl = list(self.client.getCTRL())

        # Stow landing gear using a dataref
        gear_dref = "sim/cockpit/switches/gear_handle_status"
        self.client.sendDREF(gear_dref, 0)

        while(time.time() - self.time < 2):
            self.client.sendDREF("sim/multiplayer/position/plane1_the", 0)
            self.client.sendDREF("sim/multiplayer/position/plane1_phi", 0)
            self.client.sendDREF("sim/multiplayer/position/plane1_psi", 0)

            self.client.sendDREF("sim/multiplayer/position/plane1_v_z", -100)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_x", 0)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_y", 0)
            sleep(0.5)

        self.set_state()
        self.update_parameter()

        self.start_time = time.time()
        return self.get_input()

    
    def set_state(self):
        self.Our_craft = list(self.client.getPOSI()[:6])
        self.Npc_craft = list(self.client.getPOSI(1)[:6])

    def get_state(self):
        return self.Our_craft + self.Npc_craft 

    def update_parameter(self):
        Our_craft = self.Our_craft
        npc_craft = self.Npc_craft

        if (Our_craft[5] > 180):
            Our_craft[5] = -360 + Our_craft[5]

        our_utm = utm.from_latlon(Our_craft[0], Our_craft[1])
        npc_utm = utm.from_latlon(npc_craft[0], npc_craft[1])

        diff_x = npc_utm[0] - our_utm[0]
        diff_y = npc_craft[2] - Our_craft[2]
        diff_z = our_utm[1] - npc_utm[1]

        p, r, y = Our_craft[3], Our_craft[4], Our_craft[5]
        npc_p, npc_r, npc_y = 0, 0, npc_craft[5]
        now = time.time()

        if(self.first_input):
            vx, vy, vz = 0, 0, 0
            pitch_rate, roll_rate, yaw_rate = 0, 0, 0
            n_vx, n_vy, n_vz = 0, 0, 0

            self.first_input = False
            self.prev_Our_position = [our_utm[0], our_utm[1], Our_craft[2]]
            self.prev_NPC_position = [npc_utm[0], npc_utm[1], npc_craft[2]]
            self.prev_Our_orientation = [p, r, y]
            self.prev_position_time = now
        else:
            vx = (our_utm[0] - self.prev_Our_position[0])/(now - self.prev_position_time)
            vy = (Our_craft[2] - self.prev_Our_position[2])/(now - self.prev_position_time)
            vz = (self.prev_Our_position[1] - our_utm[1])/(now - self.prev_position_time)

            if(p - self.prev_Our_orientation[0] < -180):
                pitch_rate = (360 + p - self.prev_Our_orientation[0])/(now - self.prev_position_time)
            elif(p - self.prev_Our_orientation[0] > 180):
                pitch_rate = (p - self.prev_Our_orientation[0] - 360)/(now - self.prev_position_time)
            else:
                pitch_rate = (p - self.prev_Our_orientation[0])/(now - self.prev_position_time)

            if(r - self.prev_Our_orientation[1] < -180):
                roll_rate = (360 + r - self.prev_Our_orientation[1])/(now - self.prev_position_time)
            elif(r - self.prev_Our_orientation[1] > 180):
                roll_rate = (r - self.prev_Our_orientation[1] - 360)/(now - self.prev_position_time)
            else:
                roll_rate = (r - self.prev_Our_orientation[1])/(now - self.prev_position_time)

            if(y - self.prev_Our_orientation[2] < -180):
                yaw_rate = (360 + y - self.prev_Our_orientation[2])/(now - self.prev_position_time)
            elif(y - self.prev_Our_orientation[2] > 180):
                yaw_rate = (y - self.prev_Our_orientation[2] - 360)/(now - self.prev_position_time)
            else:
                yaw_rate = (y - self.prev_Our_orientation[2])/(now - self.prev_position_time)

            n_vx = (npc_utm[0] - self.prev_NPC_position[0])/(now - self.prev_position_time)
            n_vy = (npc_craft[2] - self.prev_NPC_position[2])/(now - self.prev_position_time)
            n_vz = (self.prev_NPC_position[1] - npc_utm[1])/(now - self.prev_position_time)

            self.prev_Our_position = [our_utm[0], our_utm[1], Our_craft[2]]
            self.prev_NPC_position = [npc_utm[0], npc_utm[1], npc_craft[2]]
            self.prev_Our_orientation = [p, r, y]
            self.prev_position_time = now

        gun_time = 2
        diff_x += (n_vx - vx) * gun_time
        diff_y += (n_vy - vy) * gun_time
        diff_z += (n_vz - vz) * gun_time

        output_data = np.matmul(eular2dcm(r, p, y), np.array([diff_x, diff_y, diff_z]))
        self.Our_to_NPC_vec = [-output_data[2]/1000, -output_data[0]/1000, output_data[1]/1000]

        output_data = np.matmul(eular2dcm(npc_r, npc_p, npc_y), np.array([-diff_x, -diff_y, -diff_z]))
        self.NPC_to_Our_vec = [-output_data[2]/1000, -output_data[0]/1000, output_data[1]/1000]

        temp_Our_vel_from_body_frame = np.matmul(eular2dcm(r, p, y), np.array([vx, vy, vz]))
        self.Our_vel_from_body_frame = [-temp_Our_vel_from_body_frame[2]/100, -temp_Our_vel_from_body_frame[0]/100, temp_Our_vel_from_body_frame[1]/100]

        temp_NPC_vel_from_body_frame = np.matmul(eular2dcm(r, p, y), np.array([n_vx, n_vy, n_vz]))
        self.NPC_vel_from_body_frame = [-temp_NPC_vel_from_body_frame[2]/100, -temp_NPC_vel_from_body_frame[0]/100, temp_NPC_vel_from_body_frame[1]/100]

        Our_craft = list(Our_craft)

        vel_xy = get_norm_2d(self.Our_vel_from_body_frame[0], self.Our_vel_from_body_frame[1])
        vel_xz = get_norm_2d(self.Our_vel_from_body_frame[0], self.Our_vel_from_body_frame[2])

        self.angle_of_attack = math.acos(vel_xz[0])
        if(vel_xz[1] > 0):
            self.angle_of_attack = -self.angle_of_attack

        self.side_slip = math.acos(vel_xy[0])
        if(vel_xy[1] < 0):
            self.side_slip = -self.side_slip

        self.Height = (self.Our_craft[2] - 2500) / 2500
        self.Our_NPC_distance = make_distance(Our_craft[:3], npc_craft[:3])
        self.angle = [Our_craft[4]/180, Our_craft[3]/180, Our_craft[5]/180]
        self.angle_rate = [roll_rate/180, pitch_rate/180, yaw_rate/180]


    ##########################################
    # simulation coordinate in map
    # x : right, y : up, z : back
    #
    # utm coordinate
    # x : right, y : front, z : up
    #
    # A/C coordinate
    # x : front, y : left, z : up
    def get_input(self):
        plane_xy = get_norm_2d(self.Our_to_NPC_vec[0], self.Our_to_NPC_vec[1])
        plane_xz = get_norm_2d(self.Our_to_NPC_vec[0], self.Our_to_NPC_vec[2] + 0.008)

        if (plane_xz[1] > 0):
            pitch_error = math.acos(plane_xz[0]) / math.pi
        else:
            pitch_error = -math.acos(plane_xz[0]) / math.pi

        if (plane_xy[1] > 0):
            roll_error = -math.acos(plane_xy[0]) / math.pi
        else:
            roll_error = math.acos(plane_xy[0]) / math.pi

        total_vel = math.sqrt(self.Our_vel_from_body_frame[0]**2 + self.Our_vel_from_body_frame[1]**2 + self.Our_vel_from_body_frame[2]**2)
        
        input_distance = self.Our_NPC_distance
        if (input_distance > 2):
            input_distance = 2

        cliped_roll_error = clip(-0.6, 0.6, 6 * roll_error)
        cliped_pitch_error = clip(-0.6, 0.6, 6 * pitch_error)

        input_list =    [self.Height, input_distance] + [cliped_roll_error, cliped_pitch_error] + \
                        [self.angle_of_attack / (math.pi/2), self.side_slip / (math.pi/2), total_vel] + \
                        self.angle[:2] + self.ctrl[0:2] + [self.ctrl[3]]
        
        self.step_cnt += 1
        if(self.print_input and self.step_cnt % 20 == 0):
            # os.system('cls') 
            print("height, distance : " + str(input_list[:2]))
            print("Angle Error : " + str(input_list[2:4]))
            print("AOA / SS / Vel : " + str(input_list[4:7]))
            print("Our Angle : " + str(input_list[7:9]))
            print(" ")

        return input_list

    def not_attack(self):
        command = "sim/weapons/fire_guns_off"
        if(self.Fighting_Mode == 1):
            print("Try Not Attack")
            self.attacking_time += time.time() - self.start_attack_time
            self.prev_toggle = time.time()
            self.client.sendCOMM(command)
            self.is_attacking = False

    def attack(self):
        command = "sim/weapons/fire_guns_on"
        if(self.Fighting_Mode == 1):
            print("Try Attack")
            self.start_attack_time = time.time()
            self.prev_toggle = time.time()
            self.client.sendCOMM(command)
            self.is_attacking = True

    def get_reward(self):
        reward = 0

        state = self.get_state()

        norm_Our_to_NPC_vec = get_norm(self.Our_to_NPC_vec[0], self.Our_to_NPC_vec[1], self.Our_to_NPC_vec[2])
        norm_NPC_to_Our_vec = get_norm(self.NPC_to_Our_vec[0], self.NPC_to_Our_vec[1], self.NPC_to_Our_vec[2])

        Offense_Angle = math.acos(norm_Our_to_NPC_vec[0])
        Defense_Angle = math.acos(norm_NPC_to_Our_vec[0])

        # Consider Enemy Angle [-1, 1]
        # Score = (math.cos(Offense_Angle) - math.cos(Defense_Angle))/2

        # Only Ownship Angle
        if(math.acos(norm_Our_to_NPC_vec[0]) < 0.7):
            reward += math.exp(0.3 - math.acos(norm_Our_to_NPC_vec[0]))
        # reward += 0.3 - math.acos(norm_Our_to_NPC_vec[0])

        # Consider Gun
        if (self.Our_NPC_distance < 1 and self.Our_NPC_distance > 0.2):
            reward += 1 - self.Our_NPC_distance
            
        # Consider Corner Speed
        total_vel = math.sqrt(self.Our_vel_from_body_frame[0]**2 + self.Our_vel_from_body_frame[1]**2 + self.Our_vel_from_body_frame[2]**2)
        if (total_vel > 1.5 and self.Our_NPC_distance < 1):
            reward -= 0.5

        # NOT stable
        # if (state[2] < self.Terminal_alt):
        #     reward -= 2000

        # if (state[3] > self.Terminal_pitch or state[3] < -self.Terminal_pitch or state[4] > self.Terminal_roll or state[4] < -self.Terminal_roll):
        #     reward -= 5000
        # elif ((self.angle_of_attack > 55 * d2r or self.angle_of_attack < -10 * d2r) and (time.time() - self.start_time > 5)):
        #     reward -= 5000
        # elif ((abs(self.side_slip) > 30 * d2r) and (time.time() - self.start_time > 5)):
        #     reward -= 5000
        # elif (state[2] < self.Terminal_alt):
        #     reward -= 5000

        if(self.first_reward == True):
            self.prev_enemy_pitch = self.Npc_craft[3]
            self.prev_enemy_roll = self.Npc_craft[4]
            self.prev_enemy_yaw = self.Npc_craft[5]
            self.first_reward = False
        else:
            diff_pitch = abs(self.prev_enemy_pitch - self.Npc_craft[3])
            diff_roll = abs(self.prev_enemy_roll - self.Npc_craft[4])
            diff_yaw = abs(self.prev_enemy_yaw - self.Npc_craft[5])

            if(diff_pitch > 180):
                diff_pitch = 360 - diff_pitch
            if(diff_roll > 180):
                diff_roll = 360 - diff_roll
            if(diff_yaw > 180):
                diff_yaw = 360 - diff_yaw

            # print(diff_pitch + diff_roll + diff_yaw)

            if(diff_pitch + diff_roll + diff_yaw > 5 and time.time() - self.start_time > 10 and self.start_attack_time != 0 and self.stability == 1):
                reward += 10000
                self.attack_success = True

            if(diff_pitch + diff_roll + diff_yaw < 1):
                self.stability = 1

            self.prev_enemy_pitch = self.Npc_craft[3]
            self.prev_enemy_roll = self.Npc_craft[4]
            self.prev_enemy_yaw = self.Npc_craft[5]

        return reward

    def is_gameover(self):
        state = self.get_state()
        Our_alt = state[2]
        NPC_alt = state[8]
        NPC_pose = state[6:]

        # time constrain
        time_constrain = 100
        if (time.time() - self.time > time_constrain):
            print("Time done")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True

        # Consider Angle
        if (state[3] > self.Terminal_pitch or state[3] < -self.Terminal_pitch or state[4] > self.Terminal_roll or state[4] < -self.Terminal_roll):
            print("Angle done")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True

        # Consider Angle of Attack
        if ((self.angle_of_attack > 55 * d2r or self.angle_of_attack < -10 * d2r) and (time.time() - self.start_time > 5)):
            print("Angle of Attack")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True

        # Consider Side slip
        if ((abs(self.side_slip) > 30 * d2r) and (time.time() - self.start_time > 5)):
            print("Side Slip")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True

        # Consider Alt
        if (Our_alt < self.Terminal_alt):
            print("Our Alt done")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True

        # Hit Weapon
        if(self.attack_success == True):
            print("               Attack Success!!!!!")

            if(self.is_attacking == True):
                sleep(2)
                self.not_attack()

            self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
            return True
        return False

        # Arm Weapon
        # if(self.need_arm == True):
        #     print("Need Arm")

        #     if(self.is_attacking == True):
        #         sleep(2)
        #         self.not_attack()

        #     self.client.sendCTRL([0.0, 0.0, 0.0, 0.0])
        #     return True
        # return False


    def step(self, action):
        if(self.check_Hz == 1):
            now = time.time()
            print(1/(now-self.prev_time))
            self.prev_time = now 

        if (action == 0):
            self.ctrl[0] -= 0.02
        elif (action == 1):
            self.ctrl[0] += 0.02
        elif (action == 2):
            self.ctrl[1] -= 0.02
        elif (action == 3):
            self.ctrl[1] += 0.02
        elif (action == 4):
            self.ctrl[3] -= 0.02
        elif (action == 5):
            self.ctrl[3] += 0.02

        self.ctrl[3] = clip(0.2, 0.7, self.ctrl[3])

        # send control
        # if(self.Our_NPC_distance > 1):
        #     self.client.sendCTRL([self.ctrl[0], self.ctrl[1], 0, 0.8])
        # else:
        #     self.client.sendCTRL([self.ctrl[0], self.ctrl[1], 0, self.ctrl[3]])
        self.client.sendCTRL([self.ctrl[0], self.ctrl[1], 0, self.ctrl[3]])

        ###################################################
        if (self.Fighting_Mode == 1):
            # Fighting Mode
            norm_Our_to_NPC_vec = get_norm(self.Our_to_NPC_vec[0], self.Our_to_NPC_vec[1], self.Our_to_NPC_vec[2])
            Our_to_NPC_angle = math.acos(norm_Our_to_NPC_vec[0])

            if(Our_to_NPC_angle < 0.8 and self.Our_NPC_distance < 0.8 and self.is_attacking == False and time.time() - self.prev_toggle > self.toggle_wait_time and self.need_arm == False):
                self.attack()
            if((Our_to_NPC_angle > 1 or self.Our_NPC_distance > 1) and self.is_attacking == True and time.time() - self.prev_toggle > self.toggle_wait_time and self.need_arm == False):
                self.not_attack()
            
            # re-arm
            if(self.attacking_time > 18 or (self.is_attacking and self.attacking_time + (time.time() - self.start_attack_time) > 15)):
                if(self.is_attacking and time.time() - self.prev_toggle > self.toggle_wait_time):
                    self.not_attack()
                    self.need_arm = True
                elif(self.is_attacking == False):
                    self.need_arm = True

                if(time.time() - self.prev_toggle > self.toggle_wait_time and self.need_arm == True):
                    command = "sim/weapons/re_arm_aircraft"
                    self.client.sendCOMM(command)
                    sleep(0.5)

                    print("Arm Complete")
                    self.attacking_time = 0
                    self.prev_toggle = time.time()
                    self.need_arm = False
        ###################################################

        sleep(0.001)

        self.set_state()
        self.update_parameter()
        return self.get_input(), self.get_reward(), self.is_gameover(), None

    def get_npc_action_by_vel(self):
        if (self.Enemy_Mode == 1):  # Easy Mode
            self.client.sendDREF("sim/multiplayer/position/plane1_v_z", -100 + self.noise1)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_x", self.noise2)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_y", 0)
        elif (self.Enemy_Mode == 2): # Normal Mode
            self.client.sendDREF("sim/multiplayer/position/plane1_v_z", self.noise1)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_x", self.noise2)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_y", 0)
        else: # Hard Mode
            norm_NPC_to_Our_vec_x = self.NPC_to_Our_vec[0]/math.sqrt(self.NPC_to_Our_vec[0] ** 2 + self.NPC_to_Our_vec[1]**2)
            norm_NPC_to_Our_vec_y = self.NPC_to_Our_vec[1]/math.sqrt(self.NPC_to_Our_vec[0] ** 2 + self.NPC_to_Our_vec[1]**2)

            if(self.use_enemy_heading == False):
                noise = random.random() * 0.3

                # if(self.random_number > 0.5):
                #     noise = -noise

                if(norm_NPC_to_Our_vec_y > 0.2):
                    self.enemy_heading -= noise
                elif(norm_NPC_to_Our_vec_y < -0.2):
                    self.enemy_heading += noise
                elif (norm_NPC_to_Our_vec_x < 0):
                    self.enemy_heading += noise

                # print(self.enemy_heading)

                z_v = -100 * math.cos(d2r * self.enemy_heading)
                x_v = 100 * math.sin(d2r * self.enemy_heading)
            else:
                npc_craft = self.Npc_craft
                NPC_y = npc_craft[5]
                heading_vec = NPC_y

                if (norm_NPC_to_Our_vec_y > 0.2):
                    heading_vec = -2 + NPC_y
                elif (norm_NPC_to_Our_vec_y < -0.2):
                    heading_vec = 2 + NPC_y
                elif (norm_NPC_to_Our_vec_x < 0):
                    heading_vec = 2 + NPC_y

                z_v = -100 * math.cos(d2r * heading_vec)
                x_v = 100 * math.sin(d2r * heading_vec)

            self.client.sendDREF("sim/multiplayer/position/plane1_v_z", z_v)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_x", x_v)
            self.client.sendDREF("sim/multiplayer/position/plane1_v_y", 0)


def main():
    if(args.use_bat == True):
        sleep(100)

    env = X_Env()
    model = PPO()
    model = model.to('cuda')
    score = 0.0
    epi_reward = 0.0
    print_interval = 10
    n_epi = 0

    # PATH = 'model/supervised_learning.pth'
    # model.load_state_dict(torch.load(PATH))

    if (args.best_model == True):
        path = 'model/2020_08_19_00_35_save_model_first_hidden_128_second_hidden_64_input_12_output_6/ckpt_9553.pth'
        print(path)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    elif (args.load_model == True):
        path = "./model/"
        file_list = os.listdir(path)
        last_folder = file_list[-2]
        
        path = "./model/" + last_folder
        file_list = os.listdir(path)
        max_index = 0
        max_value = int(file_list[0][5:-4])
        for idx in range(len(file_list)):
            if(max_value < int(file_list[idx][5:-4])):
                max_index = idx
                max_value = int(file_list[idx][5:-4])
        last_file = file_list[max_index]

        load_PATH = "model/" + last_folder + "/" + last_file
        print(load_PATH)

        checkpoint = torch.load(load_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    year = datetime.datetime.today().year

    month = datetime.datetime.today().month
    if(len(str(month)) == 1):   
        month = "0" + str(month)

    day = datetime.datetime.today().day
    if(len(str(day)) == 1):
        day = "0" + str(day)

    hour = datetime.datetime.today().hour
    if(len(str(hour)) == 1):
        hour = "0" + str(hour)

    minute = datetime.datetime.today().minute
    if(len(str(minute)) == 1):
        minute = "0" + str(minute)

    Save_path = "model/" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_save_model_" + \
                "first_hidden_" + str(start_hidden) + "_second_hidden_" + str(second_hidden) + "_input_" + str(input_state) + "_output_" + str(output_state)

    writer = SummaryWriter("tensorboard")
    
    while(True):
        n_epi += 1
        h_out = (torch.zeros([1, 1, second_hidden], dtype=torch.float, device = device), torch.zeros([1, 1, second_hidden], dtype=torch.float, device = device))
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                if(done):
                    s = env.reset()

                h_in = h_out
                temp_s = torch.from_numpy(np.array(s)).float()
                prob, h_out = model.pi(temp_s.to('cuda'), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()

                # print(a)

                s_prime, r, done, info = env.step(a)

                ########################
                # npc action
                env.get_npc_action_by_vel()
                ########################

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                epi_reward += r
                if done:
                    break

            model.train_net()

        writer.add_scalar("Reward function", epi_reward, n_epi)
        writer.add_scalar("Loss", model.epi_loss, n_epi)
        print("Reward : ", epi_reward)
        epi_reward = 0

        if n_epi%print_interval==0 and n_epi!=0:
            model_path = os.path.join(os.getcwd(), Save_path)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            avg_score_str = str(int(score/print_interval))
            ckpt_path = os.path.join(model_path, 'ckpt_'+ avg_score_str + '.pth')

            torch.save(
                {
                'episode' : n_epi,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : model.optimizer.state_dict(),
                }, ckpt_path)

            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

if __name__ == "__main__":
    main()