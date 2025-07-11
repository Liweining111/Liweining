# -*- coding: utf-8 -*-
import argparse
import random
import socket
import time
import math
import os
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 连接参数
key = "liweining_4f8cd43d-57c0-4bd9-b234-1d43d72a3de3:6"
host = 'curling-server-7788.jupyterhub.svc.cluster.local'
port = 7788

# 配置参数
TRAINING_MODE = True  # 设置为True进行训练，False进行对战
GAMES_PER_SAVE = 10   # 每10局保存一次模型

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-p', '--port', help='tcp server port', default="7788", required=False)
parser.add_argument('-host', '--host', help='host', default="192.168.198.208", required=False)
parser.add_argument('--training', help='training mode', action='store_true')
args, _ = parser.parse_known_args()

host = args.host
port = int(args.port)
if args.training:
    TRAINING_MODE = True

print(f"正在尝试连接：{host}:{port}")
print(f"训练模式: {TRAINING_MODE}")

obj = socket.socket()
obj.connect((host, port))
print("连接成功")
obj.send(f"CONNECTKEY:{key}".encode())
msg_recv = obj.recv(1024)
print(msg_recv.decode())

def send_message(sock, message):
    message_with_delimiter = message
    sock.sendall(message_with_delimiter.encode())

def recv_message(sock):
    buffer = bytearray()
    while True:
        data = sock.recv(1)
        if not data or data == b'\0':
            break
        buffer.extend(data)
    return buffer.decode()

retNullTime = 0
while True:
    ret = recv_message(obj)
    messageList = ret.split(" ")
    if ret == "":
        retNullTime = retNullTime + 1
    if retNullTime == 5:
        break
    if messageList[0] == "NAME":
        order = messageList[1]
    if messageList[0] == "ISREADY":
        time.sleep(0.5)
        send_message(obj, "READYOK")
        time.sleep(0.5)
        send_message(obj, "NAME XMUice")
        break

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = torch.tanh(self.fc3(x))
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        return x

# 策略置信度评估器
class StrategyConfidenceEvaluator:
    def __init__(self):
        self.strategy_success_rate = {}
        self.strategy_usage_count = {}
        
    def update_strategy_performance(self, strategy_name, success):
        if strategy_name not in self.strategy_success_rate:
            self.strategy_success_rate[strategy_name] = []
            self.strategy_usage_count[strategy_name] = 0
        
        self.strategy_success_rate[strategy_name].append(success)
        self.strategy_usage_count[strategy_name] += 1
        
        # 保持最近100次记录
        if len(self.strategy_success_rate[strategy_name]) > 100:
            self.strategy_success_rate[strategy_name].pop(0)
    
    def get_strategy_confidence(self, strategy_name):
        if strategy_name not in self.strategy_success_rate:
            return 0.5  # 默认置信度
        
        success_list = self.strategy_success_rate[strategy_name]
        if len(success_list) < 5:
            return 0.5
        
        return sum(success_list) / len(success_list)

# 动态策略选择器
class DynamicStrategySelector:
    def __init__(self):
        self.opponent_patterns = {}
        self.game_phase_weights = {
            'early': {'aggressive': 0.3, 'defensive': 0.7},
            'mid': {'aggressive': 0.5, 'defensive': 0.5},
            'late': {'aggressive': 0.8, 'defensive': 0.2}
        }
    
    def analyze_opponent_pattern(self, game_history):
        """分析对手行为模式"""
        if len(game_history) < 3:
            return 'unknown'
        
        aggressive_moves = sum(1 for move in game_history[-5:] if move.get('velocity', 0) > 0.7)
        if aggressive_moves > 3:
            return 'aggressive'
        elif aggressive_moves < 2:
            return 'defensive'
        else:
            return 'balanced'
    
    def get_game_phase(self, round_num):
        """获取比赛阶段"""
        if round_num <= 4:
            return 'early'
        elif round_num <= 10:
            return 'mid'
        else:
            return 'late'
    
    def recommend_strategy_type(self, round_num, score, opponent_pattern):
        """推荐策略类型"""
        phase = self.get_game_phase(round_num)
        
        if score > 0.25:
            return 'defensive'
        elif score < -0.25:
            return 'aggressive'
        else:
            if opponent_pattern == 'aggressive' and phase == 'late':
                return 'defensive'
            elif opponent_pattern == 'defensive' and phase == 'late':
                return 'aggressive'
            else:
                return 'balanced'

# 模型管理器
class ModelManager:
    def __init__(self):
        self.model_dir = "models"
        self.create_model_dir()
    
    def create_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def get_model_path(self, model_name):
        return os.path.join(self.model_dir, f"{model_name}.pth")
    
    def save_model(self, model, model_name):
        path = self.get_model_path(model_name)
        torch.save(model.state_dict(), path)
        print(f"模型已保存: {path}")
    
    def load_model(self, model, model_name):
        path = self.get_model_path(model_name)
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path))
                print(f"模型已加载: {path}")
                return True
            except Exception as e:
                print(f"加载模型失败: {e}")
                return False
        else:
            print(f"模型文件不存在: {path}")
            return False

class TD3Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, lr_actor, lr_critic, policy_noise, noise_clip, policy_delay, model_manager):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.memory = ReplayMemory()
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=0.001)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr_critic, weight_decay=0.001)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr_critic, weight_decay=0.001)
        self.total_it = 0
        self.ep = 0
        self.epsa = 0.995
        self.lim = 0.8
        self.confidence_evaluator = StrategyConfidenceEvaluator()
        self.strategy_selector = DynamicStrategySelector()
        self.game_history = []
        self.last_strategy_used = None
        self.model_manager = model_manager
        self.game_count = 0
        self.training_mode = TRAINING_MODE

    def select_action(self, state):
        round_num = int(state[32] * 16)
        score = state[33]
        opponent_stones = state[34]
        opponent_pattern = self.strategy_selector.analyze_opponent_pattern(self.game_history)
        recommended_type = self.strategy_selector.recommend_strategy_type(round_num, score, opponent_pattern)
        use_handcrafted = self._should_use_handcrafted_strategy(state, recommended_type)
        
        if use_handcrafted:
            action, strategy_name = self._get_handcrafted_action(state, recommended_type)
            self.last_strategy_used = strategy_name
        else:
            action = self._get_neural_network_action(state)
            self.last_strategy_used = "neural_network"
        
        return action
    
    def _should_use_handcrafted_strategy(self, state, recommended_type):
        round_num = int(state[32] * 16)
        score = state[33]
        
        if self.training_mode:
            if round_num in [15, 14, 13] or abs(score) > 0.5:
                return random.random() < 0.7
            return random.random() < 0.3
        
        if round_num in [15, 14, 13] or abs(score) > 0.5:
            return True
        
        if self.last_strategy_used:
            confidence = self.confidence_evaluator.get_strategy_confidence(self.last_strategy_used)
            if confidence > 0.7:
                return True
        
        if round_num <= 4:
            return random.random() < 0.8
        
        return random.random() < 0.6
    
    def _get_handcrafted_action(self, state, recommended_type):
        round_num = int(state[32] * 16)
        score = state[33]
        z = [2.375, 4.88]
        
        if round_num <= 3:
            return self._get_opening_strategy(state, round_num)
        
        if round_num >= 13:
            return self._get_endgame_strategy(state, round_num, score)
        
        return self._get_midgame_strategy(state, score, recommended_type)
    
    def _get_opening_strategy(self, state, round_num):
        if round_num == 0:
            return np.array([-0.56754, 0, 0]), "opening_first"
        elif round_num == 1:
            xy = [state[0]*4.2996, state[1]*10.4154]
            z = [2.375, 4.88]
            if z[1] <= xy[1] <= z[1]+2.5 and abs(xy[0]-z[0]) <= 1.875:
                return np.array([self.getv1(xy[1])-0.06, (xy[0]-z[0]-0.05)/2, 0]), "opening_guard"
            else:
                return np.array([-0.56854, 0, 0]), "opening_center"
        elif round_num == 2:
            x = state[2]*4.2996
            z = [2.375, 4.88]
            if x <= z[0]:
                return np.array([-0.5464, 0.5, 0.1]), "opening_side_left"
            else:
                return np.array([-0.5464, -0.5, 0.1]), "opening_side_right"
        elif round_num == 3:
            xy = [state[4]*4.2996, state[5]*10.4154]
            z = [2.375, 4.88]
            if z[1] <= xy[1] <= z[1]+2.5 and abs(xy[0]-z[0]) <= 1.875:
                return np.array([self.getv1(xy[1])-0.05, (xy[0]-z[0]-0.05)/2, 0]), "opening_counter"
            else:
                return np.array([-0.55, 0, 0]), "opening_default"
        
        return np.array([-0.55, 0, 0]), "opening_default"
    
    def _get_endgame_strategy(self, state, round_num, score):
        z = [2.375, 4.88]
        
        if round_num == 15:
            if score < 0:
                return self._get_aggressive_final_shot(state), "endgame_aggressive"
            elif score > 0:
                return self._get_defensive_final_shot(state), "endgame_defensive"
            else:
                return self._get_balanced_final_shot(state), "endgame_balanced"
        
        elif round_num in [13, 14]:
            return self._get_setup_strategy(state, score), "endgame_setup"
        
        return np.array([-0.5, 0, 0]), "endgame_default"
    
    def _get_midgame_strategy(self, state, score, recommended_type):
        if recommended_type == 'aggressive':
            return self._get_aggressive_strategy(state), "midgame_aggressive"
        elif recommended_type == 'defensive':
            return self._get_defensive_strategy(state), "midgame_defensive"
        else:
            return self._get_balanced_strategy(state), "midgame_balanced"
    
    def _get_aggressive_strategy(self, state):
        target = self._find_best_hit_target(state)
        if target:
            return np.array([random.uniform(0.8, 1.0), target[0], target[1]])
        else:
            return np.array([0.7, 0, 0])
    
    def _get_defensive_strategy(self, state):
        guard_pos = self._find_best_guard_position(state)
        return np.array([-0.6, guard_pos, 0])
    
    def _get_balanced_strategy(self, state):
        if self._should_attack(state):
            return self._get_aggressive_strategy(state)
        else:
            return self._get_defensive_strategy(state)
    
    def _find_best_hit_target(self, state):
        z = [2.375, 4.88]
        best_target = None
        min_distance = float('inf')
        
        for i in range(0, 32, 2):
            if state[i] > 0:
                x = state[i] * 4.2996
                y = state[i+1] * 10.4154
                distance = self.dist([x, y], z)
                
                if distance < 1.875 and distance < min_distance:
                    min_distance = distance
                    best_target = [(x - z[0])/2, 0]
        
        return best_target
    
    def _find_best_guard_position(self, state):
        return random.choice([-0.5, 0.5])
    
    def _should_attack(self, state):
        score = state[33]
        round_num = int(state[32] * 16)
        
        if score < -0.1:
            return True
        
        if round_num > 10:
            return True
        
        return random.random() < 0.4
    
    def _get_aggressive_final_shot(self, state):
        return np.array([0.9, 0, 0])
    
    def _get_defensive_final_shot(self, state):
        return np.array([-0.7, 0, 0])
    
    def _get_balanced_final_shot(self, state):
        return np.array([-0.5, 0, 0])
    
    def _get_setup_strategy(self, state, score):
        return np.array([-0.6, 0, 0])
    
    def _get_neural_network_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        if self.training_mode:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action += noise
        
        return np.clip(action, -1, 1)
    
    def update_strategy_performance(self, success):
        if self.last_strategy_used:
            self.confidence_evaluator.update_strategy_performance(self.last_strategy_used, success)
    
    def save_models(self):
        self.model_manager.save_model(self.actor, "actor")
        self.model_manager.save_model(self.target_actor, "target_actor")
        self.model_manager.save_model(self.critic1, "critic1")
        self.model_manager.save_model(self.target_critic1, "target_critic1")
        self.model_manager.save_model(self.critic2, "critic2")
        self.model_manager.save_model(self.target_critic2, "target_critic2")
    
    def load_models(self):
        success = True
        success &= self.model_manager.load_model(self.actor, "actor")
        success &= self.model_manager.load_model(self.target_actor, "target_actor")
        success &= self.model_manager.load_model(self.critic1, "critic1")
        success &= self.model_manager.load_model(self.target_critic1, "target_critic1")
        success &= self.model_manager.load_model(self.critic2, "critic2")
        success &= self.model_manager.load_model(self.target_critic2, "target_critic2")
        return success
    
    def on_game_end(self):
        self.game_count += 1
        if self.training_mode and self.game_count % GAMES_PER_SAVE == 0:
            print(f"已完成 {self.game_count} 局游戏，保存模型...")
            self.save_models()
    
    def dist(self, xy1, xy2):
        return sum((a-b)**2 for a, b in zip(xy1, xy2))**0.5
    
    def getv1(self, x):
        p = [-0.1047, 3.486]
        v = p[0]*x + p[1]
        if 2.4 <= v < 2.8:
            return (v - 2.4) / 0.8 - 1
        elif 2.8 <= v < 3.2:
            return (v - 2.8) / 0.4 - 0.5
        elif 3.2 <= v <= 6:
            return (v - 3.2) / 5.6 + 0.5
        return 0
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.total_it += 1
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.FloatTensor(np.array(batch.action))
        rewards = torch.FloatTensor(np.array(batch.reward))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(np.array(batch.done))

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.clamp(torch.randn_like(next_actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            Q_targets_next1 = self.target_critic1(next_states, next_actions)
            Q_targets_next2 = self.target_critic2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
            Q_targets = rewards + (1 - dones) * self.gamma * Q_targets_next

        Q = Q_targets.detach().numpy()
        l = [Q[i][i] for i in range(batch_size)]
        Q_targets = torch.FloatTensor(np.array(l)).unsqueeze(1)

        self.optimizer_critic1.zero_grad()
        Q_expected1 = self.critic1(states, actions)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets.detach())
        critic_loss1.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        Q_expected2 = self.critic2(states, actions)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets.detach())
        critic_loss2.backward()
        self.optimizer_critic2.step()

        if self.total_it % self.policy_delay == 0:
            self.optimizer_actor.zero_grad()
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.optimizer_actor.step()
            self.soft_update(self.actor, self.target_actor, self.tau)
            self.soft_update(self.critic1, self.target_critic1, self.tau)
            self.soft_update(self.critic2, self.target_critic2, self.tau)

        if self.ep > self.lim:
            self.ep *= self.epsa

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# 初始化模型管理器和智能体
model_manager = ModelManager()
noise = 0.1
agent = TD3Agent(state_dim=35, action_dim=3, hidden_dim=256, gamma=0.9, tau=0.0005, lr_actor=0.0005, lr_critic=0.0005,
                 policy_delay=8, policy_noise=noise, noise_clip=2 * noise, model_manager=model_manager)

# 尝试加载模型
if TRAINING_MODE:
    print("训练模式：尝试加载已有模型以继续训练...")
    if not agent.load_models():
        print("没有找到预训练模型，将从随机初始化开始训练")
else:
    print("对战模式：尝试加载模型...")
    if not agent.load_models():
        print("没有找到预训练模型，将使用随机初始化的模型进行对战")

# 初始化游戏状态
state1 = []
state2 = []
action1 = []
batch_size = 64
total_size = 0
startflag1 = False
startflag2 = False
startflag3 = False
shotnum = "0"
order = "Player1"
state = []
shotnum1 = []
houshou = 0
com = 0
retNullTime = 0

def getv(x):
    p = [0.210936613469035, 1.91673185127749]
    v = p[0]*x + p[1]
    v1 = 0
    if 2.4 <= v < 2.8:
        v1 = (v - 2.4) / 0.8 - 1
    elif 2.8 <= v < 3.2:
        v1 = (v - 2.8) / 0.4 - 0.5
    elif 3.2 <= v <= 6:
        v1 = (v - 3.2) / 5.6 + 0.5
    return v1

def getv1(x):
    p = [-0.1047, 3.486]
    v = p[0]*x + p[1]
    v1 = 0
    if 2.4 <= v < 2.8:
        v1 = (v - 2.4) / 0.8 - 1
    elif 2.8 <= v < 3.2:
        v1 = (v - 2.8) / 0.4 - 0.5
    elif 3.2 <= v <= 6:
        v1 = (v - 3.2) / 5.6 + 0.5
    return v1

def getv2(xy):
    p = [-0.08411, 0.1947, 0.003559]
    if xy[0] < 2.375:
        x = 2.375*2 - xy[0]
    else:
        x = xy[0]
    v2 = p[0] + p[1]*x + p[2]*xy[1]
    return v2

def dist(xy1, xy2):
    s = 0
    for (a, b) in zip(xy1, xy2):
        s += (a - b)**2
    return s**0.5

def getp(xy, score, huihe, houshou):
    z = [2.375, 4.88]
    dis = dist(xy, z)
    p = 0
    if not houshou:
        huihe = 0
    if dis <= 0.15:
        p = 1
    elif dis <= 0.91:
        p = 1
    else:
        if score < 0:
            p = -3 * score + huihe
        elif score > 0:
            p = 3 * score
    return p

def getb(xy, z):
    dis = dist(xy, z)
    b = 0.145 * abs(xy[0] - z[0]) / dis
    return b

def getb1(xy, z):
    p1 = [-0.03421, -0.127, 0.1764]
    p2 = [0.02491, -0.1114, -0.1316]
    t = (z[1] - xy[1]) / (z[0] - xy[0])
    if t > 0:
        b = p2[0]*t**2 + p2[1]*t + p2[2]
    else:
        b = p1[0]*t**2 + p1[1]*t + p1[2]
    return b

def getc(xy, xy1, z):
    if xy1[0] == z[0] and xy1[1] == z[1]:
        return False
    if xy[0] != z[0]:
        k = (xy[1] - z[1]) / (xy[0] - z[0])
        y1 = k * (xy1[0] - xy[0]) + xy[1] - 0.29
        y2 = k * (xy1[0] - xy[0]) + xy[1] + 0.29
        if y1 <= xy1[1] <= y2:
            return True
        else:
            return False

def getxy(houshou, state):
    xy = []
    z = [2.375, 4.88]
    r = 1.830
    re = []
    if houshou:
        for i in range(0, 32, 4):
            xy = [state[i] * 4.2996, state[i + 1] * 10.4154]
            dis = dist(xy, z)
            if dis < r:
                r = dis
                re = [state[i] * 4.2996, state[i + 1] * 10.4154]
    else:
        for i in range(2, 32, 4):
            xy = [state[i] * 4.2996, state[i + 1] * 10.4154]
            dis = dist(xy, z)
            if dis < r:
                r = dis
                re = [state[i] * 4.2996, state[i + 1] * 10.4154]
    return re

def huan(action1):
    action = []
    for i in range(0, 32, 2):
        action.append(action1[i] / 4.2996)
        action.append(action1[i + 1] / 10.4154)
    return np.array(action)

def huan1(value, mode):
    v1 = 3
    v2 = 0
    v3 = 0
    if mode == 0:
        if value >= -1 and value < -0.5:
            v1 = 0.8 * (value + 1) + 2.4
        elif value >= -0.5 and value < 0.5:
            v1 = 0.4 * (value + 0.5) + 2.8
        elif value >= 0.5 and value <= 1:
            v1 = 5.6 * (value - 0.5) + 3.2
        return v1
    elif mode == 1:
        v2 = 2 * (value + 1) - 2
        return v2
    elif mode == 2:
        v3 = 3 * (value + 1) - 3
        return v3

def strategy(action):
    lis = []
    lis.append(huan1(action[0], 0))
    lis.append(huan1(action[1], 1))
    lis.append(huan1(action[2], 2))
    text = " ".join(list(map(str, lis)))
    bestshot = "BESTSHOT " + text
    return bestshot

def get_reward(houshou, state):
    xy1 = []
    xy2 = []
    dis1 = []
    dis2 = []
    s = 0
    z = [2.375, 4.88]
    r = 1.830
    num1 = 0
    num2 = 0
    re = []
    for i in range(0, 32, 4):
        xy1.append([state[i], state[i + 1]])
        if state[i] != 0 and state[i + 1] != 0:
            num1 += 1
        xy2.append([state[i + 2], state[i + 3]])
        if state[i + 2] != 0 and state[i + 3] != 0:
            num2 += 1
    for i in range(8):
        dis1.append(dist(xy1[i], z))
        dis2.append(dist(xy2[i], z))
    dis1 = sorted(dis1)
    dis2 = sorted(dis2)
    if dis1[0] > r and dis2[0] > r:
        if houshou:
            re.append(-s / 8)
            re.append(num1 / 8)
            return re
        else:
            re.append(s / 8)
            re.append(num2 / 8)
            return re
    elif dis1[0] < dis2[0]:
        if dis2[0] < r:
            r = dis2[0]
        for i in range(8):
            if dis1[i] < r:
                s += 1
            else:
                break
    else:
        if dis1[0] < r:
            r = dis1[0]
        for i in range(8):
            if dis2[i] < r:
                s -= 1
            else:
                break
    if houshou:
        re.append(-s / 8)
        re.append(num1 / 8)
        return re
    else:
        re.append(s / 8)
        re.append(num2 / 8)
        return re

# 主游戏循环
while True:
    ret = recv_message(obj)
    messageList = ret.split(" ")
    if ret == "":
        retNullTime += 1
    if retNullTime == 5:
        break
    if messageList[0] == "NAME":
        order = messageList[1]
    if messageList[0] == "ISREADY":
        time.sleep(0.5)
        send_message(obj, "READYOK")
        time.sleep(0.5)
        send_message(obj, "NAME XMUice")
    if messageList[0] == "POSITION":
        if state:
            state = []
        state.append(ret.split(" ")[1:33])
        if startflag1:
            s = np.array(list(map(float, state[0])))
            state2.append(s)
            startflag1 = False
    if messageList[0] == "SETSTATE":
        shotnum = ret.split(" ")[1]
        state.append(shotnum)
    if messageList[0] == "GO":
        s = np.array(list(map(float, state[0])))
        state1.append(s)
        startflag1 = True
        state3 = np.append(huan(s), int(state[1]) / 16)
        if int(state[1]) % 2 == 0:
            re = get_reward(False, s)
        else:
            re = get_reward(True, s)
        state3 = np.append(state3, re[0])
        state3 = np.append(state3, re[1])
        action = agent.select_action(state3)
        action1.append(np.array(action))
        shot = strategy(action)
        send_message(obj, shot)
        com = 0
        houshou = int(state[1])
        shotnum1.append(int(state[1]) / 16)
        # 更新策略性能
        success = re[0] > 0
        agent.update_strategy_performance(success)
        # 更新游戏历史
        agent.game_history.append({'velocity': abs(action[0]), 'action': action})
    if messageList[0] == "SCORE" and com != 1:
        reward1 = []
        reward2 = []
        if houshou == 15:
            startflag1 = False
            startflag3 = True
            s = np.array(list(map(float, state[0])))
            state2.append(s)
            state3 = np.array([])
        for i in range(min(7, len(state1))):
            reward1 = get_reward(startflag3, state1[i])
            reward2 = get_reward(startflag3, state2[i])
            state3 = np.append(huan(state1[i]), shotnum1[i])
            state3 = np.append(state3, reward1[0])
            state3 = np.append(state3, reward1[1])
            state4 = np.append(huan(state2[i]), (shotnum1[i] * 16 + 1) / 16)
            state4 = np.append(state4, reward2[0])
            state4 = np.append(state4, reward2[1])
            transition = Transition(state=state3, action=action1[i],
                                  reward=reward2[0],
                                  next_state=state4, done=False)
            agent.memory.push(transition)
        if startflag3 and len(state1) >= 8:
            reward1 = get_reward(startflag3, state1[7])
            reward2 = get_reward(startflag3, state2[7])
            state3 = np.append(huan(state1[7]), shotnum1[7])
            state3 = np.append(state3, reward1[0])
            state3 = np.append(state3, reward1[1])
            state4 = np.append(huan(state2[7]), (shotnum1[7] * 16 + 1) / 16)
            state4 = np.append(state4, reward2[0])
            state4 = np.append(state4, reward2[1])
            transition = Transition(state=state3, action=action1[7],
                                  reward=reward2[0],
                                  next_state=state4, done=True)
            agent.memory.push(transition)
            startflag3 = False
        elif len(state1) >= 8:
            reward1 = get_reward(startflag3, state1[7])
            reward2 = get_reward(startflag3, state2[7])
            state3 = np.append(huan(state1[7]), shotnum1[7])
            state3 = np.append(state3, reward1[0])
            state3 = np.append(state3, reward1[1])
            state4 = np.append(huan(state2[7]), (shotnum1[7] * 16 + 1) / 16)
            state4 = np.append(state4, reward2[0])
            state4 = np.append(state4, reward2[1])
            transition = Transition(state=state3, action=action1[7],
                                  reward=reward2[0],
                                  next_state=state4, done=False)
            agent.memory.push(transition)
        if agent.training_mode:
            agent.update(batch_size=batch_size)
        state1 = []
        state2 = []
        action1 = []
        shotnum1 = []
        total_size += 1
        print(f"已完成游戏局数: {total_size}")
        com = 1
        agent.on_game_end()
    if messageList[0] == "MOTIONINFO":
        x_coordinate = float(messageList[1])
        y_coordinate = float(messageList[2])
        x_velocity = float(messageList[3])
        y_velocity = float(messageList[4])
        angular_velocity = float(messageList[5])
        agent.game_history.append({
            'velocity': math.sqrt(x_velocity**2 + y_velocity**2),
            'action': [x_coordinate, y_coordinate, angular_velocity]
        })

# 关闭连接
obj.close()