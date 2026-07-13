import csv
import os
import pickle
import random
import sys
import time
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

pygame.init()

# ===== CONFIG =====
BOARD_WIDTH = BOARD_HEIGHT = 500
PANEL_WIDTH = 380
WINDOW_WIDTH = BOARD_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT
CELL = 20
COLS = BOARD_WIDTH // CELL
ROWS = BOARD_HEIGHT // CELL
RENDER_FPS = 60
NUM_FOODS = 2

RED_SAVE_DIR = "models_red_dqn"
BLUE_SAVE_DIR = "models_blue_qtable"
os.makedirs(RED_SAVE_DIR, exist_ok=True)
os.makedirs(BLUE_SAVE_DIR, exist_ok=True)
RED_MODEL_FILE = os.path.join(RED_SAVE_DIR, "red_snake_dueling_ddqn_nstep_v1.pth")
RED_LOG_CSV = os.path.join(RED_SAVE_DIR, "red_training_log_dueling_ddqn_nstep_v1.csv")
BLUE_QTABLE_FILE = os.path.join(BLUE_SAVE_DIR, "blue_snake_qtable.pkl")
LOG_CSV = os.path.join(BLUE_SAVE_DIR, "blue_training_log.csv")
QTABLE_STATE_VERSION = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== RED RAINBOW DQN-LITE SETTINGS =====
# Rainbow-lite here means Dueling DQN + Double DQN + prioritized replay + n-step returns.
RED_ALGORITHM_NAME = "Rainbow DQN-lite"
DQN_GAMMA = 0.99
DQN_LR = 1.0e-4
DQN_BATCH_SIZE = 128
DQN_REPLAY_SIZE = 100_000
DQN_MIN_REPLAY = 512
DQN_TARGET_UPDATE_EVERY = 1000
DQN_TRAIN_EVERY = 1
DQN_UPDATES_PER_STEP = 1
DQN_EPSILON_START = 0.30
DQN_EPSILON_MIN = 0.02
DQN_EPSILON_DECAY = 0.9995
DQN_PRIORITY_ALPHA = 0.60
DQN_PRIORITY_BETA_START = 0.40
DQN_PRIORITY_BETA_FRAMES = 80_000
DQN_PRIORITY_EPS = 1e-5
DQN_N_STEP = 3
DQN_MAX_GRAD_NORM = 10.0
RED_STATE_DIM = 28
RED_DQN_OBJECTIVE_VERSION = 1
RED_RECENT_HEAD_WINDOW = 12
RED_STALL_PENALTY = 8.0
RED_TRAINING_OPPONENT_ATE_PENALTY = 95.0
RED_OPPONENT_ATE_EP_PENALTY = 150.0
RED_STEP_COST = 0.04
RED_PROGRESS_REWARD = 2.8
RED_BACKTRACK_PENALTY = 1.2
RED_FOOD_REWARD = 95.0
RED_CONTESTED_FOOD_BONUS = 22.0
RED_FOOD_LENGTH_BONUS = 1.0
RED_DEATH_PENALTY = 130.0
RED_RACE_DELTA_REWARD = 0.35
RED_RACE_LEAD_REWARD = 0.45
RED_RACE_BEHIND_PENALTY = 0.75

# ===== BLUE Q-TABLE SETTINGS =====
ALPHA = 0.15
GAMMA = 0.97
EPSILON_START = 0.35
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9997

SPEED_LEVELS = [6.0, 10.0, 14.0, 18.0]
N_SPEED = len(SPEED_LEVELS)
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # U D L R
DIR_TO_IDX = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
N_ACTIONS = len(DIRS) * N_SPEED

RED_START_SPEED = SPEED_LEVELS[1]
BLUE_START_SPEED = SPEED_LEVELS[1]

MA_WINDOW = 100
GRAPH_HISTORY = 120
PRINT_EVERY_EPISODES = 10
SAVE_PROGRESS_EVERY_EPISODES = 5
SAVE_PROGRESS_EVERY_STEPS = 2000
OPPONENT_ATE_PENALTY = 150.0

FAST_TRAIN = False
EVAL_MODE = False
MAX_SESSION_FRAMES = int(os.environ.get("SNAKE_MAX_FRAMES", "0") or 0)


def make_initial_action_values():
    speed_bias = [-0.05, 0.20, 0.10, -0.15]
    return [speed_bias[action % N_SPEED] for action in range(N_ACTIONS)]


INITIAL_BLUE_ACTION_VALUES = make_initial_action_values()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake AI Battle - Red Rainbow DQN-lite vs Blue Q-Table")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)
small_font = pygame.font.SysFont("Consolas", 14)


def to_cell(pos):
    return pos[0] // CELL, pos[1] // CELL


def to_pixel(cell):
    return cell[0] * CELL, cell[1] * CELL


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def speed_to_idx(speed):
    return min(range(N_SPEED), key=lambda i: abs(SPEED_LEVELS[i] - speed))


def speed_to_onehot(speed):
    idx = speed_to_idx(speed)
    onehot = [0.0] * N_SPEED
    onehot[idx] = 1.0
    return onehot


# ===== SAFE ZONE =====
SAFE_W_CELLS = 6
SAFE_H_CELLS = 6
SAFE_X0 = (COLS - SAFE_W_CELLS) // 2
SAFE_Y0 = (ROWS - SAFE_H_CELLS) // 2
SAFE_CELLS = {
    (x, y)
    for x in range(SAFE_X0, SAFE_X0 + SAFE_W_CELLS)
    for y in range(SAFE_Y0, SAFE_Y0 + SAFE_H_CELLS)
}
SAFE_RECT_PIX = pygame.Rect(
    SAFE_X0 * CELL,
    SAFE_Y0 * CELL,
    SAFE_W_CELLS * CELL,
    SAFE_H_CELLS * CELL,
)


def in_safe_zone_cell(cell_xy):
    return cell_xy in SAFE_CELLS


def make_background_surface(width, height):
    surf = pygame.Surface((width, height)).convert()
    cx, cy = width * 0.45, height * 0.35
    maxd = (width * width + height * height) ** 0.5
    base_r, base_g, base_b = 5, 5, 15
    glow_r, glow_g, glow_b = 12, 12, 28

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            d = (dx * dx + dy * dy) ** 0.5
            t = min(1.0, d / maxd)
            v = t ** 1.6
            r = int(glow_r * (1 - v) + base_r * v)
            g = int(glow_g * (1 - v) + base_g * v)
            b = int(glow_b * (1 - v) + base_b * v)
            surf.set_at((x, y), (r, g, b))
    return surf


BG = make_background_surface(BOARD_WIDTH, BOARD_HEIGHT)


def random_food(snake1, snake2, extra_occupied=None):
    occupied = set(snake1 + snake2)
    if extra_occupied:
        occupied.update(extra_occupied)
    while True:
        cell = (random.randrange(COLS), random.randrange(ROWS))
        if in_safe_zone_cell(cell):
            continue
        pos = to_pixel(cell)
        if pos not in occupied:
            return pos


def spawn_foods(snake1, snake2, count=NUM_FOODS):
    foods = []
    while len(foods) < count:
        foods.append(random_food(snake1, snake2, foods))
    return foods


def nearest_food(snake, foods):
    if not foods:
        return to_pixel((COLS // 2, ROWS // 2))
    head_cell = to_cell(snake[0])
    return min(foods, key=lambda food: manhattan(head_cell, to_cell(food)))


def replace_eaten_food(foods, eaten_food, snake1, snake2):
    remaining = [food for food in foods if food != eaten_food]
    while len(remaining) < NUM_FOODS:
        remaining.append(random_food(snake1, snake2, remaining))
    return remaining


def legal_moves_for_snake(snake, opponent):
    head = to_cell(snake[0])
    obstacles = set(to_cell(p) for p in snake[1:] + opponent)
    legal = []
    for dx, dy in DIRS:
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in obstacles:
            legal.append((dx, dy))
    return legal


def legal_action_indices(snake, opponent):
    legal_dirs = legal_moves_for_snake(snake, opponent)
    if not legal_dirs:
        return list(range(N_ACTIONS))

    legal_dir_idxs = [DIR_TO_IDX[move] for move in legal_dirs]
    actions = []
    for dir_idx in legal_dir_idxs:
        for speed_idx in range(N_SPEED):
            actions.append(dir_idx * N_SPEED + speed_idx)
    return actions


def get_red_state(snake, opponent, food_pos, cur_dir, my_speed, opp_speed):
    hx, hy = to_cell(snake[0])
    fx, fy = to_cell(food_pos)

    food_dx = (fx - hx) / max(1, COLS - 1)
    food_dy = (fy - hy) / max(1, ROWS - 1)

    if len(snake) > 2:
        tx, ty = to_cell(snake[-1])
        tail_dx = (tx - hx) / max(1, COLS - 1)
        tail_dy = (ty - hy) / max(1, ROWS - 1)
    else:
        tail_dx = 0.0
        tail_dy = 0.0

    my_len = len(snake)
    opp_len = len(opponent)
    size_adv = 1.0 if my_len > opp_len else (-1.0 if my_len < opp_len else 0.0)
    my_len_norm = my_len / (COLS * ROWS)
    opp_len_norm = opp_len / (COLS * ROWS)

    obstacles = set(to_cell(p) for p in snake[1:] + opponent)

    dangers = []
    for dx, dy in DIRS:
        nx, ny = hx + dx, hy + dy
        blocked = not (0 <= nx < COLS and 0 <= ny < ROWS) or (nx, ny) in obstacles
        dangers.append(1.0 if blocked else 0.0)

    food_dist = manhattan((hx, hy), (fx, fy))
    food_dist_norm = food_dist / (COLS + ROWS)

    if opponent:
        ox, oy = to_cell(opponent[0])
        opp_dir_x = (ox - hx) / max(1, COLS - 1)
        opp_dir_y = (oy - hy) / max(1, ROWS - 1)
    else:
        opp_dir_x = 0.0
        opp_dir_y = 0.0

    free_count = 0
    for dx, dy in [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        nx, ny = hx + dx, hy + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in obstacles:
            free_count += 1
    head_free_ratio = free_count / 8.0

    center_dist = manhattan((hx, hy), (COLS // 2, ROWS // 2))
    max_center = max(1, COLS // 2 + ROWS // 2)
    dist_center_norm = center_dist / max_center

    dir_onehot = [0.0, 0.0, 0.0, 0.0]
    dir_onehot[DIR_TO_IDX.get(cur_dir, 3)] = 1.0

    features = [
        float(food_dx),
        float(food_dy),
        float(tail_dx),
        float(tail_dy),
        float(size_adv),
        float(my_len_norm),
        float(opp_len_norm),
        *dangers,
        float(food_dist_norm),
        float(opp_dir_x),
        float(opp_dir_y),
        float(head_free_ratio),
        float(dist_center_norm),
        *dir_onehot,
        *speed_to_onehot(my_speed),
        *speed_to_onehot(opp_speed),
    ]
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


def get_legal_mask(snake, opponent):
    head = to_cell(snake[0])
    obstacles = set(to_cell(p) for p in snake[1:] + opponent)
    mask = []
    for dx, dy in DIRS:
        nx, ny = head[0] + dx, head[1] + dy
        illegal = not (0 <= nx < COLS and 0 <= ny < ROWS) or (nx, ny) in obstacles
        value = -1e9 if illegal else 0.0
        mask.extend([value] * N_SPEED)
    return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)


def legal_actions_from_mask(mask):
    flat_mask = mask.squeeze(0) if mask.dim() > 1 else mask
    legal = (flat_mask > -1e8).nonzero(as_tuple=False).flatten().tolist()
    return legal or list(range(N_ACTIONS))


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim=RED_STATE_DIM, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, N_ACTIONS),
        )

    def forward(self, x):
        features = self.shared(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, next_mask, done):
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        transition = (
            state.copy(),
            int(action),
            float(reward),
            next_state.copy(),
            next_mask.copy(),
            float(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        size = len(self.buffer)
        priorities = self.priorities[:size]
        scaled = priorities ** self.alpha
        total = scaled.sum()
        probs = scaled / total if total > 0 else np.full(size, 1.0 / size, dtype=np.float32)
        probs = probs.astype(np.float64)
        probs = probs / probs.sum()

        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[index] for index in indices]

        states = torch.tensor(np.stack([sample[0] for sample in samples]), dtype=torch.float32, device=device)
        actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long, device=device)
        rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack([sample[3] for sample in samples]), dtype=torch.float32, device=device)
        next_masks = torch.tensor(np.stack([sample[4] for sample in samples]), dtype=torch.float32, device=device)
        dones = torch.tensor([sample[5] for sample in samples], dtype=torch.float32, device=device)

        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, next_masks, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = max(float(priority), DQN_PRIORITY_EPS)


class RedRainbowDQNAgent:
    def __init__(self, path):
        self.path = path
        self.net = DuelingQNetwork().to(device)
        self.target_net = DuelingQNetwork().to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.opt = optim.Adam(self.net.parameters(), lr=DQN_LR)

        self.replay = PrioritizedReplayBuffer(DQN_REPLAY_SIZE, DQN_PRIORITY_ALPHA)
        self.n_step_buffer = deque()

        self.epsilon = DQN_EPSILON_START
        self.steps = 0
        self.training_updates = 0
        self.episodes = 0
        self.total_food = 0
        self.total_deaths = 0
        self.best_score = 0
        self.history_food = []
        self.history_deaths = []

        self.last_loss = 0.0
        self.last_td_error = 0.0
        self.last_q_mean = 0.0
        self.last_reward = 0.0

        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            checkpoint = torch.load(self.path, map_location=device)
            objective_version = checkpoint.get("red_objective_version")
            if objective_version != RED_DQN_OBJECTIVE_VERSION:
                print(
                    f"Red {RED_ALGORITHM_NAME} checkpoint objective version "
                    f"{objective_version or 'legacy'} does not match "
                    f"{RED_DQN_OBJECTIVE_VERSION}; starting fresh."
                )
                return
            self.net.load_state_dict(checkpoint["net"])
            self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["net"]))
            self.opt.load_state_dict(checkpoint["opt"])
            self.epsilon = checkpoint.get("epsilon", DQN_EPSILON_START)
            self.steps = checkpoint.get("steps", 0)
            self.training_updates = checkpoint.get("training_updates", 0)
            self.episodes = checkpoint.get("episodes", 0)
            self.total_food = checkpoint.get("total_food", 0)
            self.total_deaths = checkpoint.get("total_deaths", 0)
            self.best_score = checkpoint.get("best_score", 0)
            self.history_food = checkpoint.get("history_food", [])
            self.history_deaths = checkpoint.get("history_deaths", [])
            print(
                f"Loaded red {RED_ALGORITHM_NAME} checkpoint with steps={self.steps}, "
                f"episodes={self.episodes}, epsilon={self.epsilon:.3f}"
            )
        except Exception as exc:
            print(f"Red {RED_ALGORITHM_NAME} load failed. Starting fresh. Error:", exc)

    def save(self):
        torch.save(
            {
                "net": self.net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "opt": self.opt.state_dict(),
                "algorithm": RED_ALGORITHM_NAME,
                "rainbow_lite_components": [
                    "dueling_network",
                    "double_dqn_targets",
                    "prioritized_replay",
                    "n_step_returns",
                ],
                "red_objective_version": RED_DQN_OBJECTIVE_VERSION,
                "epsilon": self.epsilon,
                "steps": self.steps,
                "training_updates": self.training_updates,
                "episodes": self.episodes,
                "total_food": self.total_food,
                "total_deaths": self.total_deaths,
                "best_score": self.best_score,
                "history_food": self.history_food[-1000:],
                "history_deaths": self.history_deaths[-1000:],
            },
            self.path,
        )

    @torch.no_grad()
    def act(self, state, mask, eval_mode=False):
        legal_actions = legal_actions_from_mask(mask)
        if not eval_mode and random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            q_values = self.net(state) + mask
            action = int(torch.argmax(q_values, dim=1).item())

        if not eval_mode:
            self.epsilon = max(DQN_EPSILON_MIN, self.epsilon * DQN_EPSILON_DECAY)

        return action

    def _tensor_to_np(self, tensor):
        return tensor.detach().squeeze(0).cpu().numpy().astype(np.float32, copy=True)

    def store(self, state, action, reward, next_state, next_mask, done):
        transition = (
            self._tensor_to_np(state),
            int(action),
            float(reward),
            self._tensor_to_np(next_state),
            self._tensor_to_np(next_mask),
            bool(done),
        )
        self.last_reward = float(reward)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= DQN_N_STEP:
            self._push_n_step_transition()

        if done:
            while self.n_step_buffer:
                self._push_n_step_transition()

    def _push_n_step_transition(self):
        if not self.n_step_buffer:
            return

        reward = 0.0
        next_state = self.n_step_buffer[-1][3]
        next_mask = self.n_step_buffer[-1][4]
        done = False

        for idx, transition in enumerate(self.n_step_buffer):
            reward += (DQN_GAMMA ** idx) * transition[2]
            next_state = transition[3]
            next_mask = transition[4]
            if transition[5]:
                done = True
                break

        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        self.replay.add(state, action, reward, next_state, next_mask, done)
        self.n_step_buffer.popleft()

    def mark_last_transition_terminal(self):
        if not self.n_step_buffer:
            return False

        state, action, reward, next_state, next_mask, _ = self.n_step_buffer[-1]
        self.n_step_buffer[-1] = (state, action, reward, next_state, next_mask, True)
        while self.n_step_buffer:
            self._push_n_step_transition()
        return True

    def apply_opponent_food_penalty(self, penalty):
        if not self.n_step_buffer:
            return False

        state, action, reward, next_state, next_mask, done = self.n_step_buffer[-1]
        reward -= float(penalty)
        self.n_step_buffer[-1] = (state, action, reward, next_state, next_mask, done)
        self.last_reward = reward
        return True

    def update(self):
        if self.steps % DQN_TRAIN_EVERY != 0:
            return False
        if len(self.replay) < DQN_MIN_REPLAY:
            return False

        beta_progress = min(1.0, self.steps / max(1, DQN_PRIORITY_BETA_FRAMES))
        beta = DQN_PRIORITY_BETA_START + beta_progress * (1.0 - DQN_PRIORITY_BETA_START)

        (
            states,
            actions,
            rewards,
            next_states,
            next_masks,
            dones,
            weights,
            indices,
        ) = self.replay.sample(DQN_BATCH_SIZE, beta)

        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q = self.net(next_states) + next_masks
            next_actions = torch.argmax(next_online_q, dim=1)
            next_target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (DQN_GAMMA ** DQN_N_STEP) * (1.0 - dones) * next_target_q

        td_errors = targets - q_values
        loss_per_sample = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights * loss_per_sample).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), DQN_MAX_GRAD_NORM)
        self.opt.step()

        priorities = td_errors.detach().abs().cpu().numpy() + DQN_PRIORITY_EPS
        self.replay.update_priorities(indices, priorities)

        self.training_updates += 1
        if self.training_updates % DQN_TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.last_loss = float(loss.detach().cpu())
        self.last_td_error = float(td_errors.detach().abs().mean().cpu())
        self.last_q_mean = float(q_values.detach().mean().cpu())
        return True


def get_blue_q_state(snake, opponent, food_pos, cur_dir, my_speed, opp_speed):
    hx, hy = to_cell(snake[0])
    fx, fy = to_cell(food_pos)
    ox, oy = to_cell(opponent[0])

    obstacles = set(to_cell(p) for p in snake[1:] + opponent)

    dangers = []
    for dx, dy in DIRS:
        nx, ny = hx + dx, hy + dy
        blocked = not (0 <= nx < COLS and 0 <= ny < ROWS) or (nx, ny) in obstacles
        dangers.append(1 if blocked else 0)

    free_neighbors = 0
    for dx, dy in [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        nx, ny = hx + dx, hy + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in obstacles:
            free_neighbors += 1

    food_bucket = min(manhattan((hx, hy), (fx, fy)) // 5, 4)
    length_adv = sign(len(snake) - len(opponent))
    free_bucket = min(free_neighbors // 2, 4)
    opponent_near = 1 if manhattan((hx, hy), (ox, oy)) <= 4 else 0
    food_race = sign(manhattan((ox, oy), (fx, fy)) - manhattan((hx, hy), (fx, fy)))
    my_speed_idx = speed_to_idx(my_speed)
    speed_adv = sign(my_speed_idx - speed_to_idx(opp_speed))

    return (
        sign(fx - hx),
        sign(fy - hy),
        *dangers,
        food_bucket,
        free_bucket,
        length_adv,
        opponent_near,
        food_race,
        my_speed_idx,
        speed_adv,
    )


class BlueQAgent:
    def __init__(self, path):
        self.path = path
        self.qtable = {}
        self.epsilon = EPSILON_START
        self.steps = 0
        self.episodes = 0
        self.total_food = 0
        self.total_deaths = 0
        self.best_score = 0
        self.last_td_error = 0.0
        self.last_reward = 0.0
        self.last_state_key = None
        self.last_action_idx = None
        self.history_food = []
        self.history_deaths = []
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "rb") as file:
                data = pickle.load(file)
            saved_version = data.get("state_version", 1)
            if saved_version != QTABLE_STATE_VERSION:
                print(
                    f"Blue Q-table state version changed ({saved_version} -> {QTABLE_STATE_VERSION}). "
                    "Starting a fresh table for the new state layout."
                )
                self.qtable = {}
            else:
                self.qtable = data.get("qtable", {})
            self.epsilon = data.get("epsilon", EPSILON_START)
            self.steps = data.get("steps", 0)
            self.episodes = data.get("episodes", 0)
            self.total_food = data.get("total_food", 0)
            self.total_deaths = data.get("total_deaths", 0)
            self.best_score = data.get("best_score", 0)
            self.history_food = data.get("history_food", [])
            self.history_deaths = data.get("history_deaths", [])
            print(
                f"Loaded blue Q-table with {len(self.qtable)} states, "
                f"episodes={self.episodes}, epsilon={self.epsilon:.3f}"
            )
        except Exception as exc:
            print("Blue Q-table load failed. Starting fresh. Error:", exc)

    def save(self):
        data = {
            "state_version": QTABLE_STATE_VERSION,
            "qtable": self.qtable,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "total_food": self.total_food,
            "total_deaths": self.total_deaths,
            "best_score": self.best_score,
            "history_food": self.history_food[-1000:],
            "history_deaths": self.history_deaths[-1000:],
        }
        with open(self.path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def ensure_state(self, state):
        if state not in self.qtable:
            self.qtable[state] = INITIAL_BLUE_ACTION_VALUES.copy()

    def select_action(self, state, legal_actions, eval_mode=False):
        self.ensure_state(state)
        qvalues = self.qtable[state]

        if not eval_mode and random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            best_q = max(qvalues[action] for action in legal_actions)
            best_actions = [action for action in legal_actions if qvalues[action] == best_q]
            action = random.choice(best_actions)

        if not eval_mode:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        return action

    def max_q(self, state, legal_actions):
        self.ensure_state(state)
        return max(self.qtable[state][action] for action in legal_actions)

    def update(self, state, action, reward, next_state=None, next_legal=None, done=False):
        self.ensure_state(state)
        old_q = self.qtable[state][action]

        if done or next_state is None or not next_legal:
            target = reward
        else:
            target = reward + GAMMA * self.max_q(next_state, next_legal)

        new_q = old_q + ALPHA * (target - old_q)
        self.qtable[state][action] = new_q
        self.last_state_key = state
        self.last_action_idx = action
        self.last_td_error = target - old_q
        self.last_reward = reward

    def apply_opponent_food_penalty(self, penalty):
        if self.last_state_key is None or self.last_action_idx is None:
            return False
        self.ensure_state(self.last_state_key)
        self.qtable[self.last_state_key][self.last_action_idx] -= ALPHA * penalty
        self.last_td_error -= penalty
        self.last_reward -= penalty
        return True


def safe_spawn_positions():
    x_left = SAFE_X0 + 1
    x_right = SAFE_X0 + SAFE_W_CELLS - 2
    y_mid = SAFE_Y0 + SAFE_H_CELLS // 2
    y1 = max(SAFE_Y0 + 1, y_mid - 1)
    y2 = min(SAFE_Y0 + SAFE_H_CELLS - 2, y_mid + 1)

    head1 = to_pixel((x_left, y1))
    head2 = to_pixel((x_right, y2))

    snake1 = [head1, (head1[0] - CELL, head1[1]), (head1[0] - 2 * CELL, head1[1])]
    snake2 = [head2, (head2[0] + CELL, head2[1]), (head2[0] + 2 * CELL, head2[1])]

    dir1 = (1, 0)
    dir2 = (-1, 0)
    return snake1, dir1, snake2, dir2


def step_red_dqn(
    agent,
    snake,
    opponent,
    foods,
    cur_dir,
    training_enabled,
    snake1_ref,
    snake2_ref,
    my_speed,
    opp_speed,
    ep_stats,
    recent_red_heads,
    eval_mode=False,
):
    target_food = nearest_food(snake, foods)
    target_cell = to_cell(target_food)
    head_cell = to_cell(snake[0])
    prev_dist = manhattan(head_cell, target_cell) if snake else 0
    opp_prev_dist = manhattan(to_cell(opponent[0]), target_cell) if opponent else prev_dist
    prev_race_margin = opp_prev_dist - prev_dist

    state = get_red_state(snake, opponent, target_food, cur_dir, my_speed, opp_speed)
    mask = get_legal_mask(snake, opponent)
    action = agent.act(state, mask, eval_mode=eval_mode)
    agent.steps += 1

    dir_idx = action // N_SPEED
    speed_idx = action % N_SPEED
    dx, dy = DIRS[dir_idx]
    new_dir = (dx, dy)
    new_speed = SPEED_LEVELS[speed_idx]

    new_head = (snake[0][0] + dx * CELL, snake[0][1] + dy * CELL)
    new_head_cell = to_cell(new_head)
    repeat_count = recent_red_heads.count(new_head_cell)
    snake.insert(0, new_head)

    eaten_food = new_head if new_head in foods else None
    ate = eaten_food is not None
    missed = False

    new_foods = foods
    if ate:
        new_foods = replace_eaten_food(foods, eaten_food, snake1_ref, snake2_ref)
    else:
        snake.pop()

    dead = (
        new_head[0] < 0
        or new_head[0] >= BOARD_WIDTH
        or new_head[1] < 0
        or new_head[1] >= BOARD_HEIGHT
        or new_head in snake[1:]
        or new_head in opponent
    )

    target_dist_now = manhattan(new_head_cell, target_cell)
    next_target_food = nearest_food(snake, new_foods)
    future_options = 0 if dead else len(legal_moves_for_snake(snake, opponent))
    opp_dist_now = manhattan(to_cell(opponent[0]), target_cell) if opponent else target_dist_now
    race_margin_now = opp_dist_now - target_dist_now
    race_delta = race_margin_now - prev_race_margin

    len_bonus = float(len(snake))
    progress = prev_dist - target_dist_now
    contested_food = opp_prev_dist <= prev_dist + 2

    reward = RED_PROGRESS_REWARD * progress
    reward -= RED_STEP_COST
    if progress < 0:
        reward += RED_BACKTRACK_PENALTY * progress
    if ate:
        reward += RED_FOOD_REWARD
        reward += RED_FOOD_LENGTH_BONUS * len_bonus
        if contested_food:
            reward += RED_CONTESTED_FOOD_BONUS
    if missed:
        reward -= 45.0
    if dead:
        reward -= RED_DEATH_PENALTY

    if not dead:
        if repeat_count > 0 and not ate:
            reward -= RED_STALL_PENALTY * (1.0 + 0.35 * (repeat_count - 1))
        if future_options <= 1:
            reward -= 8.0
        elif future_options == 2:
            reward -= 3.0
        else:
            reward += min(2, future_options - 2) * 0.15
        if not ate:
            reward += max(-4.0, min(4.0, race_delta)) * RED_RACE_DELTA_REWARD
            if race_margin_now > 0:
                reward += RED_RACE_LEAD_REWARD
            elif race_margin_now < 0:
                reward -= RED_RACE_BEHIND_PENALTY
            if prev_dist <= 4 and progress <= 0:
                reward -= 4.0
            elif target_dist_now <= 2:
                reward += 1.0
        if speed_idx == N_SPEED - 1 and future_options <= 2:
            reward -= 2.5
        elif speed_idx >= 2 and prev_dist > 7 and future_options >= 3:
            reward += 0.7
        elif speed_idx == 0 and prev_dist > 7:
            reward -= 0.8

    if not dead and 0 <= new_head[0] < BOARD_WIDTH and 0 <= new_head[1] < BOARD_HEIGHT:
        recent_red_heads.append(new_head_cell)

    ep_stats["ret"] += float(reward)
    ep_stats["len"] += 1
    if ate:
        ep_stats["food"] += 1
    if dead:
        ep_stats["dead"] = 1

    if training_enabled and not eval_mode:
        if dead:
            next_state = state
            next_mask = torch.zeros_like(mask)
        else:
            next_state = get_red_state(snake, opponent, next_target_food, new_dir, new_speed, opp_speed)
            next_mask = get_legal_mask(snake, opponent)

        agent.store(state, action, reward, next_state, next_mask, dead)
        for _ in range(DQN_UPDATES_PER_STEP):
            agent.update()

    return dead, new_dir, new_speed, ate, missed, new_foods


def step_blue_q(agent, snake, opponent, foods, cur_dir, training_enabled, snake1_ref, snake2_ref, my_speed, opp_speed, ep_stats, eval_mode=False):
    target_food = nearest_food(snake, foods)
    prev_dist = manhattan(to_cell(snake[0]), to_cell(target_food)) if snake else 0

    state = get_blue_q_state(snake, opponent, target_food, cur_dir, my_speed, opp_speed)
    legal_actions = legal_action_indices(snake, opponent)
    action = agent.select_action(state, legal_actions, eval_mode=eval_mode or not training_enabled)
    agent.steps += 1

    dir_idx = action // N_SPEED
    speed_idx = action % N_SPEED
    dx, dy = DIRS[dir_idx]
    new_dir = (dx, dy)
    new_speed = SPEED_LEVELS[speed_idx]

    new_head = (snake[0][0] + dx * CELL, snake[0][1] + dy * CELL)
    snake.insert(0, new_head)

    eaten_food = new_head if new_head in foods else None
    ate = eaten_food is not None
    missed = False

    new_foods = foods
    if ate:
        new_foods = replace_eaten_food(foods, eaten_food, snake1_ref, snake2_ref)
    else:
        snake.pop()

    dead = (
        new_head[0] < 0
        or new_head[0] >= BOARD_WIDTH
        or new_head[1] < 0
        or new_head[1] >= BOARD_HEIGHT
        or new_head in snake[1:]
        or new_head in opponent
    )

    next_target_food = nearest_food(snake, new_foods)
    dist_now = manhattan(to_cell(new_head), to_cell(next_target_food))
    future_options = 0 if dead else len(legal_moves_for_snake(snake, opponent))
    opp_to_food = manhattan(to_cell(opponent[0]), to_cell(next_target_food))
    food_race_after = sign(opp_to_food - dist_now)

    len_bonus = float(len(snake))
    size_edge = max(0.0, float(len(snake) - len(opponent)))

    reward = (prev_dist - dist_now) * 2.8
    reward -= 0.04
    if ate:
        reward += 85.0
    if missed:
        reward -= 40.0
    if dead:
        reward -= 120.0
    reward += 0.08 * len_bonus
    reward += 0.24 * size_edge
    if ate:
        reward += 2.0 * len_bonus

    if not dead:
        reward += future_options * 0.50
        if future_options <= 1:
            reward -= 4.0
        elif future_options == 2:
            reward -= 1.2
        if dist_now > prev_dist:
            reward -= 1.0
        if food_race_after > 0:
            reward += 1.3
        elif food_race_after < 0:
            reward -= 0.9
        if speed_idx == N_SPEED - 1 and future_options <= 2:
            reward -= 1.8
        elif speed_idx == 0 and dist_now > 8:
            reward -= 0.45

    ep_stats["ret"] += float(reward)
    ep_stats["len"] += 1
    if ate:
        ep_stats["food"] += 1
    if dead:
        ep_stats["dead"] = 1

    if training_enabled and not eval_mode:
        next_state = None
        next_legal = None
        if not dead:
            next_state = get_blue_q_state(
                snake,
                opponent,
                next_target_food,
                new_dir,
                new_speed,
                opp_speed,
            )
            next_legal = legal_action_indices(snake, opponent)
        agent.update(state, action, reward, next_state=next_state, next_legal=next_legal, done=dead)

    return dead, new_dir, new_speed, ate, missed, new_foods


def draw_progress_graph(surface, history_food, history_deaths, x, y, w, h):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, (18, 22, 34), rect, border_radius=8)
    pygame.draw.rect(surface, (70, 82, 120), rect, width=1, border_radius=8)

    inner = rect.inflate(-24, -36)
    if inner.w <= 0 or inner.h <= 0:
        return

    pygame.draw.line(surface, (90, 100, 120), (inner.left, inner.bottom), (inner.right, inner.bottom), 1)
    pygame.draw.line(surface, (90, 100, 120), (inner.left, inner.top), (inner.left, inner.bottom), 1)

    if len(history_food) < 2:
        surface.blit(
            small_font.render("Need more episodes for graph", True, (180, 180, 190)),
            (inner.x, inner.y + inner.h // 2 - 8),
        )
        return

    max_value = max(1, max(history_food), max(history_deaths))

    def to_points(values):
        points = []
        denom = max(1, len(values) - 1)
        for i, value in enumerate(values):
            px = inner.left + int((i / denom) * inner.w)
            py = inner.bottom - int((value / max_value) * inner.h)
            points.append((px, py))
        return points

    food_points = to_points(history_food)
    death_points = to_points(history_deaths)

    if len(food_points) >= 2:
        pygame.draw.lines(surface, (80, 190, 255), False, food_points, 2)
    if len(death_points) >= 2:
        pygame.draw.lines(surface, (255, 120, 120), False, death_points, 2)

    surface.blit(small_font.render("Blue food", True, (80, 190, 255)), (rect.x + 12, rect.y + 8))
    surface.blit(small_font.render("Blue deaths", True, (255, 120, 120)), (rect.x + 110, rect.y + 8))
    surface.blit(small_font.render(f"Max: {max_value}", True, (190, 190, 200)), (rect.right - 78, rect.y + 8))


def draw_compare_graph(surface, red_values, blue_values, x, y, w, h, title, red_label, blue_label):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, (18, 22, 34), rect, border_radius=8)
    pygame.draw.rect(surface, (70, 82, 120), rect, width=1, border_radius=8)

    inner = rect.inflate(-24, -42)
    if inner.w <= 0 or inner.h <= 0:
        return

    pygame.draw.line(surface, (90, 100, 120), (inner.left, inner.bottom), (inner.right, inner.bottom), 1)
    pygame.draw.line(surface, (90, 100, 120), (inner.left, inner.top), (inner.left, inner.bottom), 1)

    surface.blit(small_font.render(title, True, (220, 220, 220)), (rect.x + 12, rect.y + 8))
    surface.blit(small_font.render(red_label, True, (255, 120, 120)), (inner.left + 4, rect.y + 24))
    surface.blit(small_font.render(blue_label, True, (80, 190, 255)), (inner.left + 145, rect.y + 24))

    if len(red_values) < 2 and len(blue_values) < 2:
        surface.blit(
            small_font.render("Need more episodes for graph", True, (180, 180, 190)),
            (inner.x, inner.y + inner.h // 2 - 8),
        )
        return

    max_value = max(1, max(red_values or [0]), max(blue_values or [0]))

    def to_points(values):
        if len(values) < 2:
            return []
        points = []
        denom = max(1, len(values) - 1)
        for i, value in enumerate(values):
            px = inner.left + int((i / denom) * inner.w)
            py = inner.bottom - int((value / max_value) * inner.h)
            points.append((px, py))
        return points

    red_points = to_points(red_values)
    blue_points = to_points(blue_values)

    if len(red_points) >= 2:
        pygame.draw.lines(surface, (255, 120, 120), False, red_points, 2)
    if len(blue_points) >= 2:
        pygame.draw.lines(surface, (80, 190, 255), False, blue_points, 2)

    surface.blit(small_font.render(f"Max: {max_value}", True, (190, 190, 200)), (rect.right - 78, rect.y + 8))


def rolling_death_rate(values, window=20):
    rates = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sample = values[start:i + 1]
        rates.append(100.0 * sum(sample) / max(1, len(sample)))
    return rates


def draw_death_rate_graph(surface, red_deaths, blue_deaths, x, y, w, h, window=20):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, (18, 22, 34), rect, border_radius=8)
    pygame.draw.rect(surface, (70, 82, 120), rect, width=1, border_radius=8)

    inner = rect.inflate(-30, -46)
    if inner.w <= 0 or inner.h <= 0:
        return

    title = f"Death Rate ({window} ep avg)"
    surface.blit(
        small_font.render(title, True, (220, 220, 220)),
        (rect.x + 12, rect.y + 8)
    )

    hint = "Lower is better"
    hint_surface = small_font.render(hint, True, (180, 180, 190))
    surface.blit(
        hint_surface,
        (rect.right - hint_surface.get_width() - 12, rect.y + 8)
    )

    for pct in (0, 50, 100):
        py = inner.bottom - int((pct / 100.0) * inner.h)
        color = (70, 78, 98) if pct in (0, 100) else (45, 52, 70)
        pygame.draw.line(surface, color, (inner.left, py), (inner.right, py), 1)
        surface.blit(small_font.render(f"{pct}%", True, (145, 150, 165)), (rect.x + 6, py - 7))

    red_rates = rolling_death_rate(red_deaths, window)
    blue_rates = rolling_death_rate(blue_deaths, window)

    if len(red_rates) < 2 and len(blue_rates) < 2:
        surface.blit(
            small_font.render("Need more episodes for graph", True, (180, 180, 190)),
            (inner.x, inner.y + inner.h // 2 - 8),
        )
        return

    def to_points(values):
        if len(values) < 2:
            return []
        points = []
        denom = max(1, len(values) - 1)
        for i, value in enumerate(values):
            px = inner.left + int((i / denom) * inner.w)
            py = inner.bottom - int((value / 100.0) * inner.h)
            points.append((px, py))
        return points

    red_points = to_points(red_rates)
    blue_points = to_points(blue_rates)

    if len(red_points) >= 2:
        pygame.draw.lines(surface, (255, 120, 120), False, red_points, 3)
    if len(blue_points) >= 2:
        pygame.draw.lines(surface, (80, 190, 255), False, blue_points, 3)

    red_latest = red_rates[-1] if red_rates else 0.0
    blue_latest = blue_rates[-1] if blue_rates else 0.0
    surface.blit(small_font.render(f"Red {red_latest:.0f}%", True, (255, 120, 120)), (inner.left + 4, rect.y + 26))
    surface.blit(small_font.render(f"Blue {blue_latest:.0f}%", True, (80, 190, 255)), (inner.left + 115, rect.y + 26))


def ensure_csv_header():
    if os.path.exists(LOG_CSV):
        return
    with open(LOG_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "episode",
                "blue_steps",
                "blue_epsilon",
                "qtable_states",
                "blue_return",
                "blue_len",
                "blue_food",
                "blue_dead",
                "blue_best_score",
                "blue_total_food",
                "blue_total_deaths",
                "last_td_error",
            ]
        )


def append_csv(row):
    with open(LOG_CSV, "a", newline="") as file:
        csv.writer(file).writerow(row)


def ensure_red_csv_header():
    if os.path.exists(RED_LOG_CSV):
        return
    with open(RED_LOG_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "episode",
                "red_steps",
                "red_return",
                "red_len",
                "red_food",
                "red_dead",
                "red_best_score",
                "red_total_food",
                "red_total_deaths",
                "red_loss",
                "red_td_error",
                "red_q_mean",
                "red_epsilon",
                "red_replay_size",
                "red_updates",
            ]
        )


def append_red_csv(row):
    with open(RED_LOG_CSV, "a", newline="") as file:
        csv.writer(file).writerow(row)


ensure_csv_header()
ensure_red_csv_header()
red_agent = RedRainbowDQNAgent(RED_MODEL_FILE)
blue_agent = BlueQAgent(BLUE_QTABLE_FILE)

snake1, dir1, snake2, dir2 = safe_spawn_positions()
foods = spawn_foods(snake1, snake2)
recent_red_heads = deque((to_cell(pos) for pos in snake1), maxlen=RED_RECENT_HEAD_WINDOW)

score1 = 0
score2 = 0
best1 = red_agent.best_score
best2 = blue_agent.best_score

accum1 = 0.0
accum2 = 0.0
cur_speed1 = RED_START_SPEED
cur_speed2 = BLUE_START_SPEED
session_frames = 0

ma_red_ret = deque(maxlen=MA_WINDOW)
ma_blue_ret = deque(maxlen=MA_WINDOW)
# Use session-only comparison histories so the overlay graph compares the
# current head-to-head run instead of mixing unrelated saved runs.
red_food_history = deque(maxlen=GRAPH_HISTORY)
red_death_history = deque(maxlen=GRAPH_HISTORY)
blue_food_history = deque(maxlen=GRAPH_HISTORY)
blue_death_history = deque(maxlen=GRAPH_HISTORY)

ep_red = {"ret": 0.0, "len": 0, "food": 0, "dead": 0}
ep_blue = {"ret": 0.0, "len": 0, "food": 0, "dead": 0}

while True:
    dt = (clock.tick(RENDER_FPS) / 1000.0) if not FAST_TRAIN else (1.0 / 60.0)

    dead1 = False
    dead2 = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            red_agent.save()
            blue_agent.save()
            print(f"Saved red {RED_ALGORITHM_NAME} to {RED_MODEL_FILE}")
            print(f"Saved blue Q-table to {BLUE_QTABLE_FILE}")
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                FAST_TRAIN = not FAST_TRAIN
                print("[TOGGLE] FAST_TRAIN =", FAST_TRAIN)
            if event.key == pygame.K_e:
                EVAL_MODE = not EVAL_MODE
                print("[TOGGLE] EVAL_MODE =", EVAL_MODE, "(greedy play, no learning)")

    accum1 += dt * cur_speed1
    accum2 += dt * cur_speed2
    red_food_events = 0
    blue_food_events = 0

    while accum1 >= 1.0 and not dead1:
        dead1, dir1, cur_speed1, ate1, _, foods = step_red_dqn(
            red_agent,
            snake1,
            snake2,
            foods,
            dir1,
            training_enabled=not EVAL_MODE,
            snake1_ref=snake1,
            snake2_ref=snake2,
            my_speed=cur_speed1,
            opp_speed=cur_speed2,
            ep_stats=ep_red,
            recent_red_heads=recent_red_heads,
            eval_mode=EVAL_MODE,
        )
        accum1 -= 1.0
        if ate1:
            score1 += 1
            red_food_events += 1

    if red_food_events > 0:
        penalty = OPPONENT_ATE_PENALTY * red_food_events
        ep_blue["ret"] -= penalty
        if not EVAL_MODE:
            blue_agent.apply_opponent_food_penalty(penalty)

    while accum2 >= 1.0 and not dead2:
        dead2, dir2, cur_speed2, ate2, _, foods = step_blue_q(
            blue_agent,
            snake2,
            snake1,
            foods,
            dir2,
            training_enabled=not EVAL_MODE,
            snake1_ref=snake1,
            snake2_ref=snake2,
            my_speed=cur_speed2,
            opp_speed=cur_speed1,
            ep_stats=ep_blue,
            eval_mode=EVAL_MODE,
        )
        accum2 -= 1.0
        if ate2:
            score2 += 1
            blue_food_events += 1

    if blue_food_events > 0:
        penalty = RED_OPPONENT_ATE_EP_PENALTY * blue_food_events
        ep_red["ret"] -= penalty
        if not EVAL_MODE:
            red_agent.apply_opponent_food_penalty(
                RED_TRAINING_OPPONENT_ATE_PENALTY * blue_food_events
            )

    if dead1 or dead2:
        if not EVAL_MODE:
            if dead2 and not dead1:
                red_agent.mark_last_transition_terminal()
            for _ in range(DQN_UPDATES_PER_STEP):
                red_agent.update()

        best1 = max(best1, score1)
        best2 = max(best2, score2)
        red_agent.best_score = max(red_agent.best_score, score1)
        blue_agent.best_score = max(blue_agent.best_score, score2)

        if dead1:
            ep_red["dead"] = 1
        if dead2:
            ep_blue["dead"] = 1

        score1 = 0
        score2 = 0

        red_agent.episodes += 1
        red_agent.total_food += ep_red["food"]
        red_agent.total_deaths += ep_red["dead"]
        red_agent.history_food.append(ep_red["food"])
        red_agent.history_deaths.append(ep_red["dead"])
        red_agent.history_food = red_agent.history_food[-1000:]
        red_agent.history_deaths = red_agent.history_deaths[-1000:]

        blue_agent.episodes += 1
        blue_agent.total_food += ep_blue["food"]
        blue_agent.total_deaths += ep_blue["dead"]
        blue_agent.history_food.append(ep_blue["food"])
        blue_agent.history_deaths.append(ep_blue["dead"])
        blue_agent.history_food = blue_agent.history_food[-1000:]
        blue_agent.history_deaths = blue_agent.history_deaths[-1000:]

        ma_red_ret.append(ep_red["ret"])
        ma_blue_ret.append(ep_blue["ret"])
        red_food_history.append(ep_red["food"])
        red_death_history.append(ep_red["dead"])
        blue_food_history.append(ep_blue["food"])
        blue_death_history.append(ep_blue["dead"])

        timestamp = int(time.time())
        append_red_csv(
            [
                timestamp,
                red_agent.episodes,
                red_agent.steps,
                ep_red["ret"],
                ep_red["len"],
                ep_red["food"],
                ep_red["dead"],
                red_agent.best_score,
                red_agent.total_food,
                red_agent.total_deaths,
                red_agent.last_loss,
                red_agent.last_td_error,
                red_agent.last_q_mean,
                red_agent.epsilon,
                len(red_agent.replay),
                red_agent.training_updates,
            ]
        )
        append_csv(
            [
                timestamp,
                blue_agent.episodes,
                blue_agent.steps,
                blue_agent.epsilon,
                len(blue_agent.qtable),
                ep_blue["ret"],
                ep_blue["len"],
                ep_blue["food"],
                ep_blue["dead"],
                blue_agent.best_score,
                blue_agent.total_food,
                blue_agent.total_deaths,
                blue_agent.last_td_error,
            ]
        )

        if red_agent.episodes % PRINT_EVERY_EPISODES == 0:
            print(
                f"EP {red_agent.episodes} | Red steps {red_agent.steps} | Blue steps {blue_agent.steps} | "
                f"FAST={FAST_TRAIN} EVAL={EVAL_MODE}\n"
                f"  RED RAINBOW-lite : epRet {ep_red['ret']:.1f} food {ep_red['food']} dead {ep_red['dead']} | "
                f"MA{MA_WINDOW} Ret {mean_or_zero(ma_red_ret):.1f} | "
                f"Loss {red_agent.last_loss:.3f} TD {red_agent.last_td_error:.2f} "
                f"EPS {red_agent.epsilon:.3f} Replay {len(red_agent.replay)}\n"
                f"  BLUE Q  : epRet {ep_blue['ret']:.1f} food {ep_blue['food']} dead {ep_blue['dead']} | "
                f"MA{MA_WINDOW} Ret {mean_or_zero(ma_blue_ret):.1f} | "
                f"QStates {len(blue_agent.qtable)} EPS {blue_agent.epsilon:.3f} TD {blue_agent.last_td_error:.2f}"
            )

        if red_agent.episodes % SAVE_PROGRESS_EVERY_EPISODES == 0:
            red_agent.save()
            blue_agent.save()
            print(
                f"[SAVE] episode={red_agent.episodes} | "
                f"red={RED_MODEL_FILE} | blue={BLUE_QTABLE_FILE}"
            )

        ep_red = {"ret": 0.0, "len": 0, "food": 0, "dead": 0}
        ep_blue = {"ret": 0.0, "len": 0, "food": 0, "dead": 0}

        snake1, dir1, snake2, dir2 = safe_spawn_positions()
        accum1 = 0.0
        accum2 = 0.0
        cur_speed1 = RED_START_SPEED
        cur_speed2 = BLUE_START_SPEED
        foods = spawn_foods(snake1, snake2)
        recent_red_heads = deque((to_cell(pos) for pos in snake1), maxlen=RED_RECENT_HEAD_WINDOW)

    session_frames += 1
    if MAX_SESSION_FRAMES and session_frames >= MAX_SESSION_FRAMES:
        pygame.quit()
        sys.exit()

    if session_frames % SAVE_PROGRESS_EVERY_STEPS == 0:
        red_agent.save()
        blue_agent.save()
        print(
            f"[SAVE] frame={session_frames} | "
            f"red={RED_MODEL_FILE} | blue={BLUE_QTABLE_FILE}"
        )

    if not FAST_TRAIN:
        screen.blit(BG, (0, 0))

        safe_fill = pygame.Surface((SAFE_RECT_PIX.w, SAFE_RECT_PIX.h), pygame.SRCALPHA)
        safe_fill.fill((0, 0, 0, 35))
        screen.blit(safe_fill, (SAFE_RECT_PIX.x, SAFE_RECT_PIX.y))
        pygame.draw.rect(screen, (35, 35, 60), SAFE_RECT_PIX, width=2)

        for food in foods:
            pygame.draw.rect(screen, (255, 80, 80), (*food, CELL, CELL))

        for i, pos in enumerate(snake1):
            t = i / max(1, len(snake1) - 1)
            hue = (340 + 55 * t) % 360
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 100)
            pygame.draw.rect(screen, color, (*pos, CELL, CELL))

        for i, pos in enumerate(snake2):
            hue = 180 + 60 * (i / max(1, len(snake2) - 1))
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 100)
            pygame.draw.rect(screen, color, (*pos, CELL, CELL))

        red_last_food = red_food_history[-1] if red_food_history else 0
        red_last_death = red_death_history[-1] if red_death_history else 0
        blue_last_food = blue_food_history[-1] if blue_food_history else 0
        blue_last_death = blue_death_history[-1] if blue_death_history else 0

        screen.blit(
            font.render(
                f"RED DQN  Score: {score1}  Best: {best1}  Len: {len(snake1)}  SPD: {cur_speed1:.0f}",
                True,
                (255, 100, 100),
            ),
            (10, 10),
        )
        screen.blit(
            font.render(
                f"BLUE Q   Score: {score2}  Best: {best2}  Len: {len(snake2)}  SPD: {cur_speed2:.0f}",
                True,
                (100, 200, 255),
            ),
            (10, 40),
        )
        screen.blit(
            font.render(
                f"Red Steps: {red_agent.steps}  Blue Steps: {blue_agent.steps}",
                True,
                (255, 255, 150),
            ),
            (10, 70),
        )
        screen.blit(
            font.render(
                f"MA{MA_WINDOW} RED Ret {mean_or_zero(ma_red_ret):.1f} | BLUE Ret {mean_or_zero(ma_blue_ret):.1f}",
                True,
                (220, 220, 220),
            ),
            (10, 95),
        )
        screen.blit(
            font.render(
                f"[F]=FAST {FAST_TRAIN}   [E]=EVAL {EVAL_MODE}",
                True,
                (200, 200, 200),
            ),
            (10, 120),
        )

        panel_x = BOARD_WIDTH + 10
        pygame.draw.rect(screen, (10, 12, 20), (BOARD_WIDTH, 0, PANEL_WIDTH, BOARD_HEIGHT))
        screen.blit(font.render("Algorithm Comparison", True, (220, 220, 220)), (panel_x, 12))
        screen.blit(small_font.render(f"Red DQN episodes {red_agent.episodes} | best {red_agent.best_score}", True, (255, 140, 140)), (panel_x, 40))
        screen.blit(small_font.render(f"Blue Q episodes {blue_agent.episodes} | best {blue_agent.best_score}", True, (120, 200, 255)), (panel_x, 60))
        screen.blit(small_font.render(f"Red loss {red_agent.last_loss:.3f} td {red_agent.last_td_error:.2f} eps {red_agent.epsilon:.3f}", True, (220, 220, 220)), (panel_x, 82))
        screen.blit(small_font.render(f"Blue qstates {len(blue_agent.qtable)} eps {blue_agent.epsilon:.3f}", True, (220, 220, 220)), (panel_x, 102))
        screen.blit(small_font.render(f"Red last food/death {red_last_food}/{red_last_death}", True, (255, 140, 140)), (panel_x, 124))
        screen.blit(small_font.render(f"Blue last food/death {blue_last_food}/{blue_last_death}", True, (120, 200, 255)), (panel_x, 144))
        draw_compare_graph(screen, list(red_food_history), list(blue_food_history), panel_x, 176, PANEL_WIDTH - 20, 130, "Food Per Episode", "Red food", "Blue food")
        draw_death_rate_graph(screen, list(red_death_history), list(blue_death_history), panel_x, 322, PANEL_WIDTH - 20, 130)
        screen.blit(small_font.render(f"Red save: {os.path.basename(RED_MODEL_FILE)}", True, (180, 180, 190)), (panel_x, 462))
        screen.blit(small_font.render(f"Blue save: {os.path.basename(BLUE_QTABLE_FILE)}", True, (180, 180, 190)), (panel_x, 480))

        pygame.display.update()
