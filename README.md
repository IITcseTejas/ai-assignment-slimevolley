# AI Assignment 3 – Game Playing using Minimax and Alpha-Beta Pruning

Tejas Meshram cs24m108
Ajitesh Gowlikar cs24m118
🎓 **IIT Tirupati | M.Tech CSE**  
📅 **Spring 2025**  
 

---

## 🎮 Environment: Slime Volleyball

We implemented two classical game-playing algorithms in the [SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym) environment using OpenAI Gym.

---

## ✅ Algorithms Implemented

| Agent             | Description                                                   |
|------------------|---------------------------------------------------------------|
| 🔁 **Minimax Agent**   | Standard depth-limited Minimax search algorithm               |
| ⚡ **Alpha-Beta Agent** | Optimized version with Alpha-Beta pruning for faster decision-making |

Both agents play against a random opponent and make decisions using a domain-specific evaluation function.

---

## 🧠 Evaluation Function

We designed a heuristic that evaluates the environment state based on:

- ✅ Distance between agent and ball  
- ✅ Ball's horizontal velocity (favoring offensive direction)  
- ✅ Score difference between agent and opponent

```python
dist = abs(agent_x - ball_x)
direction_bonus = 5 if ball_vx > 0 else -5
score_reward = (agent_score - opponent_score) * 100
return -dist + direction_bonus + score_reward
```

---

## 🖥️ Installation & Setup (Tested on WSL – Ubuntu)

### 🔧 Step-by-step Setup

```bash
# System setup
sudo apt update
sudo apt install python3-pip python3-venv ffmpeg -y

# Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# Package installation
pip install slimevolleygym gym==0.21.0 numpy==1.24.4 pyglet==1.5.27 box2d-py opencv-python
```

### 🎮 Run Basic Test

```bash
python test_env.py
```

✅ A game window should appear briefly to confirm setup.

---

## 🚀 How to Run Agents

### 🕹️ Run Minimax Agent

```bash
python play_minimax.py
```

### ⚡ Run Alpha-Beta Agent

```bash
python play_alphabeta.py
```

---

## 📹 Record Gameplay Videos

To record one game for each agent:

```bash
python record_games.py
```

Videos will be saved to:

```
videos/
├── minimax/
│   └── rl-video-episode-0.mp4
└── alphabeta/
    └── rl-video-episode-0.mp4
```

---

## 📂 Project Structure

```
slime-ai/
├── agents/
│   ├── minimax_agent.py
│   └── alphabeta_agent.py
├── videos/
│   ├── minimax/
│   └── alphabeta/
├── play_minimax.py
├── play_alphabeta.py
├── record_games.py
├── test_env.py
├── README.md

    
```

---

