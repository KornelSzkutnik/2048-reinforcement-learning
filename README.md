## 2048 RL Agent (TD(0))

This project contains a reinforcement learning agent for the game **2048**.  
The agent uses **Temporal Difference learning (TD(0))** with a simple feature representation built on the 2048 board.

The code is split into:
- `game_logic_2048.py` – core 2048 engine in log2 space (`Game`),
- `td_agent.py` – TD(0) agent (`TDAgent`),
- `train_agent.py` – training script,
- `board.py`, `game_2048.py`, `play_trained.py` – Pygame visualisation and manual play.

### Installation

Create (optionally) a virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Watching the trained agent

A pre-trained agent (`agent_2048.pkl`) is included in the repository,  
so you can run the demo without training anything:

```bash
python play_trained.py
```

### Training (optional)

If you want to retrain or experiment with different settings, run:

```bash
python train_agent.py
```

This will train the TD agent and save weights to `agent_2048.pkl`.  
Hyperparameters are defined at the top of `train_agent.py`.

### Manual play

To play 2048 yourself (keyboard arrows):

```bash
python game_2048.py
```

### Notes

- Dependencies and tools are listed in `requirements.txt`.   
- `Makefile` is provided for convenience on Unix-like systems, but is not required to run the code.


