# Pirate Intelligent Agent — Deep Q-Learning Treasure Hunt

**Course:** CS 370 – Current/Emerging Trends in Computer Science
**Author:** Jeffrey Wonder

## Project Overview

This project builds an intelligent agent, a pirate, that learns to find treasure in an 8×8 maze using **deep Q-learning**, a reinforcement learning technique. The agent isn't given a map or any rules about the best path. It learns by trial and error: it moves through the maze, receives small penalties for wasted moves and a reward for reaching the treasure, and gradually trains a neural network to estimate the long-term value (Q-value) of each action (left, up, right, down) from any position.

The implementation uses Python, NumPy, TensorFlow/Keras, and Matplotlib inside a Jupyter Notebook. Key reinforcement learning concepts include:

- **Epsilon-greedy exploration vs. exploitation:** the agent starts by exploring randomly (ε = 1.0) and shifts toward using what it has learned as ε decays.
- **Experience replay:** past moves are stored in a memory buffer and sampled in batches for training. This breaks the correlation between consecutive steps.
- **Target network:** a periodically synchronized copy of the model provides stable training targets.

**Results:** Training ran for 1,000 epochs (about 1.79 hours on CPU). The agent's rolling win rate rose from 0% in the first epochs to 50% by epoch ~100, reached 100% at epoch 627, and held at 100% through the final epoch. Loss fell to about 0.001.

## Repository Contents

| File | Description |
|---|---|
| `Wonder_Jeffrey_ProjectTwo.html` | Exported Jupyter Notebook with the full implementation and training output |
| `pirate-intelligent-agent/` | Project source: notebook, `TreasureMaze.py`, `GameExperience.py`, `requirements.txt` |

---

## Code Breakdown

### Starter code (provided)

The course supplied the environment, the supporting infrastructure, and the skeleton of the training function:

- **`TreasureMaze.py`:** the maze environment. It handles state representation, valid actions, the reward structure, and win/lose detection (`reset()`, `observe()`, `act()`, `valid_actions()`).
- **`GameExperience.py`:** the experience replay memory (`remember()`, `predict()`, `get_data()`), which stores episodes and produces training inputs and Q-value targets.
- **The 8×8 maze matrix** and the **`show()`** function that renders the maze, the pirate, visited cells, and the treasure with Matplotlib.
- **Global hyperparameters and action constants:** `LEFT/UP/RIGHT/DOWN`, `epsilon = 1.0`, `epsilon_min = 0.05`, `epsilon_decay = 0.995`.
- **`build_model()`:** the neural network architecture. It is a Keras `Sequential` model with two hidden `Dense` layers sized to the maze (64 units each), PReLU activations, a 4-unit output layer (one Q-value per action), the Adam optimizer, and MSE loss.
- **`train_step()`:** a `@tf.function`-compiled gradient step using `GradientTape`.
- **`play_game()`** and **`completion_check()`:** helpers that test the trained model from a given cell, or from every free cell.
- **The `qtrain()` skeleton:** option parsing, environment and target-network initialization, the experience replay object, win-history tracking, and the `format_time()` utility.

### My implementation

I wrote the core deep Q-learning training loop inside `qtrain()` (the `#START_HERE` section), following the provided pseudocode:

- **Episode loop:** each epoch resets the pirate to a random free cell. This makes the agent learn from every region of the maze instead of memorizing a single route.
- **Epsilon-greedy action selection:** with probability ε the agent takes a random valid action. Otherwise it picks the action with the highest Q-value predicted by the network. If the network picks an invalid move, the agent falls back to a random valid action.
- **Environment interaction and outcome tracking:** the loop calls `qmaze.act()`, records wins and losses in `win_history`, and ends the episode on a terminal state.
- **Experience replay integration:** each transition `[previous_state, action, reward, next_state, game_over]` is stored with `experience.remember()`. A batch is sampled with `get_data()` and passed to `train_step()`, and loss is accumulated and averaged per step.
- **Target network synchronization:** the target model's weights are copied from the main model every `target_update_freq` epochs.
- **Adaptive exploration schedule:** ε decays each epoch toward its minimum, and drops straight to 0.05 once the rolling win rate passes 90%. At that point the agent shifts from exploring to exploiting.
- **Progress reporting and early stopping:** each epoch prints loss, episodes, win count, win rate, and elapsed time. Training stops early only if the win rate reaches 100% *and* `completion_check()` confirms the pirate can win from every starting cell.
- **Training configuration and performance tuning:** I chose the hyperparameters for the training run (`n_epoch=1000`, `max_memory=8*maze.size`, `data_size=32`, `target_update_freq=50`). I also used `verbose=0` on predictions and passed tensors directly to the compiled `train_step()` to cut console overhead and speed up training.

---

## Computer Science Connection

### What do computer scientists do and why does it matter?

Computer scientists turn real-world problems into precise, computable models and then design systems that solve them reliably and at scale. That work covers algorithms, data structures, software engineering, and, more and more, machine learning systems that learn behavior from data instead of following hand-written rules. This project shows that shift on a small scale. I never told the pirate how to get through the maze. I defined the environment, the rewards, and the learning process, and the agent worked out its own strategy. The same reinforcement learning ideas drive robotics, logistics routing, recommendation engines, and autonomous systems. The work matters because software now shapes healthcare, finance, transportation, and communication, so the decisions computer scientists make about how systems learn, behave, and fail affect people's lives every day.

### How do I approach a problem as a computer scientist?

I start by understanding the problem and its constraints before I write any code. On this project, that meant reading the starter classes carefully, including how `TreasureMaze` represents state, what rewards `act()` returns, and how `GameExperience` builds training targets, so that my training loop fit the existing design. Next, I break the problem into smaller pieces: action selection, environment interaction, memory storage, training, and evaluation. I implement each piece step by step, guided by pseudocode. Then I test and iterate using evidence. I watched the loss and win-rate metrics during training, adjusted the exploration schedule and hyperparameters, and found practical bottlenecks such as prediction overhead that slowed training. Finally, I validate against the real goal rather than a proxy. A high win rate in recent games doesn't mean the maze is solved, which is why the completion check that tests every starting cell matters. That loop of analyzing, decomposing, building, measuring, and refining is how I approach computing problems in general.

### What are my ethical responsibilities to the end user and the organization?

When I build and train AI and machine learning models, my first responsibility is honesty about what a model can and cannot do. Metrics can be misleading. A 100% recent win rate can hide failures from untested starting positions, so I have to validate thoroughly and report results accurately, not just the ones that look best. My second responsibility is to the end user: models should be safe, fair, and transparent. Training data and reward functions should be reviewed for bias, user data should be protected and used only with consent, and people affected by a model's decisions should be able to understand those decisions and challenge them. A reward function also defines what an agent will optimize, so a poorly designed objective can produce harmful or unintended behavior. My third responsibility is to the organization: write maintainable, documented code; use computing resources responsibly; follow legal and regulatory requirements; and raise concerns when a system could create risk for users or the organization's reputation. Above all, I believe the people who build intelligent systems are accountable for how those systems behave, and acting on that responsibility is what earns user trust.

---

## How to Run

```bash
cd pirate-intelligent-agent
pip install -r requirements.txt
jupyter notebook
```

Open the project notebook and choose **Cell → Run All**. Training runs on CPU and can take more than an hour.
