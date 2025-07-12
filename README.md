# Treasure Hunt – Q-Learning Agent

## Overview
This project demonstrates a reinforcement learning agent that learns to solve a grid-based treasure hunt maze using the Q-learning algorithm combined with a neural network function approximator.  

The agent starts from random positions and learns optimal policies through repeated training episodes using an epsilon-greedy strategy and experience replay.  

---

## Key Features
- **Model-Free Q-Learning:** Learns directly from interactions without requiring a model of the environment.
- **Deep Neural Network Approximation:** A Keras-based network approximates Q-values.
- **Experience Replay:** Stores past episodes for more stable training.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation to optimize learning.
- **Custom Environment:** Grid-based maze environment designed to test pathfinding behavior.

---

## How It Works

- **Environment:**
  An 8×8 NumPy matrix represents the maze. Cells can be free or blocked. The agent must learn to reach the treasure cell.

- **Agent Training Loop:**
  1. Initialize agent in a random free cell.
  2. While the game is not over:
     - Select an action:
       - **Exploration** (random action)
       - **Exploitation** (predict best action from model)
     - Apply action, get next state and reward.
     - Store the experience.
     - Train the model on replayed experiences.
  3. Monitor win rate and stop training when performance reaches a threshold.

- **Deep Q-Network (DQN):**
  - Implemented in Keras with Dense layers and PReLU activation.
  - Optimized with Adam optimizer and Mean Squared Error loss.

---

## Screenshots

---

### Model Training in Jupyter Notebook

This screenshot shows the training process of the Deep Q-Learning agent over multiple epochs. The debug logs display each epoch's step time and loss value as the neural network improves its policy.

<img src="assets/treasurehunt%20(1).png" alt="Q-Learning Training Screenshot" width="700"/>

---

### Environment and Model Initialization

This screenshot shows the setup of the environment matrix (the maze), the import statements for Keras and NumPy, and the helper functions to visualize the maze structure.

<img src="assets/treasurehunt%20(2).png" alt="Jupyter Notebook Environment Setup Screenshot" width="700"/>

---

## Example Code Snippet

```python
if (cur_epoch % (epsilon * 100)) == 0:
    action = random.choice(valid_actions)
else:
    action = np.argmax(experience.predict(previous_envstate))
