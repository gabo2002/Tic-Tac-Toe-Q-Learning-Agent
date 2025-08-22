# Tic-Tac-Toe Q-Learning Agent

This project implements a Tic-Tac-Toe (Tris) game in Python, where an agent learns to play optimally using the Q-learning algorithm. The agent is trained to play as 'X', while the opponent ('O') plays random moves. After training, you can play against the learned agent directly in the terminal.

## Features

- **Q-Learning**: The agent uses Q-learning to learn optimal moves over thousands of episodes.
- **Customizable Training**: Training parameters (episodes, learning rate, discount factor, epsilon, etc.) are configurable.
- **Performance Tracking**: Tracks and plots rolling win and draw percentages during training using matplotlib.
- **Interactive Play**: After training, you can play against the trained agent in the terminal.
- **Simple Interface**: The board is displayed in a human-readable format, and moves are selected via keyboard input.

## How It Works

1. **Training Phase**:
    - The agent ('X') plays against a random opponent ('O') for a specified number of episodes.
    - Q-values are updated after each episode based on the outcome (win, lose, draw).
    - The agent explores moves using an epsilon-greedy strategy, with epsilon decaying over time.
    - Training statistics (win/draw rates) are collected and plotted.

2. **Playing Phase**:
    - After training, you are prompted to play against the agent.
    - The agent uses the learned Q-table to select the best moves.
    - You play as 'O' by entering positions (1-9) on the board.

## Usage

1. **Install Requirements**  
   Make sure you have Python 3 and `matplotlib` installed:
   ```sh
   pip install matplotlib
   ```

2. **Run the Program**
   ```sh
   python main.py
   ```

3. **Follow Prompts**
    - The program will train the agent and display a plot of training performance.
    - After training, you can choose to play against the agent.

## Controls

- When playing, enter a number from 1 to 9 to select your move:
  ```
   1 | 2 | 3
   4 | 5 | 6
   7 | 8 | 9
  ```
- Enter `q`, `quit`, or `exit` to quit the game.

## File Structure

- `main.py` — Main script containing all logic for training, playing, and plotting.

## License

This project is provided for educational purposes.

---

*Enjoy playing and experimenting with reinforcement learning!*