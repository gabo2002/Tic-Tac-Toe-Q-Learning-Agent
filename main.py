#!/usr/bin/env python3
"""
Tic-Tac-Toe with Q-learning using matplotlib.
- Agent learns to play as 'X'. Opponent plays random 'O'.
- Trains for N episodes, tracks rolling win/draw percentages.
- Plots results with matplotlib.
- After training, lets you play against the learned agent in the terminal.
"""

import json
import random
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
EPISODES_DEFAULT = 200000
ALPHA_DEFAULT = 0.5     # learning rate (β)
GAMMA_DEFAULT = 0.95    # discount factor (γ)
EPS_START_DEFAULT = 1.0 # ε initial
EPS_END_DEFAULT = 0.05  # ε final
EPS_DECAY_DEFAULT = 0.9995  # per-episode multiplier
Q0_DEFAULT = 0.0        # initial Q value (q0)
ROLLING_WINDOW_DEFAULT = 500

# --------------------------- Game logic ---------------------------
ALL_POS = list(range(9))
LINES = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def initial_board() -> str:
    return " " * 9

def print_board(b: str) -> None:
    def cell(c):
        return c if c != ' ' else '.'
    rows = [
        f" {cell(b[0])} | {cell(b[1])} | {cell(b[2])} ",
        f" {cell(b[3])} | {cell(b[4])} | {cell(b[5])} ",
        f" {cell(b[6])} | {cell(b[7])} | {cell(b[8])} ",
    ]
    print("\n".join(rows))

def available_actions(b: str) -> List[int]:
    return [i for i,c in enumerate(b) if c == ' ']

def make_move(b: str, pos: int, player: str) -> str:
    return b[:pos] + player + b[pos+1:]

def check_winner(b: str) -> Optional[str]:
    for a,b2,c in LINES:
        line = (b[a], b[b2], b[c])
        if line == ('X','X','X'):
            return 'X'
        if line == ('O','O','O'):
            return 'O'
    if ' ' not in b:
        return 'D'
    return None

# --------------------------- Q-Learning ---------------------------
QTable = Dict[str, List[float]]

def q_for_state(q: QTable, state: str, q0: float) -> List[float]:
    if state not in q:
        q[state] = [q0]*9
    return q[state]

def epsilon_greedy_action(q: QTable, state: str, actions: List[int], epsilon: float, q0: float) -> int:
    if random.random() < epsilon:
        return random.choice(actions)
    vals = q_for_state(q, state, q0)
    return max(actions, key=lambda a: vals[a])

def random_opponent_action(state: str) -> Optional[int]:
    acts = available_actions(state)
    if not acts:
        return None
    return random.choice(acts)

def step(b: str, pos: int, player: str) -> Tuple[str, Optional[str]]:
    nb = make_move(b, pos, player)
    outcome = check_winner(nb)
    return nb, outcome

def train(
    episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    q0: float,
    rolling_window: int,
    verbose_every: int = 0,
) -> Tuple[QTable, List[Tuple[int, float, float]]]:
    Q: QTable = {}
    epsilon = eps_start
    outcomes: List[str] = []
    stats: List[Tuple[int, float, float]] = []

    for ep in range(1, episodes+1):
        state = initial_board()
        trajectory: List[Tuple[str,int]] = []

        player = 'X'
        while True:
            if player == 'X':
                acts = available_actions(state)
                if not acts:
                    outcome = check_winner(state) or 'D'
                    break
                a = epsilon_greedy_action(Q, state, acts, epsilon, q0)
                next_state, outcome = step(state, a, 'X')
                trajectory.append((state, a))
                state = next_state
                if outcome is not None:
                    break
                player = 'O'
            else:
                a_o = random_opponent_action(state)
                if a_o is None:
                    outcome = check_winner(state) or 'D'
                    break
                state, outcome = step(state, a_o, 'O')
                if outcome is not None:
                    break
                player = 'X'

        if outcome == 'X':
            r_final = 1.0
            outcomes.append('W')
        elif outcome == 'O':
            r_final = -1.0
            outcomes.append('L')
        else:
            r_final = 0.0
            outcomes.append('D')

        next_max = 0.0
        for (s, a) in reversed(trajectory):
            q_s = q_for_state(Q, s, q0)
            target = r_final + gamma * next_max
            q_s[a] = q_s[a] + alpha * (target - q_s[a])
            next_max = max(q_s)

        epsilon = max(eps_end, epsilon * eps_decay)

        if len(outcomes) > rolling_window:
            outcomes.pop(0)
        w = outcomes.count('W') / len(outcomes)
        d = outcomes.count('D') / len(outcomes)
        stats.append((ep, w*100.0, d*100.0))

        if verbose_every and ep % verbose_every == 0:
            print(f"ep {ep} | ε={epsilon:.3f} | win%={w*100:.1f} draw%={d*100:.1f}")

    return Q, stats

# --------------------------- Plot with matplotlib ---------------------------
def plot_stats(stats: List[Tuple[int,float,float]]):
    x = [ep for ep,_,_ in stats]
    win = [w for _,w,_ in stats]
    draw = [d for _,_,d in stats]
    plt.figure(figsize=(10,5))
    plt.plot(x, win, label="Win %", color="blue")
    plt.plot(x, draw, label="Draw %", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Percentage")
    plt.title("Training performance (rolling window)")
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------- Play vs Agent ---------------------------
def greedy_best_action(q: QTable, state: str, q0: float) -> int:
    acts = available_actions(state)
    vals = q_for_state(q, state, q0)
    return max(acts, key=lambda a: vals[a])

def human_turn(state: str, who: str) -> str:
    print_board(state)
    prompt = (
        "Scegli una posizione 1-9 (numerazione):\n"
        " 1 | 2 | 3\n 4 | 5 | 6\n 7 | 8 | 9\n> "
    )
    while True:
        try:
            s = input(prompt).strip()
            if s.lower() in {"q","quit","exit"}:
                sys.exit(0)
            pos = int(s) - 1
            if pos not in range(9):
                raise ValueError
            if state[pos] != ' ':
                print("Casella occupata, riprova.")
                continue
            return make_move(state, pos, who)
        except ValueError:
            print("Inserisci un numero tra 1 e 9.")

def ai_turn(state: str, who: str, q: QTable, q0: float) -> str:
    a = greedy_best_action(q, state, q0)
    return make_move(state, a, who)

def play_loop(q: QTable, q0: float) -> None:
    print("\nGiochiamo a Tris! L'agente gioca come 'X'. Tu sei 'O'.")
    while True:
        b = initial_board()
        current = 'X'
        while True:
            if current == 'X':
                if not available_actions(b):
                    break
                b = ai_turn(b, 'X', q, q0)
                outcome = check_winner(b)
                if outcome:
                    print_board(b)
                    if outcome == 'X':
                        print("Ha vinto l'agente (X).")
                    elif outcome == 'O':
                        print("Hai vinto tu (O)!")
                    else:
                        print("Pareggio.")
                    break
                current = 'O'
            else:
                b = human_turn(b, 'O')
                outcome = check_winner(b)
                if outcome:
                    print_board(b)
                    if outcome == 'X':
                        print("Ha vinto l'agente (X).")
                    elif outcome == 'O':
                        print("Hai vinto tu (O)!")
                    else:
                        print("Pareggio.")
                    break
                current = 'X'
        again = input("Giocare di nuovo? (s/n): ").strip().lower()
        if again != 's':
            break

# --------------------------- Main ---------------------------
def main():
    episodes = EPISODES_DEFAULT
    alpha = ALPHA_DEFAULT
    gamma = GAMMA_DEFAULT
    eps_start = EPS_START_DEFAULT
    eps_end = EPS_END_DEFAULT
    eps_decay = EPS_DECAY_DEFAULT
    q0 = Q0_DEFAULT
    rolling = ROLLING_WINDOW_DEFAULT

    Q, stats = train(episodes, alpha, gamma, eps_start, eps_end, eps_decay, q0, rolling, verbose_every=episodes//10)
    plot_stats(stats)

    play = input("Vuoi giocare contro l'agente? (s/n): ").strip().lower()
    if play == 's':
        play_loop(Q, q0)

if __name__ == '__main__':
    main()