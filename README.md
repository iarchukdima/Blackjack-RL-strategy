# Blackjack RL Strategy

Reinforcement learning for Blackjack using the **REINFORCE** (Monte Carlo Policy Gradient) algorithm. Includes a custom environment, training pipeline, baseline comparison, and pygame visualization.

## Project Structure

| File | Description |
|------|-------------|
| `env.py` | Blackjack environment (finite deck, hit/stay actions) |
| `reinforce.py` | REINFORCE training, policy network, evaluation |
| `baseline.py` | Baseline policies (random, hit-below-17, basic strategy) |
| `blackjack.py` | Pygame visualization with AI suggestions |

## Requirements

```
numpy
torch
pygame
```

```bash
pip install numpy torch pygame
```

## Quick Start

### 1. Train the policy

```bash
python reinforce.py
```

Trains for ~70k–200k episodes, saves `policy.pt`, and logs sample episodes.

### 2. Play with visualization

```bash
python blackjack.py
```

- **H** / **S** or click: Hit / Stay  
- **A**: Take AI-suggested action  
- Session win rate shown in top right

### 3. Compare with baselines

```bash
python baseline.py
```

Evaluates random, hit-below-17, basic strategy, and trained policy over 10k episodes.

## Environment

- **State**: 22-dim vector
  - 10 dealer card counts (Ace–10)
  - 10 player card counts
  - `player_sum/21`, `dealer_sum/21`
- **Actions**: 0 = stay, 1 = hit
- **Rewards**: +1 win, -1 loss, 0 tie
- **Deck**: Finite (configurable number of decks), reset each episode

## REINFORCE Details

- **Policy**: MLP with sigmoid output → P(hit)
- **Features**: Entropy regularization, optional running-mean baseline, logit clamping
- **Training**: Per-episode updates, Adam optimizer

## Typical Results

| Policy | Win rate (10k eps) |
|--------|--------------------|
| Random | ~-0.35 |
| Hit below 17 | ~-0.03 |
| Basic strategy | ~+0.015 |
| REINFORCE (trained) | ~+0.019 |

(Higher is better. Win rate = (wins − losses) / total.)

## License

MIT

