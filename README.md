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

---

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

---

## Problem Formulation

We model Blackjack as a **Markov Decision Process (MDP)**:

$$
v^{\pi^{\theta}}(s) := \mathbb{E} \left[ \sum_{t=0}^{T-1} \gamma^t R_t \mid S_0 = s \right] \rightarrow \max_{\pi}, \quad s \in \mathcal{S}
$$

subject to:

$$
s_{t+1} \sim p(\cdot \mid s_t, a_t), \quad
a_t \sim \pi(\cdot \mid s_t), \quad
r_t \sim p^R(\cdot \mid s_t, a_t)
$$

Where:

- $s_t$ — state at time step $t$
- $a_t$ — action chosen at time $t$
- $r_t$ — reward received at time $t$
- $\gamma \in [0,1]$ — discount factor
- $\pi(a|s)$ — policy

The objective is to learn a policy $\pi$ that maximizes the **expected return**.

---

## Environment

### State: 
Each state $s \in \mathcal{S}$ is represented as:

$$
s_{1\times22} = (p_{1\times10}, d_{1\times10}, \Sigma_p, \Sigma_d)
$$

where:

- $p_{1\times10}$ : player’s count of cards of each value (10, J, Q, K $\to$ 10, A $\to$ {1, 11})
- $d_{1\times10}$: dealer’s visible cards  
- $\Sigma_p$: Sum value of player's cards  
- $\Sigma_d$: Sum value of dealer's cards  

### Actions: 
$$
\mathcal{A} = \{0, 1\}
$$

where:

- $0$ = **stay** (pass)  
- $1$ = **hit** (take one more card)
### Rewards: 
Rewards are only given at the end of the episode:

$$
r =
\begin{cases}
+1 & \text{if player wins} \\
0 & \text{if tie} \\
-1 & \text{if player loses}
\end{cases}
$$

### Deck: 
Number of decks is configurable one by default. It is being reset after each episode.

---

## Policy Model

We approximate the policy $\pi_\theta(a|s)$ using a **MLP**:

$$
p_\theta(s) = \sigma(f_\theta(s))
$$

Where:

- $f_\theta(s)$ is a feed-forward neural network  
- $\sigma$ is the sigmoid function  

The output is the **probability of taking action "hit"**.

---

## 📈 Learning Algorithm: REINFORCE

We use the Monte-Carlo Policy Gradient **(REINFORCE)**:

$$
\nabla_\theta J(\theta) =
\mathbb{E}_\pi \left[ G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right]
$$

Where:

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_k
$$

As the rewards are given only at the end of the game, we set $\gamma=1$.

The loss we minimize:

$$
\mathcal{L} = - G_t \log \pi_\theta(A_t \mid S_t)
$$


### Features: 
- Entropy regularization
- logit clamping
- optional running-mean baseline 

---

## Results

We are comparing Reinforced learning approach to several baselines:

- Random: Randomly sample action with equal probability of pass and hit

- Hit below 17: If sum value < 17 $\to$ hit, otherwise pass

- Basic strategy:  
    - Always hit on 11 or less
    - Always pass on 17+
    - 12-16: hit if dealer upcard is 7 or higher (dealer is likely strong)
    - Soft hands (`A` with value 11): hit on 17 or less

| Policy | Win rate (10k eps) | Expected reward |
|--------|--------------------|-----------------|
| Random | 33% | $\approx$ -0.35 |
| Hit below 17 | 48% | $\approx$ -0.03 |
| Basic strategy | 57% | $\approx$ +0.015 |
| **REINFORCE** (trained) | **60%** | $\approx$ **+0.019** |

Expected reward = $\frac{wins − losses}{total}$.

The reinforced policy outperforms all the baselines, providing the highest winrate. 

## License

MIT
