"""
REINFORCE (Monte Carlo Policy Gradient) for Blackjack.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from env import BlackjackEnv


class Policy(nn.Module):
    """MLP policy: state -> logit -> sigmoid = P(action=1)."""

    def __init__(self, state_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(action=1) via sigmoid."""
        logit = self.net(x).squeeze(-1)
        return torch.sigmoid(logit)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, torch.Tensor]:
        """Sample action and return (action, log_prob). action 1 = hit."""
        x = torch.from_numpy(state).float().unsqueeze(0)
        prob = self.forward(x).squeeze(0)
        dist = Bernoulli(probs=prob)

        if deterministic:
            action = 1 if prob.item() > 0.5 else 0
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32))
        return action, log_prob


def train_reinforce(
    env: BlackjackEnv,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    n_episodes: int = 150_000,
    gamma: float = 1.0,
    use_baseline: bool = False,
    baseline_decay: float = 0.99,
    seed: int | None = None,
) -> list[float]:
    """Train policy with REINFORCE. Optional baseline (running mean of returns)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    rewards_history: list[float] = []
    baseline: float | None = None

    for episode in range(n_episodes):
        state = env.reset(seed=seed if episode == 0 else None)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        while True:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break
            state = next_state

        # Compute returns: G_t = sum_{k>=t} gamma^{k-t} r_k
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        ep_return = sum(rewards)
        rewards_history.append(ep_return)

        if use_baseline:
            if baseline is None:
                baseline = float(ep_return)
            else:
                baseline = baseline_decay * baseline + (1 - baseline_decay) * ep_return
            returns_t = torch.tensor(
                [r - baseline for r in returns], dtype=torch.float32
            )
        else:
            returns_t = torch.tensor(returns, dtype=torch.float32)

        loss = -(sum(lp * G for lp, G in zip(log_probs, returns_t)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 5000 == 0:
            avg = np.mean(rewards_history[-5000:])
            bl = f", baseline={baseline:.4f}" if use_baseline and baseline is not None else ""
            print(f"Episode {episode + 1}, avg reward (last 5k): {avg:.4f}{bl}")

    return rewards_history


def log_episodes(
    env: BlackjackEnv,
    policy: Policy,
    n_episodes: int = 20,
    seed: int = 42,
) -> None:
    """Log n_episodes with full trajectories for inspection."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            state = env.reset()
            steps = []
            while True:
                action, _ = policy.get_action(state, deterministic=True)
                prob = policy.forward(
                    torch.from_numpy(state).float().unsqueeze(0)
                ).item()
                steps.append({
                    "player": list(env.player),
                    "dealer": list(env.dealer),
                    "action": "hit" if action == 1 else "stay",
                    "P(hit)": prob,
                })
                state, reward, done, _ = env.step(action)
                if done:
                    steps[-1]["reward"] = reward
                    steps[-1]["final_player"] = list(env.player)
                    steps[-1]["final_dealer"] = list(env.dealer)
                    break

            outcome = "WIN" if reward == 1 else ("LOSS" if reward == -1 else "TIE")
            print(f"\n--- Episode {ep + 1} ({outcome}, reward={reward}) ---")
            for i, s in enumerate(steps):
                r_str = f" -> reward={s['reward']}" if "reward" in s else ""
                print(f"  step {i + 1}: player={s['player']} dealer={s['dealer']} "
                      f"action={s['action']} P(hit)={s['P(hit)']:.3f}{r_str}")
                if "final_dealer" in s:
                    print(f"         final: player={s['final_player']} dealer={s['final_dealer']}")

    policy.train()


def evaluate(env: BlackjackEnv, policy: Policy, n_episodes: int = 10_000, seed: int = 42) -> float:
    """Evaluate policy deterministically."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            ep_reward = 0
            while True:
                action, _ = policy.get_action(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
    policy.train()
    return float(np.mean(rewards))


def main():
    env = BlackjackEnv(number_of_decks=1)
    policy = Policy(state_dim=20, hidden_dim=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    print("Training REINFORCE (150k episodes, hidden=128)...")
    train_reinforce(
        env,
        policy,
        optimizer,
        n_episodes=150_000,
        gamma=1.0,
        use_baseline=False,  # set True for variance reduction
        baseline_decay=0.9,
        seed=42,
    )

    win_rate = evaluate(env, policy, n_episodes=10_000, seed=123)
    print(f"\nEvaluated win rate (10k episodes): {win_rate:.4f}")
    print("(Random policy ≈ -0.05 to 0.0)")

    print("\n--- Logging 20 sample episodes ---")
    log_episodes(env, policy, n_episodes=20, seed=456)

    # Save policy for blackjack.py visualization
    torch.save(policy.state_dict(), "policy.pt")
    print("\nPolicy saved to policy.pt (run blackjack.py to visualize)")


if __name__ == "__main__":
    main()