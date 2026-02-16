"""
Baseline policies for Blackjack. Compare with trained REINFORCE policy.
"""

import numpy as np

from env import BlackjackEnv


def get_hand_value(hand, get_value_fn) -> int:
    """Best hand value (0 if bust)."""
    scores = get_value_fn(hand)
    return int(np.max(scores)) if len(scores) > 0 else 0


def is_soft_hand(hand, get_value_fn) -> bool:
    """True if hand has ace counting as 11 (soft hand)."""
    if 1 not in hand:
        return False
    scores = get_value_fn(hand)
    return any(s > 11 for s in scores)  # has a value > 11 means ace as 11


def random_baseline(env: BlackjackEnv) -> int:
    """50/50 hit or stay."""
    return 1 if np.random.rand() < 0.5 else 0


def hit_below_17_baseline(env: BlackjackEnv) -> int:
    """Hit if player sum < 17, else stay."""
    player_sum = get_hand_value(env.player, env.get_value)
    return 1 if player_sum < 17 else 0


def basic_strategy_baseline(env: BlackjackEnv) -> int:
    """
    Simplified basic strategy (no splitting/doubling).
    - Always hit on 11 or less
    - Always stay on 17+
    - 12-16: hit if dealer upcard is 7 or higher (dealer likely strong)
    - Soft hands (A+6 etc): hit on soft 17 or less
    """
    player_sum = get_hand_value(env.player, env.get_value)
    dealer_upcard = min(env.dealer[0], 10)  # 1-10
    soft = is_soft_hand(env.player, env.get_value)

    if player_sum <= 11:
        return 1  # always hit
    if player_sum >= 17:
        return 0  # always stay

    # 12-16: depends on dealer
    if dealer_upcard >= 7:
        return 1  # dealer likely strong, take a card
    if dealer_upcard <= 6:
        return 0  # dealer likely to bust, stay

    # Soft 12-16: generally hit to improve
    if soft and player_sum <= 16:
        return 1

    return 0


def evaluate_baseline(
    env: BlackjackEnv,
    baseline_fn,
    n_episodes: int = 10_000,
    seed: int = 42,
) -> float:
    """Evaluate a baseline policy. Returns mean reward per episode."""
    np.random.seed(seed)
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        ep_reward = 0
        while True:
            action = baseline_fn(env)
            _, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def main():
    env = BlackjackEnv(number_of_decks=1)
    n_episodes = 10_000
    seed = 42

    print("=== Baseline Comparison (10k episodes) ===\n")

    # Random
    r_random = evaluate_baseline(env, random_baseline, n_episodes, seed)
    print(f"Random (50/50):        {r_random:+.4f}")

    # Hit below 17
    r_hit17 = evaluate_baseline(env, hit_below_17_baseline, n_episodes, seed)
    print(f"Hit below 17:          {r_hit17:+.4f}")

    # Basic strategy
    r_basic = evaluate_baseline(env, basic_strategy_baseline, n_episodes, seed)
    print(f"Basic strategy:        {r_basic:+.4f}")

    # Trained policy (if available)
    try:
        import torch
        from reinforce import Policy
        import os
        path = os.path.join(os.path.dirname(__file__), "policy.pt")
        if os.path.exists(path):
            policy = Policy(state_dim=22, hidden_dim=128)
            policy.load_state_dict(torch.load(path, map_location="cpu"))
            policy.eval()

            def policy_baseline(env):
                state = env._state_to_vector()
                with torch.no_grad():
                    prob = policy(torch.from_numpy(state).float().unsqueeze(0)).item()
                return 1 if prob > 0.5 else 0

            r_policy = evaluate_baseline(env, policy_baseline, n_episodes, seed)
            print(f"REINFORCE (trained):   {r_policy:+.4f}")
        else:
            print("REINFORCE (trained):   (run reinforce.py first to train)")
    except Exception as e:
        print(f"REINFORCE (trained):   (skip: {e})")


if __name__ == "__main__":
    main()
