import numpy as np
from typing import List

## actions: stay/draw card/   double down/split



class BlackjackEnv:
    def __init__(self, number_of_decks: int=1):

        self.number_of_decks = number_of_decks
        self.reset()

    def draw_card(self) -> int:
        card = self.cards[0]
        self.cards = self.cards[1:]
        return card

    def get_value(self, hand: List[int], remove_busted: bool=True) -> np.ndarray:
        
        assert len(hand) > 0
        values = np.array([0])
        for card in hand:
            if card != 1:
                values += card
            else:
                values1 = values + 1
                values2 = values + 11
                values = np.concatenate((values1, values2))

        if remove_busted:
            values = values[values <= 21]

        return values

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Build fresh deck for this episode
        k = 4 * self.number_of_decks
        self.cards = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9] * k + [10] * k * 4
        )
        np.random.shuffle(self.cards)
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]
        return self._state_to_vector()

    def _state_to_vector(self) -> np.ndarray:
        """State: [dealer_card_counts (10), player_card_counts (10)]."""
        dealer_counts = np.zeros(10)
        player_counts = np.zeros(10)
        for c in self.dealer:
            dealer_counts[c - 1] += 1  # card 1-10 -> index 0-9
        for c in self.player:
            player_counts[c - 1] += 1
        return np.concatenate([dealer_counts, player_counts]).astype(np.float32)

    def step(self, action):
        # 0 = stay, 1 = hit
        assert action in [0, 1]
        if action == 1:
            self.player.append(self.draw_card())
            scores_player = self.get_value(self.player)
            if len(scores_player) == 0:  # bust
                return self._state_to_vector(), -1.0, True, {}
            return self._state_to_vector(), 0.0, False, {}

        if action == 0:
            # dealer takes
            scores_dealer = self.get_value(self.dealer)
            while len(scores_dealer[scores_dealer < 17]) > 0:
                self.dealer.append(self.draw_card())
                scores_dealer = self.get_value(self.dealer)

            scores_player = self.get_value(self.player)
            scores_dealer = self.get_value(self.dealer)

            player_best = np.max(scores_player) if len(scores_player) > 0 else 0
            dealer_best = np.max(scores_dealer) if len(scores_dealer) > 0 else 0

            if len(scores_dealer) == 0 or player_best > dealer_best:
                reward = 1.0
            elif player_best < dealer_best:
                reward = -1.0
            else:
                reward = 0.0

            return self._state_to_vector(), reward, True, {}
