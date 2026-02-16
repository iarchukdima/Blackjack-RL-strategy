import os
import pygame
import random
import sys
import numpy as np

# Optional: load trained REINFORCE policy for AI suggestions
POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.pt")
policy = None
try:
    import torch
    from reinforce import Policy as ReinforcePolicy
    if os.path.exists(POLICY_PATH):
        policy = ReinforcePolicy(state_dim=20, hidden_dim=128)
        policy.load_state_dict(torch.load(POLICY_PATH, map_location="cpu"))
        policy.eval()
        print("Loaded trained policy from policy.pt")
except Exception as e:
    policy = None
    print("No policy loaded (run reinforce.py first):", e)

pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 720
CARD_WIDTH = 80
CARD_HEIGHT = 120
CARD_MARGIN = 10

# Colors
GREEN_TABLE = (0, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GOLD = (255, 215, 0)
DARK_GREEN = (0, 80, 0)
BUTTON_GREEN = (0, 150, 0)
BUTTON_HOVER = (0, 180, 0)

# Card values: 1=Ace, 2-10, 11=Jack, 12=Queen, 13=King
CARD_NAMES = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
              8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

CARD_REVEAL_DELAY = 350       # ms between cards (hit, dealer draw)
CARD_REVEAL_DELAY_DEAL = 150  # ms between initial 4 cards (faster)


def hands_to_state(dealer_hand, player_hand):
    """Convert blackjack.py hands (1-13) to 20-dim policy state. Cards 11,12,13 -> 10."""
    dealer_counts = [0] * 10
    player_counts = [0] * 10
    for c in dealer_hand:
        v = min(c, 10)  # J,Q,K -> 10
        dealer_counts[v - 1] += 1
    for c in player_hand:
        v = min(c, 10)
        player_counts[v - 1] += 1
    return np.array(dealer_counts + player_counts, dtype=np.float32)


def get_policy_action(dealer_hand, player_hand):
    """Return (action, P(hit)) from trained policy. action: 1=hit, 0=stay."""
    if policy is None:
        return None, None
    state = hands_to_state(dealer_hand, player_hand)
    with torch.no_grad():
        prob = policy(torch.from_numpy(state).float().unsqueeze(0)).item()
    action = 1 if prob > 0.5 else 0
    return action, prob


def create_deck():
    """Create a standard 52-card deck. Values: 1-13 (Ace=1, J=11, Q=12, K=13)."""
    deck = []
    for _ in range(4):  # 4 suits
        for value in range(1, 14):
            deck.append(value)
    return deck


def get_hand_value(hand):
    """Calculate best hand value. Ace = 1 or 11 (whichever is better), face cards = 10."""
    values = [min(c, 10) for c in hand]  # Face cards count as 10, Ace = 1
    value = sum(values)
    aces = hand.count(1)
    # Treat each Ace as 11 initially (add 10 per ace)
    value += aces * 10
    # If bust, switch Aces from 11 to 1 (subtract 10 each time)
    while value > 21 and aces > 0:
        value -= 10
        aces -= 1
    return value


def draw_card(surface, x, y, value, face_down=False):
    """Draw a card at position (x, y). value: 1=Ace, 2-10, 11=J, 12=Q, 13=K."""
    card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
    pygame.draw.rect(surface, WHITE, card_rect)
    pygame.draw.rect(surface, BLACK, card_rect, 2)
    
    if face_down:
        inner = pygame.Rect(x + 5, y + 5, CARD_WIDTH - 10, CARD_HEIGHT - 10)
        pygame.draw.rect(surface, (50, 50, 150), inner)
        return
    
    display_val = CARD_NAMES.get(value, str(value))
    color = RED if value in (1, 11, 12, 13) else BLACK
    
    font = pygame.font.Font(None, 48)
    text = font.render(display_val, True, color)
    text_rect = text.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT // 2))
    surface.blit(text, text_rect)


class BlackjackGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        caption = 'Blackjack (AI)' if policy is not None else 'Blackjack (run reinforce.py first to train)'
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)
        self.reset()
    
    def reset(self):
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.player_turn = True
        self.dealer_turn = False
        self.game_over = False
        self.message = ''
        # Card reveal animation: dealer, player, dealer, player (first 4 use faster delay)
        self.visible_dealer = 0
        self.visible_player = 0
        self.reveal_queue = ['dealer', 'player', 'dealer', 'player']
        self.last_reveal_time = pygame.time.get_ticks()
        self.reveals_processed = 0
    
    def draw_card(self):
        """Draw a card from the deck. Returns value 1-13."""
        if not self.deck:
            self.deck = create_deck()
            random.shuffle(self.deck)
        return self.deck.pop()
    
    def hit(self):
        if not self.player_turn or self.game_over or self.reveal_queue:
            return
        self.player_hand.append(self.draw_card())
        if get_hand_value(self.player_hand) > 21:
            self.visible_player = len(self.player_hand)  # Show bust card immediately
            self.player_turn = False
            self.game_over = True
            self.message = 'Bust! You lose.'
        else:
            self.reveal_queue.append('player')
    
    def stay(self):
        if not self.player_turn or self.game_over or self.reveal_queue:
            return
        self.player_turn = False
        self.dealer_turn = True
        # Reveal dealer's 2nd card first (it was face down)
        self.reveal_queue.append('dealer')
    
    def determine_winner(self):
        player_val = get_hand_value(self.player_hand)
        dealer_val = get_hand_value(self.dealer_hand)
        
        if dealer_val > 21:
            self.message = 'Dealer busts! You win!'
        elif player_val > dealer_val:
            self.message = 'You win!'
        elif player_val < dealer_val:
            self.message = 'Dealer wins!'
        else:
            self.message = "It's a push (tie)!"
    
    def draw_buttons(self, disabled=False):
        hit_rect = pygame.Rect(200, 600, 120, 50)
        stay_rect = pygame.Rect(480, 600, 120, 50)
        
        disabled = bool(disabled)
        btn_color = (80, 80, 80) if disabled else BUTTON_GREEN
        hover_color = (100, 100, 100) if disabled else BUTTON_HOVER
        
        mouse = pygame.mouse.get_pos()
        hit_hover = not disabled and hit_rect.collidepoint(mouse)
        stay_hover = not disabled and stay_rect.collidepoint(mouse)
        
        pygame.draw.rect(self.screen, hover_color if hit_hover else btn_color, hit_rect)
        pygame.draw.rect(self.screen, BLACK, hit_rect, 2)
        hit_text = self.font.render('HIT', True, WHITE)
        self.screen.blit(hit_text, (hit_rect.centerx - hit_text.get_width() // 2, hit_rect.centery - 10))
        
        pygame.draw.rect(self.screen, hover_color if stay_hover else btn_color, stay_rect)
        pygame.draw.rect(self.screen, BLACK, stay_rect, 2)
        stay_text = self.font.render('STAY', True, WHITE)
        self.screen.blit(stay_text, (stay_rect.centerx - stay_text.get_width() // 2, stay_rect.centery - 10))
        
        return hit_rect, stay_rect
    
    def handle_click(self, pos):
        if self.game_over:
            new_rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, 570, 160, 50)
            if new_rect.collidepoint(pos):
                self.reset()
            return
        hit_rect, stay_rect = pygame.Rect(200, 600, 120, 50), pygame.Rect(480, 600, 120, 50)
        if hit_rect.collidepoint(pos):
            self.hit()
        elif stay_rect.collidepoint(pos):
            self.stay()
    
    def run(self):
        running = True
        while running:
            now = pygame.time.get_ticks()
            
            # Process reveal queue (cards appearing with delay)
            delay = CARD_REVEAL_DELAY_DEAL if self.reveals_processed < 4 else CARD_REVEAL_DELAY
            if self.reveal_queue and now - self.last_reveal_time >= delay:
                target = self.reveal_queue.pop(0)
                if target == 'dealer':
                    self.visible_dealer = min(self.visible_dealer + 1, len(self.dealer_hand))
                else:
                    self.visible_player = min(self.visible_player + 1, len(self.player_hand))
                self.last_reveal_time = now
                self.reveals_processed += 1
            
            # Dealer draws (after reveals are done)
            if (self.dealer_turn and not self.reveal_queue and
                    self.visible_dealer == len(self.dealer_hand)):
                if get_hand_value(self.dealer_hand) < 17:
                    self.dealer_hand.append(self.draw_card())
                    self.reveal_queue.append('dealer')
                else:
                    self.dealer_turn = False
                    self.game_over = True
                    self.determine_winner()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h and not self.game_over and not self.reveal_queue:
                        self.hit()
                    elif event.key == pygame.K_s and not self.game_over and not self.reveal_queue:
                        self.stay()
                    elif event.key == pygame.K_a and not self.game_over and not self.reveal_queue and policy is not None:
                        action, _ = get_policy_action(self.dealer_hand, self.player_hand)
                        if action == 1:
                            self.hit()
                        else:
                            self.stay()
                    elif event.key == pygame.K_n and self.game_over:
                        self.reset()
            
            # Draw
            self.screen.fill(GREEN_TABLE)
            
            # Title
            title = self.big_font.render('BLACKJACK', True, GOLD)
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
            
            # Dealer hand (only show visible cards)
            dealer_label = self.font.render('Dealer', True, WHITE)
            self.screen.blit(dealer_label, (50, 80))
            for i in range(self.visible_dealer):
                card = self.dealer_hand[i]
                x = 50 + i * (CARD_WIDTH + CARD_MARGIN)
                face_down = self.player_turn and i == 1  # Hide dealer's 2nd card during player turn
                draw_card(self.screen, x, 110, card, face_down)
            # Show dealer value (only first card when 2nd is hidden)
            if self.visible_dealer >= 1:
                visible_dealer_cards = self.dealer_hand[:1] if (self.player_turn and self.visible_dealer >= 2) else self.dealer_hand[:self.visible_dealer]
                dealer_val = get_hand_value(visible_dealer_cards)
                val_text = self.font.render(f'Value: {dealer_val}', True, WHITE)
                self.screen.blit(val_text, (50, 250))
            
            # Player hand (only show visible cards)
            player_label = self.font.render('Player', True, WHITE)
            self.screen.blit(player_label, (50, 320))
            for i in range(self.visible_player):
                card = self.player_hand[i]
                x = 50 + i * (CARD_WIDTH + CARD_MARGIN)
                draw_card(self.screen, x, 350, card, False)
            player_val = get_hand_value(self.player_hand[:self.visible_player]) if self.visible_player else 0
            val_text = self.font.render(f'Value: {player_val}', True, WHITE)
            self.screen.blit(val_text, (50, 490))
            
            # AI policy suggestion (during player turn, no reveal queue)
            if (not self.game_over and self.player_turn and not self.reveal_queue and
                    policy is not None):
                action, prob = get_policy_action(self.dealer_hand, self.player_hand)
                if action is not None:
                    suggest = "HIT" if action == 1 else "STAY"
                    ai_text = self.font.render(
                        f"AI suggests: {suggest}  (P(hit)={prob:.2f})", True, GOLD
                    )
                    self.screen.blit(ai_text, (50, 530))
                    hint = self.font.render("Press A to take AI action", True, (180, 180, 180))
                    self.screen.blit(hint, (50, 555))
            
            # Buttons
            if not self.game_over:
                self.draw_buttons(self.reveal_queue)
            else:
                msg_text = self.big_font.render(self.message, True, GOLD)
                self.screen.blit(msg_text, (WINDOW_WIDTH // 2 - msg_text.get_width() // 2, 500))
                new_rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, 570, 160, 50)
                pygame.draw.rect(self.screen, BUTTON_GREEN, new_rect)
                pygame.draw.rect(self.screen, BLACK, new_rect, 2)
                new_text = self.font.render('New Game', True, WHITE)
                self.screen.blit(new_text, (new_rect.centerx - new_text.get_width() // 2, new_rect.centery - 10))
            
            pygame.display.update()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game = BlackjackGame()
    game.run()
