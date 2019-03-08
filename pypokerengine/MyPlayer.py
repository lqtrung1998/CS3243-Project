from pypokerengine.players
from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck

def setup_game_state(round_state, my_hole_card):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        if uuid == self.uuid:
            # Hole card of my player should be fixed. Because we know it.
            game_state = attach_hole_card(game_state, uuid, my_hole_card)
        else:
            # We don't know hole card of opponents. So attach them at random from deck.
            game_state = attach_hole_card_from_deck(game_state, uuid)
            
class Game:
    def __init__(self, round_state, hole_card, player):
        self.player = player
        self.initial_state = setup_game_state(round_state, hole_card)
        self.hole_card = hole_card
        self.emulator = Emulator()
        self.emulator.set_game_rule(nb_player=2, max_round=10, sb_amount=10, ante_amount=0)

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        return generate_possible_actions(self.emulator, state)

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        return self.emulator.apply_action(state, move)

    def utility(self, state, player):
        """Return the value of this final state to player."""
        score = estimate_hole_card_win_rate(100, 2, self.hole_card, state['community_cards'])

        # 1 is MAX player
        if self.player == player:
            return score
        else:
            return -score
    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return self.emulator._is_last_round(state,self.emulator.game_rule)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.next_player

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial_state
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial_state))

def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(game.actions(state),
                  key=lambda a: min_value(game.result(state, a)))