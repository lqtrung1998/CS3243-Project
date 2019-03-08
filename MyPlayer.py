from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
import random as rand
import pprint
import numpy as np 

class MyPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):

        for player in round_state['seats']:
            if player['name'] == "MyPlayer":
                my_uuid = player['uuid']
        cards = list(map(lambda x: Card.from_str(x), hole_card))

        player = round_state['next_player']
        game_state = setup_game_state(round_state, cards, my_uuid)

        game = Game(game_state, cards, player)
        action = minimax_decision(game_state, game)
        return action  
  
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    #print("RoundStart\n\tRound_count:%s\n\tHole_Cart:%s\n\tSeats:%s\n\n"%(round_count,hole_card,seats))

    def receive_street_start_message(self, street, round_state):
        pass
    #print("Street Start:\n\tStreet:%s\n\tRoundState:%s\n\n"%(street, round_state))

    def receive_game_update_message(self, action, round_state):
        pass
    #print("Game_Start:\n\tAction:%s\n\tRoundState:%s\n\n"%(action,round_state))

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    #print("Round_Result:\n\tWinner:%s\n\tHand_info:%s\n\tRoundState:%s\n\n"%(winners,hand_info,round_state))

def setup_ai():
    return MyPlayer()

def setup_game_state(round_state, my_hole_card, my_uuid):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        uuid = player_info['uuid']
        if uuid == my_uuid:
            # Hole card of my player should be fixed. Because we know it.
            game_state = attach_hole_card(game_state, uuid, my_hole_card)
        else:
            # We don't know hole card of opponents. So attach them at random from deck.
            game_state = attach_hole_card_from_deck(game_state, uuid)
    return game_state
            
class Game:
    def __init__(self, game_state, hole_card, player):
        self.player = player
        self.initial_state = game_state
        self.hole_card = hole_card
        self.emulator = Emulator()
        self.emulator.set_game_rule(2, 10, 10, 0)

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        return list(map(lambda x:x['action'],self.emulator.generate_possible_actions(state)))

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        return self.emulator.apply_action(state, move)[0]

    def utility(self, state, player):
        """Return the value of this final state to player."""
        score = estimate_hole_card_win_rate(100, 2, self.hole_card, state['table']._community_card)

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
        return state['next_player']

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
    infinity = 10000000000000000000000000

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
    return np.argmax(list(map(lambda a: min_value(game.result(state, a)),
                    game.actions(state))))
                  

