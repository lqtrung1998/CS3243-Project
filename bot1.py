from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint
from My_utils import hand_strength_with_belief, hand_strength, win_rate
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

class Bot1(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    strength = hand_strength(gen_cards(hole_card), gen_cards(round_state['community_card']))
    action = 'fold'
    if strength >= 0.65:
        action = 'raise'
    elif strength > 0.39:
        action = 'call'

    return action  # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass
    #print("----------\nGameStart:",game_info)

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass
    #print("RoundStart\n\tRound_count:%s\n\tHole_Cart:%s\n\tSeats:%s\n\n"%(round_count,hole_card,seats))

  def receive_street_start_message(self, street, round_state):
    pass
    #print("Street Start:\n\tStreet:%s\n\tRoundState:%s\n\n"%(street, round_state))

  def receive_game_update_message(self, action, round_state):
    #print(round_state)
    pass
    #print("Game_Start:\n\tAction:%s\n\tRoundState:%s\n\n"%(action,round_state))

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
    #print("Round_Result:\n\tWinner:%s\n\tHand_info:%s\n\tRoundState:%s\n\n"%(winners,hand_info,round_state))

def setup_ai():
  return RandomPlayer()
