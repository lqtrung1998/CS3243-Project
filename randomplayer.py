from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    #pp = pprint.PrettyPrinter(indent=2)
    #print("------------ROUND_STATE(RANDOM)--------")
    #pp.pprint(round_state)
    #print("------------HOLE_CARD----------")
    #pp.pprint(hole_card)
    #print("------------VALID_ACTIONS----------")
    #pp.pprint(valid_actions)
    #print("-------------------------------")
    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    #print("ACTION: ",action)
    #print("----",round_state)
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
    pass
    #print("Game_Start:\n\tAction:%s\n\tRoundState:%s\n\n"%(action,round_state))

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
    #print("Round_Result:\n\tWinner:%s\n\tHand_info:%s\n\tRoundState:%s\n\n"%(winners,hand_info,round_state))

def setup_ai():
  return RandomPlayer()
