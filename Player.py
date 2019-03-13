

# In[15]:


from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

from My_utils import hand_strength_with_belief, hand_strength, win_rate
import itertools
import numpy as np
from time import time,sleep
import pprint
# In[ ]:


class ReasonablePlayer():

    def __init__(self, hole_cards, round_state, belief = None):
        self.round_state = round_state
        self.hole_cards = hole_cards
        if(belief == None):
            self.belief = {'Cards':[],'Probability':[]}
            self.initialize_belief()
        else:
            self.belief = belief 

    def initialize_belief(self):
        used = [card.to_id() for card in self.hole_cards + self.round_state['community_card']]
        unused = [card_id for card_id in range(1, 53) if card_id not in used]

        Cards = [[Card.from_id(card1),Card.from_id(card2)]
                                for card1,card2 in itertools.combinations(unused,2)]
        # start= time()
        Cards_Strength = list(map(lambda x: hand_strength(x,self.round_state['community_card']),Cards))
        # print(time()-start)
        # print("+++++++++")
        self.belief['Cards'] = list(zip(Cards,Cards_Strength))
        self.belief['Probability'] = np.ones(len(Cards))/len(Cards)

        self.belief['Cards'].sort(key=lambda x: x[1])

    # Return probability of each possible action
    def actions(self):
        strength = np.sum(np.array([x[1] for x in self.belief['Cards']]) * self.belief['Probability'])
        if strength >=0.7:
            return {'fold': 0.1, 'call': 0.35, 'raise':0.45}
        elif strength >=0.5:
            return {'fold':0.2,'call': 0.6, 'raise':0.1}
        elif strength >= 0.3:
            return {'fold':0.4,'call': 0.5, 'raise':0.1}
        else:
            return {'fold':0.8,'call':0.19, 'raise':0.01}

    # Update belief from previous action
    def update_belief(self):
        street = self.round_state['street']
        if not self.round_state['action_histories'][street]:
            return
        previous_action = self.round_state['action_histories'][street][-1]
        strong_update = 0
        week_update = 0

        if previous_action == 'raise':
            count = 0
            for i in range(len(self.belief['Cards'])):
                if self.belief['Cards'][1] >0.65:
                    count = count+1
                else:
                    break;
            self.belief['Probability'][:count] += strong_update
            self.belief['Probability'][count:] -= week_update
            self.belief['Probability'] /= sum(self.belief['Probability'])

    def copy_player(self):
        return ReasonablePlayer(self.hole_cards,self.round_state,self.belief)


# In[9]:


class MyPlayer(BasePokerPlayer):
    def __init__(self):
        self.table = {}
        self.belief = {}
        # self.opponent_belief ={}
        self.uuid = None
        self.opponent = None
        self.emulator = Emulator()
        self.emulator.set_game_rule(2,1,10,0)
        self.current_cost = 10
        self.random_game_state = None 
        
    def declare_action(self, valid_actions, hole_card, round_state):
        self.round_state = round_state
        self.table = {}
        print(self.current_cost)
        self.my_uuid = round_state['seats'][round_state['next_player']]['uuid']
        self.my_cards = gen_cards(hole_card)
        self.community_card = gen_cards(round_state['community_card'])
        self.random_game_state = None

        if round_state['seats'][round_state['big_blind_pos']]['uuid'] == self.my_uuid:
            self.current_cost = 20

        if not self.belief:
            self.initialize_belief()

        posibile_opponent_cards = [x[0] for x in self.belief['Cards']]
        opponent_card_prob = self.belief['Probability']
        pp = pprint.PrettyPrinter(indent =2)
        start = time() 
        for opponent_cards, prob in zip(posibile_opponent_cards, opponent_card_prob):
            # start = time()            
            game_state = setup_game_state(round_state, self.my_cards, opponent_cards, self.my_uuid)
            self.opponent = ReasonablePlayer(opponent_cards, round_state,self.belief) ## Should not be self.belief, fix later
                # print(time()-start)
            root = Node(self.my_cards, opponent_cards,self.community_card, game_state, round_state, round_state['street'], prob, 
                        [], self, self.opponent, self.my_uuid, self.emulator, self.current_cost)
            # start= time()
            root.update_expected_table()
            # print(time()-start)
            # print("____")
            # pp.pprint(self.table)
            # sleep(5)
            if not self.random_game_state:
                self.random_game_state = game_state
                
        print(time()-start)
                
        strategy = max(self.table, key = self.table.get)
        action = strategy.split()[0]
        
        histories = self.emulator.apply_action(self.random_game_state, action)[1][0]['round_state']['action_histories']
        if histories.get(round_state['street']):
            action_result = histories[round_state['street']][-1]
            if action_result['action'] != 'FOLD':
                print(action_result)
                self.current_cost += action_result['paid']
        
            # print(action)
        # self.update_opp_belief(opponent_belief)
        return action
    
    # def update_opp_belief(self):

    def initialize_belief(self):        
        used = [card.to_id() for card in self.my_cards + self.community_card]
        unused = [card_id for card_id in range(1, 53) if card_id not in used]

        Cards =  [[Card.from_id(card1),Card.from_id(card2)] 
                                for card1,card2 in itertools.combinations(unused,2)]
        Cards_Strength = list(map(lambda x: hand_strength(x,self.community_card),Cards))

        self.belief['Cards'] = list(zip(Cards,Cards_Strength))
        self.belief['Probability'] = np.ones(len(Cards))/len(Cards)

        self.belief['Cards'].sort(key=lambda x: x[1])
        
    
      
    def update_belief(self):
        street = self.round_state['street']
        previous_action = self.game_update[-1]['action'] if self.game_state[-1]['player_uuid'] == self.my_uuid else self.game_update[-2]['action']
        strong_update = 0.2
        week_update = 0.1

        if previous_action == 'raise':
            count = 0
            for i in range(len(self.belief['Cards'])):
                if self.belief['Cards'][1] > 0.65:
                    count = count+1
                else:
                    break;
            self.belief['Probability'][:count] += strong_update
            self.belief['Probability'][count:] -= week_update
            self.belief['Probability'] /= sum(self.belief['Probability'])
    
    def update_table(self, my_strategy, gain):
        # print(my_strategy)
        key = ' '.join(my_strategy)
        if self.table.get(key):
            self.table[key] += gain
        else:
            self.table[key] = gain
            
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.action = action 

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
            

        
def setup_game_state(round_state, my_hole_cards, opponent_hole_cards, my_uuid):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        uuid = player_info['uuid']
        if uuid == my_uuid:
            game_state = attach_hole_card(game_state, uuid, my_hole_cards)
        else:
            game_state = attach_hole_card(game_state, uuid, opponent_hole_cards)
    return game_state


# In[13]:


class Node():
    def __init__(self,my_hole_cards, opponent_hole_cards, community_cards, game_state, round_state, street, prob_reach_this_node, 
                my_strategy, my_player, opponent_player, my_uuid, emulator, current_cost = 0):
        
        self.game_state = game_state
        self.round_state = round_state
        self.street = street
        self.emulator = emulator
        
        self.my_hole_cards = my_hole_cards 
        self.opponent_hole_cards = opponent_hole_cards
        self.community_cards = community_cards

        self.prob_reach_this_node = prob_reach_this_node
        self.my_strategy = my_strategy
        self.my_uuid = my_uuid
        

        self.my_player = my_player
        self.opponent_player = opponent_player
        
        self.current_cost = current_cost
        
    def is_termial(self):
        return (self.round_state['street'] != self.street) or self.emulator._is_last_round(self.game_state, self.emulator.game_rule)
    
    def is_my_turn(self):
        return self.round_state['seats'][self.round_state['next_player']]['uuid'] == self.my_uuid 
    
    def perform_action(self, action, prob, is_my_player):
        state, events = self.emulator.apply_action(self.game_state, action)
        new_current_cost = self.current_cost
        copy_opponent = self.opponent_player.copy_player()
        if is_my_player: 
            copy_opponent.update_belief()
            if action != 'fold':
                new_current_cost += events[0]['round_state']['action_histories'][self.street][-1]['paid']            
            
        return Node(self.my_hole_cards,
                    self.opponent_hole_cards,
                    self.community_cards,
                    state, events[0]['round_state'], 
                    self.street, prob * self.prob_reach_this_node,
                    self.my_strategy + [action],
                    self.my_player,
                    copy_opponent,
                    self.my_uuid,
                    self.emulator,
                    new_current_cost)
    
    def update_expected_table(self):
        if self.is_termial():
            # start = time()
            gain = self.expected_gain(win_rate)
            # print(time()-start)
            # print(self.my_strategy)
            self.my_player.update_table(self.my_strategy, gain)
        else:
            actions = [x['action'] for x in self.emulator.generate_possible_actions(self.game_state)]
            if self.is_my_turn():       
                for action in actions:
                    child_node = self.perform_action(action, 1, True)
                    child_node.update_expected_table()
            else:
                opponent_action_prob = self.opponent_player.actions()
                if len(actions) < 3:
                    opponent_action_prob['fold'] += opponent_action_prob['raise']
                for action in actions:
                    child_node = self.perform_action(action, opponent_action_prob[action], False)
                    child_node.update_expected_table()
                    
    def expected_gain(self, eval_fn):       
        pot = self.round_state['pot']['main']['amount']
        last_action = self.round_state['action_histories'][self.street][-1]
        if last_action['action'] == 'FOLD':
            if last_action['uuid'] == self.my_uuid:
                return -self.current_cost * self.prob_reach_this_node
            else:
                return (pot - self.current_cost) * self.prob_reach_this_node

        win_rate = eval_fn(self.my_hole_cards, self.community_cards, self.opponent_hole_cards)
        return  (pot - self.current_cost) * self.prob_reach_this_node * win_rate - self.current_cost * self.prob_reach_this_node * (1 - win_rate)
