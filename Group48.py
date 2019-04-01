from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator
# from My_utils import hand_strength_with_belief, hand_strength, win_rate
import itertools
import numpy as np
from time import time,sleep
import pprint
import copy
from CardProbability import CardDeck


# In[ ]:
PREFLOP_HAND_STRENGTH = np.load('preflop_hand_strength.npy').item()


emulator = Emulator()
emulator.set_game_rule(2,1,20,0)



# In[115]:


# heuristic_weights for[HC,LP,MP,HP,2P,3C,Straight,flush,FH,4C,SF]
# heuristic_weights =  np.array([1,1.5,2.5,4,10,20,40,80,160,320,640])
heuristic_weights =  np.array([0.1,1,2,3,7,14,28,56,112,224,448])
# heuristic_weights = heuristic_weights/np.sum(heuristic_weights)
# print(heuristic_weights)
#quantiles = [1,8,20]
QUANTILE = [0.4,0.6,0.75]

def heuristic1(hole, community_card, heuristic_weights=heuristic_weights):
    cards = hole + community_card
    score_vector = np.zeros(11)
    # if HandEvaluator._HandEvaluator__is_straightflash(cards): score_vector[10]=0
    if HandEvaluator._HandEvaluator__is_fourcard(cards): score_vector[9]=1
    if HandEvaluator._HandEvaluator__is_fullhouse(cards): score_vector[8]=1
    # if HandEvaluator._HandEvaluator__is_flash(cards): score_vector[7]=0
    if HandEvaluator._HandEvaluator__is_straight(cards): score_vector[6]=1
    if HandEvaluator._HandEvaluator__is_threecard(cards): score_vector[5]=1
    if HandEvaluator._HandEvaluator__is_twopair(cards): score_vector[4]=1
    if HandEvaluator._HandEvaluator__is_onepair(cards): 
        if cards[0].rank >=10:
            score_vector[3]=1
        elif cards[0].rank >=6:
            score_vector[2]=1
        else:
            score_vector[1]=1
            
    if cards and max(list(map(lambda x: x.rank,cards))) >=11:
        score_vector[0]=1
    return np.dot(heuristic_weights,score_vector)

heuristic = heuristic1 
# heuristic = hand_strength

def setup_game_state(round_state, my_hole_cards, my_uuid, opponent_hole_cards = []):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        uuid = player_info['uuid']
        if uuid == my_uuid:
            game_state = attach_hole_card(game_state, uuid, my_hole_cards)
        else:
            if not opponent_hole_cards:
                game_state = attach_hole_card_from_deck(game_state, uuid)
            else:
                game_state = attach_hole_card(game_state, uuid, opponent_hole_cards)
    return game_state

# In[116]:

class Advisor():
    def __init__(self, hole_card, my_uuid, weight):

        self.weight = weight
        self.round_state = None
        self.pay_off_table = {}
        
        self.cd = CardDeck()
        self.my_uuid = my_uuid
        self.my_card = gen_cards(hole_card)
        self.my_current_cost = 0
        self.my_belief = self.init_my_belief()
        self.my_strength = None
        
        self.opp_statistics = {'raise': 0, 'call': 0, 'fold': 0}
        
        
    def update_my_belief(self):
        self.cd.update_deck(round_state['community_card'])
        remove = []
        for i in range(len(self.my_belief['card'])):
            prob = self.cd.get_prob(self.my_belief['card'][i].ranks)
            if prob[0] == 0 or prob[1] == 0:
                remove.append(i)
            else:
                self.my_belief['probability'][i] = prob[0] * prob[1]
                # print (self.my_belief['card'][i].ranks[0], self.my_belief['card'][i].ranks[1], prob[0], prob[1])
        for item in ['card', 'probability', 'strength']:
            self.my_belief[item] = np.delete(self.my_belief[item], remove)


    def update_opp_hand_range(self):
        ## Update based on number of Fold
        unique_values = self.my_belief['strength']
        for i in range(len(unique_values)):
            indices = np.where(self.my_belief['strength'] ==  unique_values[i])
            if i >= len(unique_values)/2:
                self.my_belief['probability'][indices] *=2
                
        self.my_belief['probability'] = self.my_belief['probability']/sum(self.my_belief['probability'])
        # print(self.my_belief)
        
    def update_round_state(self,round_state):
        self.round_state = round_state
        ## initial cost
        if self.my_current_cost == 0:
            histories = self.round_state['action_histories']['preflop']
            for action in histories:
                if action['uuid'] == self.my_uuid:
                    if action['action'] in ['SMALLBLIND', 'BIGBLIND']:
                        self.my_current_cost += action['amount']

        elif round_state['seats'][round_state['big_blind_pos']]['uuid']==self.my_uuid and self.my_current_cost == 40:
            histories = self.round_state['action_histories']['preflop']
            for action in histories:
                if action['uuid'] == self.my_uuid:
                    if action['action'] in ['RAISE', 'CALL']:
                        self.my_current_cost += action['paid']

        elif round_state['seats'][round_state['big_blind_pos']]['uuid']!=self.my_uuid and self.my_current_cost == 20:
            histories = self.round_state['action_histories']['preflop']
            for action in histories:
                if action['uuid'] == self.my_uuid:
                    if action['action'] in ['RAISE', 'CALL']:
                        self.my_current_cost += action['paid']
            
        ## Update opp_statistics
        prev_street = {'flop': 'preflop', 'turn':'flop', 'river': 'turn'}
        histories = self.round_state['action_histories']
        if histories.get(self.round_state['street']):
            prev_opp_action = histories[self.round_state['street']][-1]['action'].lower()
            if self.opp_statistics.get(prev_opp_action) != None:
                self.opp_statistics[prev_opp_action] += 1
        elif prev_street.get(self.round_state['street']):
            if not histories.get(prev_street.get(self.round_state['street'])): 
                return 
            prev_opp_action = histories[prev_street.get(self.round_state['street'])][-1]['action'].lower()
            self.opp_statistics[prev_opp_action] += 1

    def init_my_belief(self):

        self.cd.update_deck([str(self.my_card[0]), str(self.my_card[1])])
        unsuit_rank = list(itertools.combinations_with_replacement(range(2,15),2))
        unsuit_card = np.array(list(map(lambda x: AbstractHoleCard([x[0],x[1]]),unsuit_rank)))
        probability = np.ones(91)
        for i in range(len(unsuit_card)):
            hand = unsuit_card[i]
            prob = self.cd.get_prob(hand.ranks)
            probability[i] = prob[0] * prob[1]

        strength = np.array(list(map(lambda x: heuristic1(x.to_hole_card(),community),unsuit_card)))
        my_strength = heuristic1(self.my_card,community)
        my_strength = my_strength/np.sum(strength)

        sort_indices = np.argsort(strength)
        strength = strength[sort_indices]/np.sum(strength)
        unique_values = np.unique(strength)

        for i in range(len(unique_values)):
            indices = np.where(strength == unique_values[i])
            strength[indices] = (i+1.0)/len(unique_values)
            if my_strength == unique_values[i]:
                self.my_strength = (i+1.0)/len(unique_values)     

        return {  'card':unsuit_card[sort_indices],
                  'strength': strength,
                  'probability':probability[sort_indices]
                }

    def get_bucket(self):
        bucketed_cards = {'card':[], 'strength':[], 'probability': []}
        
        unique_values = list(set(self.my_belief['strength']))
        
        for i in range(len(unique_values)):
            indices = np.where(self.my_belief['strength'] == unique_values[i])[0]           
            
            bucketed_cards['card'] += [self.my_belief['card'][indices][len(indices) // 2]]
            bucketed_cards['strength'] += [unique_values[i]]
            bucketed_cards['probability'] += [np.sum(self.my_belief['probability'][indices])]
            
        return bucketed_cards

    
#     def suggest_action(self):
#         # s =time()
#         buckets = self.get_bucket()
        
#         self.pay_off_table = {}

#         a_game_state = None

#         pp =pprint.PrettyPrinter(indent =2)
#         for i in range(len(buckets['strength'])):

#             # print(self.my_strength)
#             opp_card, strength, prob = buckets['card'][i] , buckets['strength'][i], buckets['probability'][i]
#             opp_card = opp_card.to_hole_card()
            
#             # print("Bucket",i,'Opponent_Strength',strength, [x.rank for x in opp_card])
            
#             game_state = setup_game_state(self.round_state, self.my_card, self.my_uuid, opp_card)
            
#             #print(HandEvaluator.gen_hand_rank_info(self.my_card, game_state['table'].get_community_card()))
# #             printGameState(game_state,self.my_uuid)
#             if not a_game_state:
#                 a_game_state = game_state
            
#             reasonable_opp = ReasonablePlayer(strength, self.weight ,self.opp_statistics, self.round_state['pot']['main']['amount']-self.my_current_cost)
#             root = Node(game_state, self.round_state, self.my_card, game_state['table'].get_community_card(), opp_card, self.round_state['street'], 
#                         prob, [], reasonable_opp, self, self.my_uuid, self.my_strength, strength, self.my_current_cost)
            
#             root.update_pay_off_table()
#             # print(self.pay_off_table)

#         strategies = table_to_strategies(self.pay_off_table)
#         # printCards(self.my_card)
#         # print(self.round_state['community_card'])
#         print(self.my_current_cost)
#         pp.pprint(strategies)
#         strategy = max(strategies, key = strategies.get)
#         action = strategy.split()[0]
        
#         histories = emulator.apply_action(a_game_state, action)[1][0]['round_state']['action_histories']
#         if histories.get(self.round_state['street']):
#             action_result = histories[self.round_state['street']][-1]
#             if action_result['action'] != 'FOLD':
#                 #print(action_result['uuid'], self.my_uuid)
#                 self.my_current_cost += action_result['paid']
#         # print("______",time()-s)
#         # print(time()-s)
#         return action

    def suggest_action(self):

        buckets = self.get_bucket()
        
        self.pay_off_table = {}

        a_game_state = None
        # print([(str(i),j) for (i,j) in zip(self.my_belief['card'],self.my_belief['probability'])])
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(buckets)
        # sleep(3)
        # print(self.my_current_cost)
        # sleep(3)
        for i in range(len(buckets['strength'])):
            # print(self.my_strength)
            opp_card, strength, prob = buckets['card'][i] , buckets['strength'][i], buckets['probability'][i]

            # print("PROBBBBB",prob, strength)
            available_suits = copy.deepcopy(SUITS)

            for card in self.my_card + gen_cards(self.round_state['community_card']):
                rank = card.rank
                suit =  card.suit
                available_suits[rank].remove(suit)
            # print(available_suits)
            opp_card = opp_card.to_hole_card(available_suits)
            
            game_state = setup_game_state(self.round_state, self.my_card, self.my_uuid, opp_card)
            # printGameState(game_state, self.my_uuid)
            if not a_game_state:
                a_game_state = game_state

            my_strength_with_opp = win_rate_with_opponent(self.my_card, game_state['table'].get_community_card(), opp_card)

            reasonable_opp = ReasonablePlayer(strength, self.weight ,self.opp_statistics, self.round_state['pot']['main']['amount']-self.my_current_cost)
            root = Node(game_state, self.round_state, self.my_card, game_state['table'].get_community_card(), opp_card, self.round_state['street'], 
                        prob, [], reasonable_opp, self, self.my_uuid, self.my_strength, my_strength_with_opp, strength, self.my_current_cost)
            
            root.update_pay_off_table()
            pp.pprint(self.pay_off_table)
            # sleep(1)
        
        strategies = table_to_strategies(self.pay_off_table)
        # print(self.my_current_cost)
        pp.pprint(strategies)
        # sleep(1)
        # sleep(3)
        strategy = max(strategies, key = strategies.get)
        action = strategy.split()[0]
        
        histories = emulator.apply_action(a_game_state, action)[1][0]['round_state']['action_histories']
        if histories.get(self.round_state['street']):
            action_result = histories[self.round_state['street']][-1]
            if action_result['action'] != 'FOLD':
                #print(action_result['uuid'], self.my_uuid)
                self.my_current_cost += action_result['paid']
        # print("____",time()-s)
        # print(time()-s)
        return action

# In[117]:
def printGameState(game_state,my_uuid):
    players = game_state['table'].seats.players
    community_cards = game_state['table']._community_card
    my_card = players[0].hole_card if players[0].uuid == my_uuid else players[1].hole_card
    opp_card = players[0].hole_card if players[0].uuid != my_uuid else players[1].hole_card

    print("Community Card: ",list(map(lambda x: str(x), community_cards)))
    print("My Card: ",list(map(lambda x: str(x), my_card)))
    print("Opp Card: ",list(map(lambda x: str(x), opp_card)))

class AbstractHoleCard():
    def __init__(self, ranks):
        self.ranks = ranks
        
    def from_hole_card(self, hole_card):
        card_ranks = []
        for card in hole_card:
            card_ranks.append(card.rank)
        return AbstractHoleCard(card_ranks)
    
    def to_hole_card(self, available_suits = None):
        if not available_suits:
            SUITS = np.array([2,4,8,16])
            cards = []
            for rank in self.ranks:
                cards.append(Card(np.random.choice(SUITS,1), rank))
            return cards     
        cards = [Card(x[0], x[1]) for x in assign_suit(self.ranks, available_suits)]
        # print('--')
        return cards     
    def __str__(self):
        return str(self.ranks)

# In[118]:


class ReasonablePlayer():
    def __init__(self, strength, weight, statistic, current_cost):
        self.weight = weight
        self.strength = strength
        self.statistic = statistic
        self.current_cost = current_cost

    # def actions(self):
    #     features = np.array([self.strength, self.current_cost, self.getAgressiveLevel()])
    #     weight = self.weight['opp_feature']
    #     bias = self.weight['opp_bias']

    #     values = np.matmul(features,weight) + bias
    #     probs = np.exp(values)/np.sum(np.exp(values))
    #     # print(probs)
    #     # print(np.sum(np.exp(values)))
    #     return {'raise':probs[0],
    #             'call':probs[1],
    #             'fold':probs[2]}

    
    
    def actions(self):
        
        if self.strength >= 0.65:
            # print("raise")
            return {'raise':1,'call':0,'fold':0}
        elif self.strength > 0.39:
            # print("call")
            return {'raise':0,'call':1,'fold':0}
        # elif self.strength > 0.3:
            # return {'raise':0,'call':1,'fold':0}
        # elif self.strength >0.3:
            # print("call")
            # return {'raise':0,'call':1,'fold':0}
        else:
            return {'raise':0,'call':0,'fold':1}
        # return {'raise':1,'call':0,'fold':0}
            
    def getAgressiveLevel(self):
        return 0

    def copy_player(self):
        return ReasonablePlayer(self.strength,self.weight,self.statistic, self.current_cost)


# In[ ]:


class MyPlayer(BasePokerPlayer):

    def __init__(self, weightPath = None, weight = None):
        self.name = "Group 48"
        self.my_uuid = ""
        self.advisor = None
        self.current_street = ""
        self.round_count = None
        
        if type(weight) != type(None):
            self.weight = weight
        else:
            if weightPath != None:
                self.weight = cPickle.load(open(weightPath,'rb'))
            else:
                self.weight = self.init_weight()

    def init_weight(self):
        weight = {}
        weight['opp_feature'] = np.random.normal(0,0.3,size =(3,3))
        weight['opp_bias'] = np.random.normal(0,0.3)
        return weight

    def declare_action(self, valid_actions, hole_card, round_state):
        # print(hole_card)
        # s = time()
        if round_state['street'] == 'preflop':
            
            return self.preflop_action(hole_card, round_state)
            
        self.advisor.update_round_state(round_state)
        #print(self.advisor.opp_statistics)
        action = self.advisor.suggest_action()
        # print("-___",time()-s)
        return action
    def preflop_action(self, hole_card, round_state):
        hole_card_ids = [card.rank for card in gen_cards(hole_card)]
        #my_strength = get_hand_strength(hole_card, [])
        hole = [x.rank for x in gen_cards(hole_card)]
        for i in range(len(PREFLOP_HAND_STRENGTH['card'])):
            if set(hole) == set(list(PREFLOP_HAND_STRENGTH['card'][i])):
                my_strength = PREFLOP_HAND_STRENGTH['strength'][i]
            
        action = 'fold'
        if my_strength >= 0.60:
            action = 'raise'
        elif my_strength >= 0.39:
            action = 'call'
        return action
        
    def receive_game_start_message(self, game_info):
        self.my_uuid = [seat['uuid'] for seat in game_info['seats'] if seat['name']==self.name][0]
        
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        new_advisor =  Advisor(hole_card, self.my_uuid, self.weight)
        if self.advisor != None:
            new_advisor.opp_statistics = self.advisor.opp_statistics
        self.advisor = new_advisor

    # def receive_street_start_message(self, street, round_state):
    #     self.current_street = street
    #     if street != 'preflop':
    #         self.advisor.my_belief = self.advisor.init_my_belief()

    def receive_street_start_message(self, street, round_state):
        self.advisor.update_round_state(round_state)
        self.current_street = street
        if street != 'preflop':
            self.advisor.my_belief = self.advisor.init_my_belief()
    def receive_game_update_message(self, action, round_state):
        
        if self.round_count >=10 & self.advisor.opp_statistics['fold'] >= float(self.round_count)/4:
            self.advisor.update_opp_hand_range()
        

    def receive_round_result_message(self, winners, hand_info, round_state):
        #print(hand_info)
        pass


# In[119]:


class Node():
    def __init__(self, game_state, round_state, my_hole_card, community_card, opp_hole_card,
                street, prob, my_path, reasonable_opp, advisor, my_uuid, my_strength,my_strength_with_opp, opp_strength,current_cost = 0):        
        self.game_state = game_state
        self.round_state = round_state
        self.street = street
        self.depth = 1
        self.is_terminal = False
        
        self.prob = prob
        # print("Initial",prob)
        self.my_path = my_path
        self.my_uuid = my_uuid
        self.advisor = advisor
        self.reasonable_opp = reasonable_opp
        self.current_cost = current_cost
        
        self.players = game_state['table'].seats.players
        self.my_hole_card = my_hole_card
        self.my_strength = my_strength
        self.my_strength_with_opp = my_strength_with_opp

        self.community_card = community_card
        
        self.opp_hole_card = opp_hole_card
        self.opp_strength = opp_strength
        
    def is_termial(self):
        return self.street != self.round_state['street'] or self.game_state['street'] == 5
    
    def is_my_turn(self):
        return self.round_state['seats'][self.round_state['next_player']]['uuid'] == self.my_uuid 

    def perform_action(self, action, prob, is_my_player):
        state, events = emulator.apply_action(self.game_state, action)
#         printGameState(state,self.my_uuid)
        # print(events)
        new_current_cost = self.current_cost
        copy_opponent = self.reasonable_opp.copy_player()

        if is_my_player: 
            if action != 'fold' and events[0]['round_state']['action_histories'][self.street][-1].get('paid') != None:
                new_current_cost += events[0]['round_state']['action_histories'][self.street][-1]['paid']
            # print(events[0]['round_state']['seats'][events[0]['round_state']['next_player']]['uuid'] == self.my_uuid)
            return Node(state, events[0]['round_state'], self.my_hole_card, self.community_card, self.opp_hole_card, self.street,
                    prob * self.prob,
                    self.my_path + [action],
                    copy_opponent,
                    self.advisor,
                    self.my_uuid,
                    self.my_strength,
                    self.my_strength_with_opp,
                    self.opp_strength,
                    new_current_cost)

        copy_opponent.current_cost += events[0]['round_state']['pot']['main']['amount'] - new_current_cost
        # print("_--___--",events[0]['round_state']['pot']['main']['amount'],self.current_cost)
        return Node(state, events[0]['round_state'], self.my_hole_card, self.community_card, self.opp_hole_card, self.street,
                    prob * self.prob,
                    self.my_path,
                    copy_opponent,
                    self.advisor,
                    self.my_uuid,
                    self.my_strength,
                    self.my_strength_with_opp,
                    self.opp_strength,
                    self.current_cost)
    
    def update_pay_off_table(self):
        if self.is_termial():
            # print("Terminal",self.game_state['street'])
            gain = self.expected_gain()
            key = ' '.join(self.my_path)
            if self.advisor.pay_off_table.get(key):
                self.advisor.pay_off_table[key] += gain
            else:
                self.advisor.pay_off_table[key] = gain
        else:
            # print(self.is_my_turn(),self.my_path,self.game_state['street'])
            actions = [x['action'] for x in emulator.generate_possible_actions(self.game_state)]
            if self.is_my_turn():       
                for action in actions:
                    print("My action",action)
                    child_node = self.perform_action(action, 1, True)
                    child_node.update_pay_off_table()
            else:
                opponent_action_prob = self.reasonable_opp.actions()

                if len(actions) < 3:
                    opponent_action_prob['call'] += opponent_action_prob['raise']
                for action in actions:                 
                    child_node = self.perform_action(action, opponent_action_prob[action], False)
                    child_node.update_pay_off_table()
                    
    def expected_gain(self):       
        pot = self.round_state['pot']['main']['amount']
        last_action = self.round_state['action_histories'][self.street][-1]
        if last_action['action'] == 'FOLD':
             if last_action['uuid'] == self.my_uuid:
                # print("my",self.my_path)
                # return 0*self.prob
                if self.prob >0: 
                    print(-self.current_cost * self.prob)
                    print("My Below in",self.prob)
                return -self.current_cost * self.prob
             else:
                # print("opp",self.my_path)
                # return 1*self.prob
                if self.prob >0:
                    print((pot - self.current_cost) * self.prob)
                    print("Opp Below in",self.prob)
                return (pot - self.current_cost) * self.prob
        
        # winrate_ = win_rate_with_opponent(self.my_hole_card,self.community_card,self.opp_hole_card)
        winrate  = self.my_strength_with_opp
        printGameState(self.game_state,self.my_uuid)
        if self.prob !=0:
            print("herre",self.prob)
            printCards(self.my_hole_card)
            printCards(self.community_card)
            printCards(self.opp_hole_card)
            # print("My current Cost",self.current_cost)
            print(self.street)
            print(self.my_path)
            print(winrate,self.my_strength,self.opp_strength)
            # print(pot)
            print("Out",(pot - self.current_cost) * self.prob * winrate - self.current_cost * self.prob * (1 - winrate))
#         print(win_rate(self.my_hole_card, self.community_card),win_rate(self.opp_hole_card,self.community_card))
        # print(winrate,self.current_cost,pot,(pot - self.current_cost) * self.prob * winrate - self.current_cost * self.prob * (1 - winrate))
        # return winrate
        print(self.street)
        # sleep(1)
        return  SCALING_FACTOR[self.street]*((pot - self.current_cost) * self.prob * winrate - self.current_cost * self.prob * (1 - winrate))

        
SCALING_FACTOR = {
    'flop': 2,
    'turn': 1.3,
    'river':1
}
def printCards(card):
    toPrint = list(map(lambda x: str(x), card))
    print(toPrint)
   

# In[120]:

def paths_to_actions(path_table):
    actions = {'raise': 0, 'call': 0, 'fold': 0}
    for key, val in path_table:
        action = key.split()[0]
        actions[action] += val

    return actions


def table_to_strategies(path_table):
    table = path_table.copy()
    strategies = path_table.copy()
    keys = list(table.keys())
    keys.sort(key = lambda x: len(x.split(" ")))
    # print(keys)
    # print("Keys",keys)
    for i in range(len(keys) - 1):
        key = keys[i]
        length = len(key)
        for j in range(i+1, len(keys)):
            key1 = keys[j]
            if key1[:length] == key:
                # print(key1,key)
                strategies[key1] += table[key]
    
    for i in range(len(keys) - 1):
        key = keys[i]
        length = len(key)
        for j in range(i+1, len(keys)):
            key1 = keys[j]
            if key1[:length] == key:
                if strategies.get(key) !=None:
                    del strategies[key]

    return strategies


SUITS = {   2  :  [2,4,8,16],
            3  :  [2,4,8,16],
            4  :  [2,4,8,16],
            5  :  [2,4,8,16],
            6  :  [2,4,8,16],
            7  :  [2,4,8,16],
            8  :  [2,4,8,16],
            9  :  [2,4,8,16],
            10 :  [2,4,8,16],
            11 :  [2,4,8,16],
            12 :  [2,4,8,16],
            13 :  [2,4,8,16],
            14 :  [2,4,8,16]}


def assign_suit(ranks, suits = None):    
    if not suits:
        suits = copy.deepcopy(SUITS)
    
    cards_with_suit = []
    for rank in ranks:
        suit = np.random.choice(suits[rank],1)[0]
        cards_with_suit.append([suit, rank])
        suits[rank].remove(suit)
    return cards_with_suit

def win_rate_with_opponent(my_card, community, opp_card):
    # s=time()
    my_card_ranks = [x.rank for x in my_card]
    community_ranks = [x.rank for x in community]
    opp_card_ranks = [x.rank for x in opp_card]
    combined = my_card_ranks + community_ranks + opp_card_ranks
    unique, counts = np.unique(combined, return_counts=True)
    
    max_count = 5 - len(community_ranks)
    if max_count > 0:
        elements = {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4}

        for i in range(len(unique)):
            elements[unique[i]] -= counts[i]

        L = list(itertools.combinations_with_replacement(elements, max_count))   
        for element in elements.keys():
            L = list(filter(lambda x: x.count(element) <= elements[element], L))
    else:
        L = [()]
    # print("First",time()-s)
    # s=time()
    ahead = 0
    behind = 0
    tied = 0
    
    for cards in L:
        # t=time()
        full_combined = [Card(x[0], x[1]) for x in assign_suit(my_card_ranks + community_ranks + list(cards) + opp_card_ranks)]
        community_cards_ = full_combined[2:-2]
        # t=time()
        opp_stength = HandEvaluator.eval_hand(full_combined[-2:], community_cards_)
        my_strength = HandEvaluator.eval_hand(full_combined[:2], community_cards_)
        # print((time()-t))
        if(my_strength > opp_stength):
            
            ahead = ahead + 1
        elif(my_strength < opp_stength):
            behind = behind + 1
        else:
            tied = tied + 1
        # print("One loop",time()-t)
    # print("second",time()-s)
    return float(ahead +tied/2)/(ahead +tied + behind)