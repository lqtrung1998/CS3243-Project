from pypokerengine.utils.card_utils import _fill_community_card, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator
from time import time
def hand_strength(cards, community_cards):
    nb_simulation = 10
    return estimate_hole_card_win_rate(nb_simulation, 2, cards, community_cards)

def win_rate(cards,community_cards,opp_cards):
    nb_simulation = 10

    ahead = 0
    behind = 0
    tied = 0

    for i in range(nb_simulation):
        community_cards_ = _fill_community_card(community_cards, used_card=cards+community_cards+opp_cards)
        opp_stength = HandEvaluator.eval_hand(opp_cards, community_cards_)
        my_strength = HandEvaluator.eval_hand(cards, community_cards_)
        if(my_strength > opp_stength):
            ahead = ahead + 1
        elif(my_strength < opp_stength):
            behind = behind + 1
        else:
            tied = tied + 1

    return (ahead +tied/2)/(ahead +tied + behind)

def hand_strength_with_belief(my_cards, community_cards, belief):
    
    overall_strength = 0
    nb_simulation = 10

    num_cards = len(belief['Cards'])

    for i in range(num_cards):
        opp_cards = belief['Cards'][i][0]
        prob = belief['Probability'][i]

        ahead = 0
        behind = 0
        tied = 0
        
        for i in range(nb_simulation):
            community_cards_ = _fill_community_card(community_cards, used_card=my_cards+community_cards+opp_cards)
            opp_stength = HandEvaluator.eval_hand(opp_cards, community_cards_)
            my_strength = HandEvaluator.eval_hand(my_cards, community_cards_)
            if(my_strength > opp_stength):
                ahead = ahead + 1
            elif(my_strength < opp_stength):
                behind = behind + 1
            else:
                tied = tied + 1

        overall_strength += prob * (ahead +tied/2)/(ahead +tied + behind)
    return overall_strength

