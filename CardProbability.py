##!/usr/bin/env python

class CardDeck():

    def __init__(self):
        self.deck = { 2: 4, 3: 4, 4: 4 , 5: 4 , 6: 4 , 7: 4 , 8: 4 , 9: 4 , 10: 4 , 11: 4 , 12: 4 , 13: 4 , 14: 4 }
        self.total_cards = 52

    def convert_card_to_num(self, card):
        dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        if (card[1] in dict.keys()):
            return dict[card[1]]
        return int(card[1:])

    def update_deck(self, known_cards):
        for card in known_cards:
            card = self.convert_card_to_num(card)
            if card in self.deck:
                self.deck[card] -= 1
                self.total_cards -= 1

    def get_prob(self, cards):
        prob_first = float(self.deck[cards[0]]) / self.total_cards
        val = 1 if (cards[1] == cards[0]) else 0
        prob_second = float(self.deck[cards[1]]-val) / (self.total_cards-1)
        return [prob_first, prob_second]

    def print_deck(self):
        print (self.deck)

if __name__ == "__main__":
    c = CardDeck()
    c.get_prob([2,3])
    c.update_deck(['H3'])
    c.get_prob([2,2])
