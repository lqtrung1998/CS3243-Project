from pypokerengine.players
class State:
    def __init__(self, m_cards, o_cards, m_pot, o_pot, m_stack, o_stack,
                 is_end, action_history, is_small_blind, community_cards, to_move):
        self.m_cards = m_cards
        self.o_cards = o_cards
        self.m_pot = m_pot 
        self.o_pot = o_pot
        self.m_stack =  m_stack
        self.o_stack =  o_stack

        self.to_move = to_move
        self.is_end = is_end

        self.action_history = action_history
        self.is_small_blind = is_small_blind
        self.community_cards = community_cards

class Game:

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        """Zero sum game at the end"""

        community_cards = self.community_cards

        pmax_score = HandEvaluator.eval_hand(state.pmax.hole_card,community_cards)
        pmin_score = HandEvaluator.eval_hand(state.pmin.hole_card,community_cards)

        if(pmax_score > pmin_score):
            return  state.pmax_pot
        else if (pmax_score < pmin_score):
            return -state.pmax_pot
        return 0


    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class StochasticGame(Game):
    """A stochastic game includes uncertain events which influence
    the moves of players at each state. To create a stochastic game, subclass
    this class and implement chances and outcome along with the other
    unimplemented game class methods."""

    def chances(self, state):
        """Return a list of all possible uncertain events at a state."""
        raise NotImplementedError

    def outcome(self, state, chance):
        """Return the state which is the outcome of a chance trial."""
        raise NotImplementedError

    def probability(self, chance):
        """Return the probability of occurence of a chance."""
        raise NotImplementedError

    def play_game(self, *players):
        """Play an n-person, move-alternating stochastic game."""
        state = self.initial
        while True:
            for player in players:
                chance = random.choice(self.chances(state))
                state = self.outcome(state, chance)
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

def alphabeta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state, depth: depth > d or
                    game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action