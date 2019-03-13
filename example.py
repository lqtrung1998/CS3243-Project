from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from player import MyPlayer
#TODO:config the config as our wish
config = setup_config(max_round=11, initial_stack=10000, small_blind_amount=10)



config.register_player(name="f1", algorithm=RaisedPlayer())
config.register_player(name="MyPlayer", algorithm=MyPlayer())


game_result = start_poker(config, verbose=1)
