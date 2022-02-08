import collections
import numpy as np
import random
import abc
import setproctitle
import zmq


def pfsp(win_rates, weighting="linear"):
    weightings = {
        "variance": lambda x: x * (1-x),
        "linear": lambda x: 1-x,
        "linear_capped": lambda x: np.minimum(0.5, 1-x),
        "squared": lambda x: (1-x)**2
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    
    return probs / norm


class PayOff:
    def __init__(self):
        self._players = []
        self._wins = collections.defaultdict(lambda : 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
    

    def _win_rate(self, _own, _enemy):
        if self._games[_own, _enemy] == 0:
            return 0.5
        
        return (self._wins[_own, _enemy] + 0.5 * self._draws[_own, _enemy]) / self._games[_own, _enemy]

    def __getitem__(self, match):
        home, away = match

        if not isinstance(home, list):
            home = [home]
        
        if isinstance(away, list):
            away = [away]
        
        win_rates = np.array([[self.win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshap(-1)

        return win_rates


    def update(self, own, enemy, result):
        self._games[own, enemy] += 1
        self._games[enemy, own] += 1

        if result == "win":
            self._wins[own, enemy] += 1
            self._losses[enemy, own] += 1
        
        elif result == "draw":
            self._draws[own, enemy] += 1
            self._draws[enemy, own] += 1
        
        else:
            self._wins[enemy, own] += 1
            self._losses[own, enemy] += 1
    
    def add_player(self, player):
        self._players.append(player)
    
    @property
    def players(self):
        return self._players


# class League(object):
#     def __init__(self, main_agents=1, main_exploiters=1, league_exploiters=1):
        
#         self._learning_agents = []
#         self.payoff = PayOff()

#         for _ in range(main_agents):
#             main_agent = MainAgent()
#             self._learning_agents.append(main_agent)

#         for _ in range(main_exploiters):
#             main_exploiter = MainExploiter()
#             self._learning_agents.append(main_exploiter)
        
#         for _ in range(league_exploiters):
#             league_exploiter = LeagueExploiter()
#             self._learning_agents.append(league_exploiter)

#         for player in self._learning_agents:
#             self.payoff.add_player(player)
        
#     def update(self, own, enemy, result):
#         return self.payoff.update(own, enemy, result)

#     def add_player(self,player):
#         self.payoff.add_player(player)