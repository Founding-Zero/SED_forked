from abc import ABC, abstractmethod

import numpy as np


class VotingMechanism(ABC):
    @abstractmethod
    def vote_on_p(self, player_values: np.ndarray):
        pass


class SimpleMean(VotingMechanism):
    def vote_on_p(self, player_values: np.ndarray):
        return np.average(player_values)


class SimpleMedian(VotingMechanism):
    def vote_on_p(self, player_values: np.ndarray):
        return np.median(player_values)


class VotingFactory:
    @staticmethod
    def get_voting_mechanism(voting_type: str) -> VotingMechanism:
        if voting_type == "simple_mean":
            return SimpleMean()
        elif voting_type == "simple_median":
            return SimpleMedian()
