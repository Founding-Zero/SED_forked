from typing import Collection

import numpy as np
from dm_env import TimeStep


class Principal:
    def __init__(self, num_players, num_games) -> None:
        self.num_players = num_players
        self.num_games = num_games
        self.player_wealths = {
            f"game_{idx}": [0] * num_players for idx in range(num_games)
        }
        self.collected_tax = {f"game_{idx}": 0 for idx in range(num_games)}
        self.__tax_brackets = [(1, 10), (11, 20), (21, 10000)]
        self.tax_vals = {f"game_{idx}": [0, 0, 0] for idx in range(num_games)}

    def update_tax_vals(self, actions):
        for game_id in range(self.num_games):
            tax_choices = actions[game_id]
            for bracket_idx in range(len(tax_choices)):
                tax_val = tax_choices[bracket_idx].item()
                if tax_val != 11:
                    self.tax_vals[f"game_{game_id}"][bracket_idx] = tax_val / 10

    def set_objective(self, p_val):
        self.objective = self.swf
        self.p_val = p_val

    # tax calculated at each timestep
    def tax(self) -> float:
        games_taxes = {}
        total_collected = {}
        for game_id in self.player_wealths.keys():
            player_wealths = self.player_wealths[game_id]
            tax_vals = self.tax_vals[game_id]
            taxes = list(
                map(
                    lambda x: self.__tax_calculator(x, self.__tax_brackets, tax_vals),
                    player_wealths,
                )
            )
            total_collected[game_id] = sum(taxes)
            redistribution_amount = sum(taxes) / self.num_players
            games_taxes[game_id] = list(map(lambda x: x - redistribution_amount, taxes))
            # new_wealths = [wealth - tax + redistribution_amount for (wealth,tax) in zip(player_wealths,taxes)]
            # new_player_wealths[game_id] = new_wealths

        return games_taxes

    def __tax_calculator(self, wealth, brackets, tax_vals):
        tax = 0
        for i in range(len(brackets)):
            (lower, upper) = brackets[i]
            tax_val = tax_vals[i]
            if wealth >= lower:
                if wealth >= upper:
                    tax += (1 + upper - lower) * tax_val
                else:
                    tax += (1 + wealth - lower) * tax_val
        return tax

    def report_reward(self, reward) -> None:
        for i in range(len(reward)):
            game_id = i // self.num_players
            player_id = i % self.num_players
            self.player_wealths[f"game_{game_id}"][player_id] += reward[i]

    def swf(self, reward):
        result = []
        for i in range(0, len(reward), self.num_players):
            value = reward[i : i + self.num_players]
            welfare = (np.mean(value**self.p_val)) ** (1 / self.p_val)
            result.append(welfare)
        return np.array(result).flatten()
