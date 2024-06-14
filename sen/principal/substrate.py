# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Substrate builder."""

from typing import Any

from collections.abc import Collection, Mapping, Sequence

import chex
import dm_env
import reactivex
from meltingpot.configs import substrates as substrate_configs
from meltingpot.substrate import get_factory_from_config
from meltingpot.utils.substrates import builder
from meltingpot.utils.substrates.wrappers import (
    base,
    collective_reward_wrapper,
    discrete_action_wrapper,
    multiplayer_wrapper,
    observables,
    observables_wrapper,
)
from ml_collections import config_dict
from reactivex import subject

from . import Principal


@chex.dataclass(frozen=True)
class SubstrateObservables:
    """Observables for a substrate.

    Attributes:
      action: emits actions sent to the substrate from players.
      timestep: emits timesteps sent from the substrate to players.
      events: emits environment-specific events resulting from any interactions
        with the Substrate. Each individual event is emitted as a single element:
        (event_name, event_item).
      dmlab2d: Observables from the underlying dmlab2d environment.
    """

    action: reactivex.Observable[Sequence[int]]
    timestep: reactivex.Observable[dm_env.TimeStep]
    events: reactivex.Observable[tuple[str, Any]]
    dmlab2d: observables.Lab2dObservables


class PrincipalSubstrate(base.Lab2dWrapper):
    """Specific subclass of Wrapper with overridden spec types."""

    def __init__(self, env: observables.ObservableLab2d, principal: Principal) -> None:
        """See base class."""
        super().__init__(env)
        self._action_subject = subject.Subject()
        self._timestep_subject = subject.Subject()
        self._events_subject = subject.Subject()
        self._observables = SubstrateObservables(
            action=self._action_subject,
            events=self._events_subject,
            timestep=self._timestep_subject,
            dmlab2d=env.observables(),
        )
        self.principal = principal

    def reset(self) -> dm_env.TimeStep:
        """See base class."""
        timestep = super().reset()
        self._timestep_subject.on_next(timestep)
        for event in super().events():
            self._events_subject.on_next(event)
        return timestep

    def step(self, action: Sequence[int]) -> dm_env.TimeStep:
        """See base class."""
        self._action_subject.on_next(action)
        timestep = super().step(action)
        for player_idx in range(len(timestep.reward)):
            player_reward = timestep.reward[player_idx]
            self.principal.apple_counts[player_idx] += player_reward
            if player_reward > 0:
                if player_reward != 1:
                    raise Exception("Reward is not 1")
                # add tax from principal
                tax = self.principal.calculate_tax(self.principal.apple_counts[player_idx])
                player_reward -= tax
                self.principal.collect_tax(tax)

        self._timestep_subject.on_next(timestep)
        for event in super().events():
            self._events_subject.on_next(event)
        return timestep

    def reward_spec(self) -> Sequence[dm_env.specs.Array]:
        """See base class."""
        return self._env.reward_spec()

    def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
        """See base class."""
        return self._env.observation_spec()

    def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
        """See base class."""
        return self._env.action_spec()

    def close(self) -> None:
        """See base class."""
        super().close()
        self._action_subject.on_completed()
        self._timestep_subject.on_completed()
        self._events_subject.on_completed()

    def observables(self) -> SubstrateObservables:
        """Returns observables for the substrate."""
        return self._observables


def build_substrate(
    *,
    lab2d_settings: builder.Settings,
    individual_observations: Collection[str],
    global_observations: Collection[str],
    action_table: Sequence[Mapping[str, int]],
    principal: Principal,
) -> PrincipalSubstrate:
    """Builds a Melting Pot substrate.

    Args:
      lab2d_settings: the lab2d settings for building the lab2d environment.
      individual_observations: names of the player-specific observations to make
        available to each player.
      global_observations: names of the dmlab2d observations to make available to
        all players.
      action_table: the possible actions. action_table[i] defines the dmlab2d
        action that will be forwarded to the wrapped dmlab2d environment for the
        discrete Melting Pot action i.

    Returns:
      The constructed substrate.
    """
    env = builder.builder(lab2d_settings)
    env = observables_wrapper.ObservablesWrapper(env)
    env = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=individual_observations,
        global_observation_names=global_observations,
    )
    env = discrete_action_wrapper.Wrapper(env, action_table=action_table)
    # Add a wrapper that augments adds an observation of the collective
    # reward (sum of all players' rewards).
    env = collective_reward_wrapper.CollectiveRewardWrapper(env)
    return PrincipalSubstrate(env, principal)


from collections.abc import Collection, Mapping, Sequence, Set

from meltingpot.utils.substrates.substrate_factory import SubstrateFactory


class PrincipalSubstrateFactory(SubstrateFactory):
    def build_principal(self, roles: Sequence[str], principal: Principal) -> PrincipalSubstrate:
        """Builds the substrate.

        Args:
        roles: the role each player will take.

        Returns:
        The constructed substrate.
        """
        return build_substrate(
            lab2d_settings=self._lab2d_settings_builder(roles),
            individual_observations=self._individual_observations,
            global_observations=self._global_observations,
            action_table=self._action_table,
            principal=principal,
        )


def build_principal_from_config(
    config: config_dict.ConfigDict, *, roles: Sequence[str], principal: Principal
) -> PrincipalSubstrate:
    """Builds a substrate from the provided config.

    Args:
      config: config resulting from `get_config`.
      roles: sequence of strings defining each player's role. The length of
        this sequence determines the number of players.
      principal: the principal

    Returns:
      The training substrate.
    """
    return get_factory_from_config(config).build_principal(roles, principal)


def get_factory(name: str) -> PrincipalSubstrateFactory:
    """Returns the factory for the specified substrate."""
    config = substrate_configs.get_config(name)
    return get_factory_from_config(config)


def get_factory_from_config(config: config_dict.ConfigDict) -> PrincipalSubstrateFactory:
    """Returns a factory from the provided config."""

    def lab2d_settings_builder(roles):
        return config.lab2d_settings_builder(roles=roles, config=config)

    return PrincipalSubstrateFactory(
        lab2d_settings_builder=lab2d_settings_builder,
        individual_observations=config.individual_observation_names,
        global_observations=config.global_observation_names,
        action_table=config.action_set,
        timestep_spec=config.timestep_spec,
        action_spec=config.action_spec,
        valid_roles=config.valid_roles,
        default_player_roles=config.default_player_roles,
    )
