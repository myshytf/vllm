# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class AcceptanceLengthUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float


class AcceptanceLengthController:
    """Adjust speculative depth from the observed accepted draft length."""

    def __init__(
        self,
        max_num_spec_tokens: int,
        observation_window: int,
        choices: list[int] | None = None,
    ) -> None:
        if max_num_spec_tokens <= 0:
            raise ValueError("max_num_spec_tokens must be greater than zero.")
        if observation_window <= 0:
            raise ValueError("observation_window must be greater than zero.")
        if choices is not None:
            choices = sorted(set(int(c) for c in choices))
            if not choices or choices[0] < 1 or choices[-1] > max_num_spec_tokens:
                raise ValueError(
                    "choices must be a non-empty subset of 1..max_num_spec_tokens."
                )

        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        # Depths the controller may select; None = every depth in 1..max.
        self.choices = choices
        self.num_spec_tokens = choices[-1] if choices else max_num_spec_tokens

        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> AcceptanceLengthUpdate | None:
        """Observe one scheduler step and occasionally update the depth."""
        if num_drafts < 0 or num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("Speculative decoding counts must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError("num_accepted_tokens must not exceed num_draft_tokens.")
        if num_drafts == 0:
            if num_draft_tokens or num_accepted_tokens:
                raise ValueError("Token counts require at least one draft.")
            return None

        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if self._num_observation_steps < self.observation_window:
            return None

        mean_num_accepted_tokens = self._num_accepted_tokens / self._num_drafts
        mean_num_draft_tokens = self._num_draft_tokens / self._num_drafts
        target_num_spec_tokens = self._snap(
            min(
                self.max_num_spec_tokens,
                max(1, floor(mean_num_accepted_tokens + 1.5)),
            )
        )

        previous_num_spec_tokens = self.num_spec_tokens
        if target_num_spec_tokens < self.num_spec_tokens:
            self.num_spec_tokens = target_num_spec_tokens
        elif target_num_spec_tokens > self.num_spec_tokens:
            self.num_spec_tokens = self._next_choice(self.num_spec_tokens)

        self._reset_window()
        return AcceptanceLengthUpdate(
            previous_num_spec_tokens=previous_num_spec_tokens,
            num_spec_tokens=self.num_spec_tokens,
            mean_num_accepted_tokens=mean_num_accepted_tokens,
            mean_num_draft_tokens=mean_num_draft_tokens,
        )

    def _snap(self, target: int) -> int:
        """Largest allowed depth not above ``target`` (the smallest choice below it)."""
        if self.choices is None:
            return target
        allowed = [c for c in self.choices if c <= target]
        return allowed[-1] if allowed else self.choices[0]

    def _next_choice(self, current: int) -> int:
        """One allowed step up from ``current`` (raising is gradual)."""
        if self.choices is None:
            return current + 1
        above = [c for c in self.choices if c > current]
        return above[0] if above else current

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
