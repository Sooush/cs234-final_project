"""
SRL reward computation: sequence similarity between model action step and expert step.
Based on paper 2510.25992 (Supervised Reinforcement Learning).
Baseline: difflib.SequenceMatcher ratio. Invalid outputs get reward=-1.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Optional


# --- SRL-specific: Parse model output into (think, action_step) ---

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def parse_srl_output(text: str) -> tuple[str, str]:
    """
    Parse model output into (think_content, action_step).
    Expected format: <think>...</think> [ACTION STEP TEXT]
    Returns ("", "") if parsing fails (invalid format).
    """
    if not text or not isinstance(text, str):
        return ("", "")

    text = text.strip()

    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return ("", "")

    think_start = text.find(THINK_OPEN)
    think_end = text.find(THINK_CLOSE, think_start)
    if think_end == -1:
        return ("", "")

    think_content = text[think_start + len(THINK_OPEN) : think_end].strip()
    after_think = text[think_end + len(THINK_CLOSE) :].strip()
    action_step = after_think
    lines = [l.strip() for l in action_step.split("\n") if l.strip()]

    if len(lines) > 1:
        if lines[1] and re.match(r"^\d+\.\s", lines[1]):
            return ("", "")

    if lines:
        if len(lines) == 1:
            action_step = lines[0]
        else:
            action_step = "\n".join(lines)

    action_step = " ".join(action_step.split()) if action_step else ""
    return (think_content, action_step)


def is_valid_srl_output(text: str) -> bool:
    """Check if output has valid format (think + single action step)."""
    think, action = parse_srl_output(text)
    return bool(think or action) and len(action) > 0


# --- Reward: baseline SequenceMatcher ---

class RewardFn(ABC):
    """Abstract reward function interface for pluggable variants."""

    @abstractmethod
    def __call__(self, model_step: str, expert_step: str) -> float:
        """Compute reward in [0, 1] for model_step vs expert_step."""
        pass


class SequenceMatcherReward(RewardFn):
    """Baseline: difflib.SequenceMatcher ratio (paper-style step similarity)."""

    def __call__(self, model_step: str, expert_step: str) -> float:
        if not expert_step:
            return 0.0
        if not model_step:
            return 0.0
        matcher = SequenceMatcher(None, expert_step, model_step)
        return matcher.ratio()


DEFAULT_REWARD_FN = SequenceMatcherReward()
INVALID_REWARD = -1.0


def compute_srl_reward(
    model_output: str,
    expert_step: str,
    reward_fn: Optional[RewardFn] = None,
) -> float:
    """
    SRL step-wise reward: parse model output, extract action step, compare to expert step.
    Invalid format -> INVALID_REWARD (-1.0). Valid -> reward in [0, 1].
    """
    if reward_fn is None:
        reward_fn = DEFAULT_REWARD_FN
    if not expert_step:
        return INVALID_REWARD
    think, action_step = parse_srl_output(model_output)
    if not action_step:
        return INVALID_REWARD
    return reward_fn(action_step, expert_step)
