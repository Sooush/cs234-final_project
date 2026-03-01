"""
SRL reward computation: sequence similarity between model action step and expert step.
Baseline: difflib.SequenceMatcher ratio. Invalid outputs get reward=-1.
Stub interfaces for embedding cosine and LLM-as-judge.
"""

from __future__ import annotations

import os
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

    # Must contain think tags
    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return ("", "")

    # Extract think content
    think_start = text.find(THINK_OPEN)
    think_end = text.find(THINK_CLOSE, think_start)
    if think_end == -1:
        return ("", "")

    think_content = text[think_start + len(THINK_OPEN) : think_end].strip()

    # Action step: everything after </think>, trimmed
    after_think = text[think_end + len(THINK_CLOSE) :].strip()

    # Optional [ACTION STEP TEXT] marker - we take the line or block after </think>
    action_step = after_think
    # If multiple lines, take first non-empty as single step; if looks like multiple numbered steps, invalid
    lines = [l.strip() for l in action_step.split("\n") if l.strip()]

    # Invalid: multiple steps (e.g. "1. ... \n 2. ...")
    if len(lines) > 1:
        # Check if second line looks like a new step (starts with digit.)
        if lines[1] and re.match(r"^\d+\.\s", lines[1]):
            return ("", "")

    # Single step: use first line or whole block if single block
    if lines:
        # Heuristic: if first line is short and rest is continuation, use full block
        if len(lines) == 1:
            action_step = lines[0]
        else:
            # Could be one step with multiple lines (e.g. equation block)
            action_step = "\n".join(lines)

    # Normalize whitespace
    action_step = " ".join(action_step.split()) if action_step else ""

    return (think_content, action_step)


def get_parse_failure_reason(text: str) -> str:
    """
    Return a short reason why parse_srl_output would give empty action_step.
    Use for debugging when reward is -1.
    """
    if not text or not isinstance(text, str):
        return "empty_or_invalid_input"
    text = text.strip()
    if not text:
        return "empty_after_strip"
    if THINK_OPEN not in text:
        return "missing_think_open_tag"
    if THINK_CLOSE not in text:
        return "missing_think_close_tag"
    think_start = text.find(THINK_OPEN)
    think_end = text.find(THINK_CLOSE, think_start)
    if think_end == -1:
        return "missing_think_close_tag"
    after_think = text[think_end + len(THINK_CLOSE) :].strip()
    if not after_think:
        return "nothing_after_think_close"
    lines = [l.strip() for l in after_think.split("\n") if l.strip()]
    if not lines:
        return "only_whitespace_after_think_close"
    if len(lines) > 1 and lines[1] and re.match(r"^\d+\.\s", lines[1]):
        return "multiple_numbered_steps"
    return "unknown"


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
    """Baseline: difflib.SequenceMatcher ratio"""

    def __call__(self, model_step: str, expert_step: str) -> float:
        if not expert_step:
            return 0.0  # malformed instance — no expert step to compare against
        if not model_step:
            return 0.0
        matcher = SequenceMatcher(None, expert_step, model_step)
        return matcher.ratio()

# Default reward
DEFAULT_REWARD_FN = SequenceMatcherReward()

INVALID_REWARD = -1.0

# Debug: log only first N parse failures per process so we see reason + snippet without spam
_PARSE_FAILURE_LOG_COUNT = 0
_PARSE_FAILURE_LOG_MAX = 3


def compute_srl_reward(
    model_output: str,
    expert_step: str,
    reward_fn: Optional[RewardFn] = None,
) -> float:
    """
    SRL-specific: Parse model output, extract action step, compute reward.
    Invalid format -> -1.
    """
    if reward_fn is None:
        reward_fn = DEFAULT_REWARD_FN

    if not expert_step:
        return INVALID_REWARD

    think, action_step = parse_srl_output(model_output)
    if not action_step:
        global _PARSE_FAILURE_LOG_COUNT
        if _PARSE_FAILURE_LOG_COUNT < _PARSE_FAILURE_LOG_MAX:
            print(f"[SRL reward] No action step: model_output={model_output}")
            reason = get_parse_failure_reason(model_output)
            snippet = (model_output.strip()[:350] + "...") if len(model_output) > 350 else model_output.strip()
            print(f"[SRL reward] No action step: reason={reason} | snippet={repr(snippet)}")
            _PARSE_FAILURE_LOG_COUNT += 1
        return INVALID_REWARD

    return reward_fn(action_step, expert_step)
