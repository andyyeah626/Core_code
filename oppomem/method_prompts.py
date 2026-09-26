"""Game-agnostic prompt contracts for the OppoMem method stages.

Game rules and legal observations are supplied by the caller. These templates
cover the shared C1 and M1-M5 method constraints, not any game's role prompt.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .types import VisibleView


ACTION_ABSTRACT_RULE = """Describe a supported behavioral pattern, not SELF's
response plan. Use visible actions only as evidence. Do not include a literal
legal-action name, card identity, hint color or rank, issue-option ID, exact
deal, hidden information, or a prescription for SELF's next action. If the
evidence is insufficient, state uncertainty instead of guessing."""

JSON_CONTRACT = "Return exactly one valid JSON object; no markdown or extra prose."

MODELING_SYSTEM = ("Write one or two concise English sentences under 60 words "
                   "in an opponent_modeling field. " + ACTION_ABSTRACT_RULE + "\n" + JSON_CONTRACT)

SEGMENTATION_SYSTEM = ("Read completed player-visible events in order. Keep "
                       "participants distinct. Identify supported behavior, not advice. "
                       + ACTION_ABSTRACT_RULE + "\n" + JSON_CONTRACT)

REFLECTION_SYSTEM = ("Write reusable conditional response advice for SELF "
                     "from player-visible training evidence. The response may "
                     "name a legal action, but the memory key may not. Do not "
                     "invent future or hidden observations. Return a JSON object "
                     "with response_strategy under 40 words. " + JSON_CONTRACT)


def modeling_prompt(*, game_context: str, view: VisibleView, target_id: str,
                    previous_description: str | None) -> tuple[str, str]:
    payload = {
        "GAME_CONTEXT": game_context,
        "TARGET": target_id,
        "VISIBLE_OBSERVATION": view.observation,
        "COMPLETED_PUBLIC_HISTORY": list(view.public_history),
        "PREVIOUS_DESCRIPTION": previous_description,
    }
    return MODELING_SYSTEM, "Update the target's behavior description from current evidence.\nDATA:\n" + json.dumps(payload, ensure_ascii=False)


def segmentation_prompt(*, game_context: str, modeling_chain: Sequence[Mapping[str, Any]],
                        completed_visible_events: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    payload = {"GAME_CONTEXT": game_context,
               "ONLINE_OPPONENT_MODELING_CHAIN": list(modeling_chain),
               "COMPLETED_EVENTS": list(completed_visible_events)}
    user = ("Group supported descriptions by behavioral pattern, using coarse "
            "possibly overlapping windows with inclusive round_ranges. Omit "
            "unsupported regions. Return windows with exactly opponent_modeling "
            "and round_ranges; return an empty list when none are supported.\nDATA:\n"
            + json.dumps(payload, ensure_ascii=False))
    return SEGMENTATION_SYSTEM, user


def construction_prompt(*, game_context: str, canonical_modeling: str,
                        assigned_visible_records: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    payload = {"GAME_CONTEXT": game_context,
               "CLUSTERED_OPPONENT_MODELING": canonical_modeling,
               "CORRESPONDING_TRAJECTORY": list(assigned_visible_records)}
    user = ("Use observed actions and outcomes to write reusable conditional "
            "advice. Actual play takes precedence over the description. Return "
            "exactly a response_strategy field.\nDATA:\n"
            + json.dumps(payload, ensure_ascii=False))
    return REFLECTION_SYSTEM, user


def revision_prompt(*, game_context: str, canonical_modeling: str, prior_response: str,
                    configuration_summaries: Sequence[Mapping[str, Any]],
                    eligible_episode_evidence: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    payload = {"GAME_CONTEXT": game_context,
               "CANONICAL_OPPONENT_MODELING": canonical_modeling,
               "PRIOR_RESPONSE_STRATEGY": prior_response,
               "CONFIGURATION_SUMMARIES": list(configuration_summaries),
               "EPISODE_EVIDENCE": list(eligible_episode_evidence)}
    user = ("Compare enabled and disabled terminal returns within each opponent "
            "configuration. Each episode contributes one return regardless of "
            "retrieval count. Use supplied support and weights; do not invent "
            "missing returns. If evidence does not justify change, return the "
            "prior advice unchanged. Return exactly a response_strategy field.\nDATA:\n"
            + json.dumps(payload, ensure_ascii=False))
    return REFLECTION_SYSTEM, user
