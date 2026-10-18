# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Critic-round limits shared by the visualizer, skill CLI, and evolution view."""

from __future__ import annotations

import math
from typing import Any, Mapping

_IMAGE_SUFFIX = "_base64_jpg"
DEFAULT_MAX_CRITIC_ROUNDS = 3


def coerce_max_critic_rounds(value: Any) -> int | None:
    """Return a non-negative round count, or None when the value is not set.

    Integers and integer strings such as ``'4'`` are valid. ``None``, ``''``,
    and non-numeric values such as ``'five'`` are treated as unset so callers
    can fall through. Negatives clamp to 0. Non-integral numbers are unset.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return max(0, int(text, 10))
        except ValueError:
            return None
    return None


def resolve_max_critic_rounds(
    sample_value: Any,
    configured_value: Any = None,
    default: int = DEFAULT_MAX_CRITIC_ROUNDS,
) -> int:
    """Use the sample count, then ExpConfig, then ``default`` (3)."""
    for candidate in (sample_value, configured_value):
        resolved = coerce_max_critic_rounds(candidate)
        if resolved is not None:
            return resolved
    fallback = coerce_max_critic_rounds(default)
    return DEFAULT_MAX_CRITIC_ROUNDS if fallback is None else fallback


def critic_round_indexes(record: Mapping[str, Any], prefix: str) -> list[int]:
    """Sorted critic indexes from present keys and a configured max, if valid.

    ``prefix`` is ``target_diagram`` or ``target_plot``. Description keys and
    ``*_base64_jpg`` keys both count as available rounds. A valid
    ``max_critic_rounds`` also contributes ``range(n)``.
    """
    indexes: set[int] = set()
    desc_prefix = f"{prefix}_critic_desc"
    for key in record:
        if not isinstance(key, str) or not key.startswith(desc_prefix):
            continue
        rest = key[len(desc_prefix):]
        if rest.endswith(_IMAGE_SUFFIX):
            rest = rest[: -len(_IMAGE_SUFFIX)]
        if rest.isdigit():
            indexes.add(int(rest))
    configured = coerce_max_critic_rounds(record.get("max_critic_rounds"))
    if configured:
        indexes.update(range(configured))
    return sorted(indexes)
