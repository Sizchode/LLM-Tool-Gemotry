from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from .schema import Decision, Tool, Trace


def generate(seed: int, n_tools: int, n_decisions: int, n_traces: int) -> tuple[list[Tool], list[Decision], list[Trace], np.ndarray]:
    rng = random.Random(seed); gen = np.random.default_rng(seed)
    n_groups = max(3, n_tools // 5)
    centres = gen.normal(size=(n_groups, 8))
    tools = []
    latent = []
    for index in range(n_tools):
        group = index % n_groups
        name = f"api_{group}_tool_{index:02d}"
        # Keep schema deliberately non-diagnostic: Paper 1's mock is a sanity
        # check that the predictor comparison can recover an internal signal,
        # not a synthetic win caused by leaking the latent group into schema.
        tools.append(Tool(f"mock.{index:02d}", name, f"Perform group {group} operation {index}.", {"type": "object", "properties": {"query": {"type": "string"}}}, "mock"))
        latent.append(centres[group] + gen.normal(scale=.20, size=8))
    latent = np.asarray(latent)
    decisions = []
    for index in range(n_decisions):
        gold = rng.randrange(n_tools); candidates = {gold}
        while len(candidates) < min(6, n_tools):
            # Same latent group acts as the naturally confusable hard negative.
            pool = [j for j in range(n_tools) if j % n_groups == gold % n_groups] if rng.random() < .7 else list(range(n_tools))
            candidates.add(rng.choice(pool))
        candidates = sorted(candidates)
        closest = sorted((j for j in candidates if j != gold), key=lambda j: np.linalg.norm(latent[j] - latent[gold]))
        chosen = gold if rng.random() < .73 else (closest[0] if closest else gold)
        decisions.append(Decision(f"d{index:05d}", f"Need group {gold % n_groups} operation for request {index}", [tools[j].tool_id for j in candidates], tools[gold].tool_id, tools[chosen].tool_id, "mock"))
    traces = []
    for index in range(n_traces):
        group = rng.randrange(n_groups); chain = [j for j in range(n_tools) if j % n_groups == group]
        rng.shuffle(chain)
        traces.append(Trace(f"t{index:04d}", [tools[j].tool_id for j in chain[:rng.randint(2, min(4, len(chain)))]], "mock"))
    return tools, decisions, traces, latent
