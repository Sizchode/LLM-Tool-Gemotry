# Normalized data contract

Each adapter emits `tools.jsonl`, `decisions.jsonl`, and optional
`traces.jsonl`. IDs must be stable across reruns.

```json
{"tool_id":"weather.lookup","name":"weather_lookup","description":"...","schema":{"type":"object","properties":{}},"source":"bfcl_v4"}
{"decision_id":"d1","query":"...","candidate_tool_ids":["weather.lookup","weather.forecast"],"gold_tool_id":"weather.lookup","chosen_tool_id":"weather.forecast","source":"bfcl_v4","gold_position":0,"chosen_position":1,"menu_order_seed":null,"menu_variant_id":"original","gold_call_count":1}
{"trace_id":"t1","tool_ids":["calendar.find","calendar.create"],"source":"dataset"}
```

`candidate_tool_ids` is the complete ordered menu and therefore the statistical
risk set. `gold_position` and `chosen_position` must agree with it.
Paper 1 scores behavior only in the benchmark's original menu order.
`chosen_tool_id`/`chosen_position` may be null before measurement. The exact
reverse order is a representation-stability context and is recorded in
`context_index.jsonl`; it is not added as a second behavioral observation.

`gold_call_count` prevents multi-call examples from being silently projected
onto their first call. Paper 1's single-choice model excludes rows where this
value is not one and reports the resulting decision count.

Substitutability is not part of this base contract. A valid substitutability
estimand requires paired counterfactual records identifying the same query,
the replaced tool, its replacement, and whether task-level functionality was
preserved. Ordinary candidate co-exposure is not substitutability.

## Dataset adapters and lossless sidecars

- BFCL v4 `live_multiple`: one decision per official query, using the full
  ordered `function` menu and its matching official `possible_answer`. Exact
  name/description/schema variants are distinct tools. `gold_calls.jsonl`
  retains argument-level labels.
- Seal-Tools: one decision per row. Multi-call rows retain their true
  `gold_call_count`; `gold_calls.jsonl` preserves every call.
- ToolHop: one decision per benchmark-provided sub-task, not an invented
  first-call projection. The row's provided tools form the risk set;
  `traces.jsonl` and `trajectories.jsonl` retain all 995 ordered chains and
  intermediate answers. `executables.jsonl` preserves the 3,912 official
  Python sources as untrusted text; importing never executes them.

Every network importer writes `source_manifest.json` with its official source
and pinned revision. Sidecars stay outside `load_normalized`, so Paper 1 cannot
accidentally consume gold arguments, intermediate answers, or code as inputs.
