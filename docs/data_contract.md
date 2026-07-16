# Normalized data contract

Each source adapter emits three JSONL tables. IDs must be stable across reruns.

```json
{"tool_id":"weather.lookup","name":"weather_lookup","description":"...","schema":{"type":"object","properties":{}},"source":"bfcl_v4"}
{"decision_id":"d1","query":"...","candidate_tool_ids":["weather.lookup"],"gold_tool_id":"weather.lookup","chosen_tool_id":"weather.lookup","source":"bfcl_v4"}
{"trace_id":"t1","tool_ids":["calendar.find","calendar.create"],"source":"appworld"}
```

`chosen_tool_id` may be null before model rollouts. `trace_id` is optional for
single-turn datasets. Paper 1 only reads these fields; Paper 2 can add
`variant_of`, `is_fusion`, and `mode`; Paper 3 can add `turn`, `injected`, and
`trajectory_id` without breaking the contract.

Adapters accept either a single JSONL whose rows already follow this shape, or
a directory containing `tools.jsonl`, `decisions.jsonl`, and `traces.jsonl`.
