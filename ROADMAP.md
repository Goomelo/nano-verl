# Roadmap

## Near Term

- add persisted run traces to `outputs/`
- add more non-math reward examples
- add a small benchmark/eval report format
- improve `vLLM` backend validation and error reporting
- add architecture docs for actor / rollout / reward separation

## Mid Term

- implement a native single-process GRPO loop for learning purposes
- add a reference-model path and KL accounting
- add richer training metrics and checkpoint metadata
- support local multi-GPU experiments

## Long Term

- distributed actor / rollout / reward workers
- real Megatron training backend
- more algorithms beyond the current GRPO-oriented path
- closer alignment with `verl` concepts while staying readable

