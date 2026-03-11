# DRO_Circuit

`dro_circuit` is a research repo for robust circuit discovery in transformers.
The current implementation centers on IOI with GPT-2 and combines:

- corruption-specific edge scoring with EAP or EAP-IG
- DRO-style aggregation across corruption families
- circuit selection with `topn` or `greedy`
- robust evaluation under every corruption used during discovery

Use the getting started guide for environment setup, the first experiment run,
and the current repository layout.
