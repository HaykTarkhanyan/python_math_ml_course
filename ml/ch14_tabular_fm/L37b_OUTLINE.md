# L37b — What changed after TabPFN

Companion to L37. The subject is the model line and one educational follow-up, not a leaderboard survey. Source PDFs are in `papers/`.

1. Cold open: L37's 10,000-row boundary has moved.
2. Outline.
3. The release line: v1, v2, 2.5, 3, 3.5; distinguish design range from benchmark reach.
4. TabPFN-2.5: 50,000 rows and 2,000 features; distillation for repeated inference.
5. TabPFN-3: cached inference and row chunking, up to one million rows; many-class and task-specific variants; test-time computation.
6. TabPFN-3.5: grouped/temporal, strings, high-cardinality and multimodal settings; changes to encodings, model width and synthetic prior; benchmark claims and caveats.
7. nanoTabPFN: a small implementation for learning and experimentation, not a substitute for the production checkpoints.
8. Recap: choose the version and compare on the actual split; paper limits are dated claims.

No model execution. Figures and tables are schematics or numbers reported by the cited papers.
