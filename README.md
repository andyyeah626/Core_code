# OppoMem

Anonymous source release for the ICLR submission *OppoMem*. [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) maps the formal manuscript's method to the implementation.

## Core components

- `oppomem/agent.py`: player-visible opponent modeling, retrieval, and legal action selection.
- `oppomem/segmentation.py`, `oppomem/offline.py`, `oppomem/memory.py`, `oppomem/semantics.py`: training-trajectory segmentation, semantic memory construction, and response retrieval.
- `oppomem/activation.py`, `oppomem/utility.py`, `oppomem/evolution.py`, `oppomem/batch.py`: game-level randomized activation, utility comparison, and bank evolution between frozen training batches.
- `benchmarks/` and the corresponding `oppomem/*_adapter.py` files: RPS, IPD, Hanabi, and LLM Deliberation integration interfaces.

Memory grouping similarity, support criteria, configuration weights, game lengths, game counts, opponent schedules, and model choices are caller-controlled. The core provides no numerical grouping or memory-evolution cutoff. Retrieval selects the best available memory by the manuscript's similarity and reranking procedure, without an added numerical eligibility gate.

## Installing the source package

Use Python 3.10 or newer from the repository root:

```bash
python -m pip install -e .
```

Optional game engines and model weights are caller-provided. IPD uses Axelrod-Python; RPS and Hanabi use OpenSpiel; Deliberation reads the original game description files supplied by the caller. No credentials, checkpoints, game trajectories, or result tables are included. The release ZIP contains the core and adapters; local validation scripts and tests are kept outside that archive.

## Connecting a game

1. Build a `VisibleView` at each decision with legal actions and only information available to the acting participant. Keep another participant's private state and evaluator-only scores out of it.
2. Supply an opponent modeler and action policy to `OppoMemAgent`, along with a same-game memory bank and retriever. The method core selects at most one eligible response per modeled participant.
3. After a completed training game, call `finish_episode` and create `TrainingRecord`s from visible events. Update the bank between batches. Keep validation and test outcomes out of memory construction and updates.
4. Supply the run-specific game sizes, memory grouping setting, support rule, and opponent schedule when configuring an experiment. The repository does not set these to published-run values.

`MemoryBank.save` and `MemoryBank.load` use JSON. An episode snapshots its bank at start; a training batch stays frozen until its update.

### Game interfaces

- **RPS:** `benchmarks.roshambo.RoshamboAdapter` connects an externally installed OpenSpiel Roshambo bot. `RPSCallbackAdapter` connects a caller-supplied opponent callback that receives completed rounds but cannot see the current simultaneous focal action. `oppomem.rps_adapter.play_episode` supplies OppoMem decisions and training evidence.
- **IPD:** `benchmarks.axelrod_ipd.AxelrodAdapter` and `oppomem.ipd_adapter.play_episode` connect the Axelrod game to the shared core.
- **Hanabi:** `benchmarks.hanabi.HanabiAdapter` accepts a caller-provided partner policy. `oppomem.hanabi_adapter.OppoMemHanabiPartner` permits a second OppoMem seat and passes that seat only its own observation and completed public actions. The included rule partner validates the interface; game-specific partner assets are not bundled.
- **Deliberation:** `benchmarks.deliberation_protocol.Game` reads the caller's original game descriptions and configured participant profiles. `oppomem.deliberation_adapter.play_episode` models each other participant from public utterances while keeping their private plans and score tables outside the focal modeler's view.

This is a core-code and adapter release. It does not include a numerical experiment launcher or claim to reproduce any manuscript result. `LICENSE` and `THIRD_PARTY_NOTICES.md` give reuse terms and upstream attribution.
