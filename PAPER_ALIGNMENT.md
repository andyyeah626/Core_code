# Paper-to-code map

This anonymous source release follows the full OppoMem manuscript, including its method in Section 4 and the four game settings. It exposes the method and player-visible integration boundaries. It deliberately leaves training and evaluation schedules, game lengths, seeds, model endpoints, and numerical grouping or decision cutoffs to the caller. They are not encoded as release defaults or listed here.

## Method core

- `oppomem/types.py` defines each memory as an opponent-type description paired with a response strategy. `oppomem/segmentation.py`, `oppomem/offline.py`, and `oppomem/memory.py` create supported entries from completed, player-visible training evidence.
- `oppomem/agent.py` updates a separate description for each modeled participant at a decision, retrieves at most one eligible response per participant, and sends only the acting player's view to the action policy.
- `oppomem/semantics.py` and `oppomem/memory.py` implement E5 embedding, actual-member medoids, and directional DeBERTa NLI reranking. The retriever takes the highest-ranked item; the manuscript does not specify a numerical retrieval gate.
- `oppomem/activation.py` samples a training-game activation on a memory's first retrieval and keeps that decision fixed for the game. `oppomem/utility.py` compares enabled and disabled terminal returns within each opponent configuration, with one return per retrieved memory per game.
- `oppomem/batch.py` and `oppomem/evolution.py` keep a bank frozen within a batch and permit keep, revise, or delete updates between batches. The included LLM review policy uses completed training evidence without preset numerical cutoffs; callers supply support criteria and configuration weights. Validation and test episodes cannot construct or revise memories.

## Game integration

- **RPS:** `benchmarks/roshambo.py` supplies the OpenSpiel Roshambo fixed-bot interface and an opponent callback that receives only completed rounds, never the current simultaneous action. `oppomem/rps_adapter.py` connects decisions and training evidence to the method core.
- **IPD:** `benchmarks/axelrod_ipd.py` and `oppomem/ipd_adapter.py` bridge Axelrod's simultaneous game to the same core. The caller provides game and training sizes.
- **Hanabi:** `benchmarks/hanabi.py` supplies the acting seat's observation and public actions. `oppomem/hanabi_adapter.py` connects a focal OppoMem agent and can instantiate a second OppoMem seat for same-method self-play. The paper's external training partner policy must be supplied by the caller; the included rule partner is an interface example.
- **LLM Deliberation:** `benchmarks/deliberation_protocol.py` reads the original game description and configured cooperative or greedy participant profile. `oppomem/deliberation_adapter.py` maintains separate descriptions of the other participants from public messages while excluding their private plans and score tables.

The release does not claim that its integration interfaces generated the manuscript's reported scores.

## Scope

The release archive contains only the inspectable core, adapters, and necessary package documents. It does not contain a launcher for the reported experiments, local tests or examples, model credentials, upstream game assets, trajectories, or result tables. Game-specific prompts in these adapters use a structured JSON client; they do not claim to reproduce every appendix prompt byte for byte. A reviewer can connect their own game policies and run settings through the published interfaces.
