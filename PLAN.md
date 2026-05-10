# Project Plan: NFL Ghost Player

## Objective

Build a spatio-temporal graph learning pipeline that models expected defensive behavior during NFL passing plays. The model predicts the next-frame `(x, y)` positions for all 11 defenders jointly, using the previous 10 frames of player-tracking data. The predicted defensive positions are the "Ghost" positions. Deviation between actual and predicted defender locations is treated as scheme deviation.

This project is not estimating a counterfactual optimum. It is a deep imitation learning setup trained on actual NFL tracking data, where the model learns league-observed defensive behavior.

## Problem Definition

### Prediction target

- Input: previous 10 frames of a passing play segment
- Output: next-frame `(x, y)` positions for all 11 defenders
- Training labels: actual next-frame defender positions
- Inference interpretation: model output is expected defensive positioning under league-observed behavior

### Play scope

- Dataset: NFL Big Data Bowl 2025 tracking data from Kaggle
- Keep only pass plays where a forward pass was attempted
- Truncate each play to the interval from `ball_snap` through the first of:
  - `pass_arrived`
  - `pass_outcome_caught`
  - `pass_outcome_incomplete`
  - `interception`
- If the exact event strings differ in the released CSVs, update the parser to the dataset's canonical event names without changing the modeling scope

### Success criteria

- The advanced GNN outperforms the baseline on held-out Average Displacement Error (ADE)
- The dashboard can visualize held-out plays with actual versus predicted defender positions

## Modeling Decisions

### Graph definition

- Nodes: 22 players plus the ball
- Node set: fixed across frames
- Loss: computed only on the 11 defender nodes
- Edges: fully connected within each frame
- Edge weighting: learned implicitly through Graph Attention layers

### Node features

Each node should include:

- `x`
- `y`
- `s`
- `a`
- `o`
- `dir`
- role or position encoding
- offense/defense/ball type indicator
- ball active flag

Notes:

- Use role or position encoding only; do not use player identity embeddings in the first version
- The ball should always exist as a node, but be feature-flagged as active or inactive depending on pass-flight status and event context

### Temporal definition

- Input lookback: 10 frames
- Prediction horizon: 1 frame ahead
- Use the same 10-frame history for both the baseline and the GNN to keep comparisons fair

## Dataset and Preprocessing

### Core preprocessing tasks

1. Load tracking, plays, games, players, and relevant metadata tables from the Kaggle release
2. Filter to qualifying pass plays
3. Normalize field orientation so the offense always moves left to right
4. Identify the valid frame window from `ball_snap` to pass-arrival outcome
5. Build fixed-node frame representations for 22 players plus ball
6. Encode defender targets as next-frame positions only
7. Serialize examples into training-ready artifacts

### Data quality rules

- Drop plays missing required snap or terminal pass events
- Drop sequences shorter than 11 total usable frames
- Enforce consistent node ordering within a play
- Record missing-value handling explicitly for `s`, `a`, `o`, and `dir`
- Preserve enough metadata to recover `gameId`, `playId`, `frameId`, and player role mappings for visualization

### Train/validation/test split

- Split by `gameId`, not by frame or by play fragment
- Keep validation and test games disjoint from training games
- Freeze split logic early and reuse it for all experiments

## Model Plan

### Baseline model

Purpose: establish a simple non-relational benchmark.

- Unit of prediction: one defender at a time
- Input: that defender's own previous 10-frame feature history
- Output: next-frame `(x, y)`
- Recommended first implementation: MLP over flattened history or a small sequence model
- No information from teammates, offensive players, or ball interactions beyond what is in that defender's own history

Baseline evaluation should aggregate error across all defender predictions so it can be compared directly to the GNN.

### Advanced model

Purpose: model joint defensive behavior with relational context.

- Architecture: Spatio-Temporal Graph Attention Network
- Spatial module: stacked `GATConv` layers applied per frame
- Temporal module: `GRU` or `LSTM` over framewise graph embeddings
- Output head: predicts next-frame `(x, y)` for all 11 defenders jointly
- Loss: MSE over defender coordinates only

The GNN should exploit:

- defender-defender coordination
- defender-offense reactions
- ball context
- formation and route-driven spatial dependencies

## Evaluation Plan

### Primary metric

- ADE on held-out plays

For this project, ADE should be computed over defender predictions only.

### Secondary analytics

- Per-play Ghosting Error
- Per-defender Ghosting Error
- Worst-deviation plays for qualitative review
- Error by defensive role if role labels are reliable

### Comparison standard

- Baseline and GNN must use the same split
- Baseline and GNN must use the same 10-frame input window
- Report both aggregate metrics and a small number of held-out visual case studies

## Visualization Plan

Build a Streamlit app for final presentation.

### Required features

- Select a held-out `gameId` and `playId`
- Animate frame-by-frame movement over a football field
- Display:
  - offense
  - actual defense
  - predicted Ghost defense
  - ball location
- Show per-play summary metrics such as ADE or average Ghosting Error

### Presentation goal

Use the dashboard to show both:

- normal plays where predicted and actual defense align closely
- high-deviation plays where the model identifies unusual defensive movement

## Repository Plan

Target structure:

```text
GhostPlayer/
├── data/                    # Raw Kaggle files, gitignored
├── processed/               # Serialized training artifacts, gitignored
├── src/
│   ├── data/
│   │   ├── load.py
│   │   ├── preprocess.py
│   │   ├── build_sequences.py
│   │   └── build_graphs.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── st_gat.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_gnn.py
│   │   └── losses.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── inference.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── config.py
│       └── schema.py
├── app/
│   └── streamlit_viz.py
├── notebooks/               # Optional exploration only
├── PLAN.md
└── pyproject.toml
```

## Milestones

### Milestone 1: Data contract and preprocessing

Deliverables:

- documented schema for node features and labels
- pass-play filtering
- play truncation from snap to pass-arrival outcome
- left-to-right normalization
- split-by-`gameId` logic

Acceptance criteria:

- can produce clean sequence examples with 10-frame history and 1-frame target
- can recover metadata for any serialized example

### Milestone 2: Graph serialization

Deliverables:

- fixed-node graph construction for 22 players plus ball
- defender-node mask for loss computation
- serialized dataset for model training

Acceptance criteria:

- a dataloader yields correctly shaped batches
- node ordering is stable within and across samples

### Milestone 3: Baseline model

Deliverables:

- per-defender baseline implementation
- baseline training script
- held-out ADE evaluation

Acceptance criteria:

- baseline trains end-to-end on Colab GPU or CPU-compatible fallback
- baseline outputs reproducible held-out metrics

### Milestone 4: ST-GAT model

Deliverables:

- graph attention spatial encoder
- temporal sequence module
- joint defender prediction head
- training and validation loop

Acceptance criteria:

- model trains end-to-end on serialized graph sequences
- model produces defender-only predictions with valid masking

### Milestone 5: Evaluation and analysis

Deliverables:

- ADE computation
- per-play Ghosting Error summaries
- side-by-side baseline versus GNN comparison

Acceptance criteria:

- results clearly show whether the GNN beats the baseline
- high-error plays can be extracted for visualization

### Milestone 6: Streamlit dashboard

Deliverables:

- held-out play selector
- animation of actual versus Ghost defense
- metric display for selected play

Acceptance criteria:

- app runs locally or in Colab-compatible deployment workflow
- at least several held-out plays render correctly end-to-end

## Implementation Priorities

1. Lock the dataset schema and event mapping
2. Get sequence extraction working before any model work
3. Build the baseline first to establish an evaluation floor
4. Train the GNN only after the baseline and metrics pipeline are stable
5. Build the dashboard after prediction artifacts are available

## Risks and Mitigations

- Event label mismatch in Kaggle data
  - Mitigation: isolate event mapping in preprocessing config
- Sequence fragmentation or missing tracking rows
  - Mitigation: apply strict filtering and log dropped plays
- Graph model memory cost from fully connected frames
  - Mitigation: start with small batch sizes and profile in Colab GPU
- Weak baseline comparison due to inconsistent setup
  - Mitigation: enforce identical history window, split, and coordinate normalization

## Out of Scope for First Version

- Counterfactual optimization of truly "ideal" defense
- Player identity embeddings
- Run-play modeling
- Report-writing and ethics write-up

## Final Deliverable Definition

The first successful version of this project is complete when:

- the data pipeline produces training-ready graph sequences from Big Data Bowl 2025 data
- the baseline produces held-out ADE
- the ST-GAT produces held-out ADE and beats the baseline
- the dashboard visualizes held-out actual versus Ghost defense on selected pass plays
