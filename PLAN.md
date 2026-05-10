# Project Plan: NFL Ghost Player

## Objective

Build a spatio-temporal graph learning pipeline that models expected player movement during NFL passing plays. The model uses recent pre-throw tracking context to predict future `(x, y)` locations for the players marked by the dataset as `player_to_predict`.

The predicted locations are the "Ghost" positions. Deviation between actual and predicted locations is treated as movement deviation from league-observed behavior.

This project is not estimating a counterfactual optimum. It is a deep imitation learning setup trained on actual NFL tracking data, where the model learns likely movement under observed passing-play situations.

## Dataset Reality

### Current dataset

- Dataset: NFL Big Data Bowl 2026 Analytics files from Kaggle
- Local data folder: `data/114239_nfl_competition_files_published_analytics_final/`
- Main files:
  - `supplementary_data.csv`
  - `train/input_2023_w*.csv`
  - `train/output_2023_w*.csv`

### Key schema difference from the original idea

The 2026 dataset does not provide all 22 players plus ball for every play. It provides a focused player subset around the pass play:

- `input_*.csv`: pre-output tracking context for selected offensive and defensive players
- `output_*.csv`: future `(x, y)` labels only for players marked `player_to_predict`
- `supplementary_data.csv`: play metadata such as teams, formation, coverage, route, pass result, and game info

Therefore, the first version should predict the dataset-provided target players, not all 11 defenders.

The original full-defense goal can remain a future extension if we later obtain full 22-player tracking for the same passing plays.

## Problem Definition

### Prediction target

- Input: previous 10 frames from the 2026 `input_*.csv` tracking context
- Output: full future `(x, y)` trajectory for nodes whose source rows have `player_to_predict == True`
- Training labels: matching rows from `output_*.csv`; `frame_id == 1` is still kept as the compatibility one-frame target
- Inference interpretation: model output is expected movement for the target players under league-observed passing-play behavior

### Play scope

- Keep passing plays represented in the 2026 input/output files
- Use `supplementary_data.csv` for game/play metadata and split-by-game logic
- Do not parse `ball_snap` or pass outcome events for the 2026 path; the competition already gives input and output phases
- Normalize field orientation so all examples are represented as moving left to right

### Success criteria

- The advanced graph model outperforms the baseline on held-out Average Displacement Error (ADE)
- The dashboard can visualize held-out plays with actual versus predicted target-player locations
- Metadata clearly states that the prediction mask is `player_to_predict`, not "all defenders"

## Modeling Decisions

### Graph definition

- Nodes: up to 22 provided player-context nodes plus one ball-landing context node
- Node count: fixed at 23 through zero-padding
- Player node ordering:
  - prediction targets first
  - then contextual offensive/defensive players
  - stable ordering by side, role, position, and `nfl_id`
- Last node: ball landing context from `ball_land_x`, `ball_land_y`
- Loss: computed only on `player_to_predict` nodes
- Edges: fully connected within each frame
- Edge weighting: learned implicitly through Graph Attention layers

### Node features

Each player node should include:

- `x`
- `y`
- `s`
- `a`
- `o`
- `dir`
- position encoding from `player_position`
- offense/defense type indicator from `player_side`
- target mask from `player_to_predict`

The ball-landing context node should include:

- `x = ball_land_x`
- `y = ball_land_y`
- zero speed/acceleration/orientation features
- ball/context type indicator
- ball active flag

Notes:

- Use role or position encoding only; do not use player identity embeddings in the first version
- `GraphDataset.defender_mask` is currently reused as the prediction mask for compatibility with existing code, but semantically it means `player_to_predict`

### Temporal definition

- Input lookback: 10 frames
- Prediction horizon: all available output frames from the 2026 `output_*.csv` files
- Use the same 10-frame history for baseline and GNN
- Keep first-frame targets for baseline comparison and dashboard compatibility

## Dataset and Preprocessing

### Core preprocessing tasks

1. Load `supplementary_data.csv`, all `input_2023_w*.csv`, and all `output_2023_w*.csv`
2. Split by `game_id`, not by frame, player, or play
3. Normalize field orientation so all examples face left to right
4. Select the last 10 input frames per play
5. Build fixed-size graph tensors with 23 nodes
6. Build prediction masks from `player_to_predict`
7. Encode first-frame labels plus full future trajectory labels from matching `output_*.csv` rows
8. Serialize examples into training-ready `.npz` artifacts

### Data quality rules

- Drop plays with fewer than 10 input frames
- Drop plays with no valid `player_to_predict` label in `output_*.csv`
- Keep a build summary with dropped-play counts
- Preserve enough metadata to recover `game_id`, `play_id`, input frame range, and player role mappings for visualization
- Record clearly that padded nodes are context placeholders and should not contribute to loss

### Train/validation/test split

- Split by `game_id`
- Keep validation and test games disjoint from training games
- Freeze split logic early and reuse it for all experiments

## Model Plan

### Baseline model

Purpose: establish a simple non-relational benchmark.

- Unit of prediction: one target player at a time
- Input: that target player's own previous 10-frame continuous feature history
- Output: first output-frame `(x, y)`
- Current implementation: MLP over flattened history
- No information from teammates, opponents, ball landing, formation, or coverage context

Baseline evaluation should aggregate error across all `player_to_predict` nodes so it can be compared directly to the GNN.

### Advanced model

Purpose: model target-player movement with relational context.

- Architecture: Spatio-Temporal Graph Attention Network
- Spatial module: stacked `GATConv` layers applied per frame
- Temporal module: `GRU` or `LSTM` over framewise graph embeddings
- Output head: predicts full future `(x, y)` trajectory for every node
- Loss: MSE over available future labels on `player_to_predict` nodes only

The GNN should exploit:

- receiver-defender relationships
- route-runner and coverage context
- passer and targeted receiver roles
- ball landing context
- formation, coverage, and pass metadata if added as graph-level features later

## Evaluation Plan

### Primary metric

- ADE on held-out target-player predictions

For the 2026 path, ADE is computed over `player_to_predict` nodes only.

### Secondary analytics

- Per-play Ghosting Error
- Per-player Ghosting Error
- Error by `player_side`
- Error by `player_role`
- Error by `player_position`
- Worst-deviation plays for qualitative review

### Comparison standard

- Baseline and GNN must use the same split
- Baseline and GNN must use the same 10-frame input window
- Report both aggregate metrics and several held-out visual case studies

## Visualization Plan

Build a Streamlit app for final presentation.

### Required features

- Select a held-out `game_id` and `play_id`
- Animate input context frames and prediction output
- Display:
  - offensive context players
  - defensive context players
  - actual target-player positions
  - predicted Ghost target-player positions
  - ball landing location
- Show per-play summary metrics such as ADE or average Ghosting Error

### Presentation goal

Use the dashboard to show both:

- plays where predicted and actual target-player movement align closely
- high-deviation plays where the model identifies unusual movement

## Repository Plan

Target structure:

```text
GhostPlayer/
├── data/                    # Raw Kaggle files, gitignored
├── processed/               # Serialized training artifacts, gitignored
├── src/
│   ├── ghostplayer/
│   │   ├── data/
│   │   │   ├── load.py
│   │   │   ├── bdb2026.py
│   │   │   ├── legacy.py
│   │   │   ├── preprocess.py
│   │   │   ├── build_sequences.py
│   │   │   └── build_graphs.py
│   │   ├── models/
│   │   │   ├── baseline.py
│   │   │   └── st_gat.py
│   │   ├── training/
│   │   │   ├── train_baseline.py
│   │   │   ├── train_gnn.py
│   │   │   └── losses.py
│   │   ├── eval/
│   │   │   ├── metrics.py
│   │   │   ├── inference.py
│   │   │   └── error_analysis.py
│   │   └── utils/
│   │       ├── config.py
│   │       └── schema.py
├── app/
│   └── streamlit_viz.py
├── notebooks/
│   ├── 01_download_and_process_kaggle_data.ipynb
│   └── 02_process_2026_data.ipynb
├── PLAN.md
└── pyproject.toml
```

## Milestones

### Milestone 1: 2026 data contract and preprocessing

Deliverables:

- documented 2026 schema contract
- input/output file loading
- left-to-right normalization
- split-by-`game_id` logic
- graph-ready serialization notebook

Acceptance criteria:

- can produce clean graph examples with 10-frame history, first-frame targets, and full future trajectory targets
- can recover metadata for any serialized example
- prediction mask correctly identifies `player_to_predict` nodes

Status: mostly implemented through `src/ghostplayer/data/bdb2026.py` and `notebooks/02_process_2026_data.ipynb`.

### Milestone 2: Graph serialization

Deliverables:

- fixed-size 23-node graph construction
- prediction mask for loss computation
- serialized dataset for model training

Acceptance criteria:

- graph dataset yields correctly shaped arrays
- node ordering is stable within each sample
- padded/context nodes do not contribute to loss

Status: implemented for the 2026 path.

### Milestone 3: Baseline model

Deliverables:

- per-target-player baseline implementation
- baseline training script
- held-out ADE evaluation

Acceptance criteria:

- baseline trains end-to-end on CPU-compatible fallback
- baseline outputs reproducible held-out metrics

Status: implemented and smoke-tested on 2026 week 1 graph artifacts.

### Milestone 4: ST-GAT model

Deliverables:

- graph attention spatial encoder
- temporal sequence module
- prediction head
- training and validation loop

Acceptance criteria:

- model trains end-to-end on serialized 2026 graph sequences
- model produces either first-frame or full-trajectory predictions with valid `player_to_predict` masking

Status: upgraded to support trajectory-horizon output in addition to the original one-frame head.

### Milestone 5: Evaluation and analysis

Deliverables:

- ADE computation
- per-play Ghosting Error summaries
- side-by-side baseline versus GNN comparison

Acceptance criteria:

- results clearly show whether the GNN beats the baseline
- high-error plays can be extracted for visualization

Status: implemented through `src/ghostplayer/eval/inference.py`, `src/ghostplayer/eval/error_analysis.py`, and `src/ghostplayer/eval/metrics.py`. Test split evaluation shows ST-GAT beating the baseline.

### Milestone 6: Streamlit dashboard

Deliverables:

- held-out play selector
- football field visualization
- actual versus Ghost target-player positions
- metric display for selected play

Acceptance criteria:

- app runs locally
- several held-out plays render correctly end-to-end

Status: implemented through `app/streamlit_viz.py` and verified locally at `http://localhost:8501`.

## Implementation Priorities

1. Use the 2026 pipeline as the primary data path
2. Generate full train/validation/test graph artifacts with `notebooks/02_process_2026_data.ipynb`
3. Train and record the baseline ADE
4. Implement the ST-GAT model
5. Compare baseline versus GNN on identical splits
6. Build the visualization app after prediction artifacts are available
7. Extend dashboard animation from first-frame Ghost markers to full Ghost paths

## Risks and Mitigations

- 2026 schema predicts focused player subsets rather than all defenders
  - Mitigation: explicitly define the first version around `player_to_predict`
- Node count varies by play
  - Mitigation: pad to 23 nodes and mask loss to prediction nodes
- Baseline and GNN could accidentally use different target masks
  - Mitigation: reuse serialized `defender_mask` as the canonical prediction mask
- Graph model memory cost from fully connected frames
  - Mitigation: start with small batch sizes and profile
- Original "all 11 defenders" framing no longer matches the data
  - Mitigation: present first version as focused pass-play ghosting; keep full-defense ghosting as future work

## Out of Scope for First Version

- Counterfactual optimization of truly "ideal" movement
- Player identity embeddings
- Full 11-defender prediction
- Run-play modeling
- Report-writing and ethics write-up

## Final Deliverable Definition

The first successful version of this project is complete when:

- the 2026 data pipeline produces train/validation/test graph artifacts
- the baseline produces held-out ADE
- the ST-GAT produces held-out trajectory ADE and first-frame ADE, and beats the baseline on the comparable first-frame target
- the dashboard visualizes held-out actual versus Ghost target-player positions on selected pass plays
