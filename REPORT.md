# GhostPlayer: Full-Trajectory Player Movement Prediction in NFL Passing Plays

## 1. Project Description and Motivation

GhostPlayer is a supervised learning system for modeling expected player movement during NFL passing plays. Given a short pre-throw tracking window, the system predicts a future trajectory for each player marked by the dataset as `player_to_predict`. The predicted trajectory is treated as a “Ghost” path: an estimate of where the player would likely move under league-observed pass-play behavior. Comparing the actual path with the Ghost path gives an interpretable movement-deviation signal.

The motivation is football analytics. Coaches and analysts often want to understand whether a receiver, defender, or route runner moved in a typical way for the play context. A single end-point prediction is useful, but a full path is more informative because pass-play decisions unfold continuously. A receiver may separate early then converge late; a defender may take the correct initial leverage but fail after a route break; a passer or route-runner context may change the expected path. A trajectory-level Ghost model can visualize these differences as paths rather than isolated dots.

This project should not be interpreted as an “optimal football strategy” engine. It is an imitation-learning model trained on observed NFL tracking data. It learns likely movement patterns from historical examples, not counterfactual optimal movement. That distinction matters for both technical interpretation and ethical use: the model can surface unusual movement relative to the data distribution, but it does not prove that a player made a mistake.

## 2. Data and Learning Problem

The project uses the NFL Big Data Bowl 2026 Analytics dataset stored locally under:

```text
data/114239_nfl_competition_files_published_analytics_final/
```

The main files are:

```text
supplementary_data.csv
train/input_2023_w*.csv
train/output_2023_w*.csv
```

The 2026 dataset differs from a full player-tracking dataset. It does not provide all 22 players plus the ball for every play. Instead, each play contains a focused player subset relevant to a pass play. The input files contain pre-output tracking context for selected offensive and defensive players. The output files contain future `(x, y)` labels only for players marked `player_to_predict`. Therefore, the learning problem is not “predict all 11 defenders.” The correct formulation is:

```text
Input X: 10 historical frames of selected pass-play context
Label Y: future x/y trajectory for player_to_predict nodes
Learn f_theta: X -> Y
```

After preprocessing, each example is represented as a fixed 23-node graph:

- Up to 22 player-context nodes from the 2026 files.
- One ball-landing context node containing `ball_land_x` and `ball_land_y`.
- Zero padding for absent nodes.
- A prediction mask indicating which player nodes have supervised trajectory labels.

Each player node has six continuous tracking features across the 10-frame history:

```text
x, y, s, a, o, dir
```

The model also receives categorical embeddings for `player_position` and team type derived from `player_side` (`Offense`, `Defense`, or `Ball landing`). Field orientation is normalized so all plays are represented left-to-right. This is important because otherwise the model would need to learn separate mirrored movement patterns for left-moving and right-moving plays.

The processed split sizes are:

| Split | Examples |
|---|---:|
| Train | 9,727 |
| Validation | 2,187 |
| Test | 2,181 |

Each example has shape:

```text
history_continuous: (10 frames, 23 nodes, 6 features)
target_trajectories: (94 future frames, 23 nodes, 2 coordinates)
target_trajectory_mask: (94 future frames, 23 nodes)
```

Although the model outputs 94 future frames for every example, not every play has actual labels for all 94 frames. The mask records which future labels exist. Evaluation is computed only where actual labels exist. The dashboard still displays unscored Ghost forecasts beyond the last available actual frame, but these should be interpreted as unverified model forecasts rather than measured accuracy.

Important data quality and preprocessing issues include:

- Variable number of context players per play, handled by fixed-size padding.
- Variable future-label length, handled by a trajectory mask.
- Missing labels for non-target context players, handled by excluding them from the loss.
- Game-level splitting to reduce leakage across train, validation, and test sets.
- Reuse of the legacy field name `defender_mask` in code, which semantically now means `player_to_predict`.

## 3. Methodology

The project compares a simple baseline model with a spatio-temporal graph neural network. The baseline predicts one target player at a time using only that player’s flattened 10-frame feature history. It does not see teammate, opponent, role, or ball-landing context. This provides a useful lower bound: if the graph model cannot beat this baseline, the relational architecture is not adding practical value.

The advanced model is a Spatio-Temporal Graph Attention Network (ST-GAT). It is designed for the structure of the problem: football tracking is both spatial and temporal. Spatially, each player’s expected movement depends on nearby receivers, defenders, offensive/defensive side, route roles, and ball landing location. Temporally, the previous 10 frames define the player’s recent direction, speed trend, and route phase. A flat MLP loses most of this relational structure.

For each frame, the model builds node features by concatenating:

```text
continuous tracking features
ball/context active flag
player-position embedding
team-type embedding
```

The node features are projected into a hidden representation and passed through dense multi-head graph attention layers. The graph is fully connected within each frame, allowing every node to attend to every other node. The attention layer follows the central idea of Graph Attention Networks: instead of assigning a fixed weight to each neighbor, the model learns attention weights based on node features, making the relational aggregation adaptive to the play context [1].

After spatial encoding, each node’s 10 framewise embeddings are passed through a GRU temporal encoder. GRUs are appropriate here because they are lightweight recurrent units designed to model sequential dependencies with gating [2]. The final hidden state for each node is passed through an output head that predicts:

```text
94 future frames x 2 coordinates
```

The full-trajectory model is a direct multi-horizon predictor. It does not autoregressively roll forward one frame at a time. This is important because future output files contain only future `(x, y)` labels, not full future rows with `s`, `a`, `o`, `dir`, and all context-player states. A repeated one-step simulator would need to invent those missing future features. Direct multi-horizon prediction instead learns:

```text
10-frame context -> complete future x/y path
```

The objective is masked coordinate regression. Let `M_i` be the binary mask over valid target-player future labels. The empirical objective is:

```math
\min_{\theta}
\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{|M_i|}
\sum_{(t,v)\in M_i}
\left\| f_{\theta}(X_i)_{t,v} - Y_{i,t,v} \right\|_2^2
+ \lambda \|\theta\|_2^2
```

The loss `L` is masked squared Euclidean coordinate error. ADE is used for reporting because it is measured in yards and is easier to interpret. The optimizer is AdamW, an Adam-style adaptive first-order optimizer with decoupled weight decay. Adam is suitable for noisy minibatch objectives and large neural networks because it adapts learning rates using gradient moment estimates [3]. Regularization includes dropout in the GAT/input/output layers, weight decay, validation-based checkpoint selection, and gradient clipping. ReLU activations are used in hidden projections and the output MLP because they are simple, computationally efficient, and avoid the saturation behavior of sigmoid-like nonlinearities.

The advanced part of this methodology is the combination of graph attention with direct full-horizon trajectory forecasting. The graph attention component is appropriate because football movement is relational and the important neighbors vary by play. The direct multi-horizon head is appropriate because the dataset provides future positions but not the full future state required for clean autoregressive simulation.

## 4. Results and Discussion

The legacy one-frame evaluation compares the baseline against the previous single-frame ST-GAT checkpoint on the same held-out test split:

| Model | Count | ADE | Median Error | P90 | P95 | Max Error |
|---|---:|---:|---:|---:|---:|---:|
| Baseline MLP | 7,192 | 2.347 | 2.122 | 3.811 | 4.721 | 17.959 |
| ST-GAT, frame 1 | 7,192 | 1.403 | 1.203 | 2.604 | 3.157 | 9.078 |

The one-frame ST-GAT improves ADE by about 0.945 yards on average and beats the baseline on 86.5% of held-out plays. This supports the core hypothesis that relational context improves prediction over a player-only baseline.

The current trajectory model is evaluated over all available future labels:

| Model | Count | Full-Trajectory ADE | Median Error | P90 | P95 | Max Error |
|---|---:|---:|---:|---:|---:|---:|
| ST-GAT trajectory | 88,144 | 2.238 | 1.823 | 4.211 | 5.376 | 24.161 |

The trajectory model’s best validation ADE is 2.218 yards. On test data, full-horizon ADE is 2.238 yards, which is close to validation and does not suggest a severe validation-test gap. Errors grow as the forecast horizon increases, which is expected because uncertainty increases farther into the future and because later frames have fewer labeled examples. Selected horizon-level results:

| Output Frame | Label Count | ADE |
|---:|---:|---:|
| 1 | 7,192 | 1.766 |
| 5 | 7,192 | 1.687 |
| 10 | 4,443 | 2.234 |
| 15 | 1,768 | 3.319 |
| 19 | 943 | 4.050 |

The trajectory model’s frame-1 ADE is worse than the separate one-frame ST-GAT checkpoint. This is not surprising: the trajectory model optimizes a harder 94-frame objective, so it distributes capacity across many future horizons rather than specializing only on the first output frame. The practical gain is that it produces a complete Ghost path, which is more useful for visual analysis.

The current dashboard supports this interpretation. It shows actual paths, scored Ghost paths where labels exist, and unscored Ghost forecasts beyond the actual label horizon. This is an important presentation choice: it makes clear which predictions are evaluated and which are extrapolations.

Training and validation loss curves are the main missing artifact in the current report draft. The training script prints epoch metrics and stores the best validation ADE in the checkpoint, but it does not yet persist a full epoch history. For the final report, the training script should be rerun with CSV logging enabled, or stdout should be saved, so the report can include curves of training loss and validation ADE. Based on the saved endpoint metrics, the model does not appear to catastrophically overfit, but a curve is needed to diagnose whether training has plateaued, whether the model is underfitting early horizons, or whether late-horizon labels cause high-variance validation behavior.

## 5. Safety, Security, and Ethics

If deployed at large scale by a team, broadcaster, betting company, or league partner, GhostPlayer would raise several concerns.

First, the model can be misinterpreted as a measure of player quality or effort. A large deviation from the Ghost path does not automatically mean the player made an error. The actual player may have had an assignment not visible in the features, reacted to a subtle cue, improvised correctly, or been affected by physical contact. The model learns historical averages, not the playbook or the coach’s intent. Any deployment should present predictions as descriptive analytics, not definitive grading.

Second, the dataset is incomplete for full tactical interpretation. The 2026 files contain a selected player subset, not full 22-player tracking for every play. The model may miss off-screen or excluded context that explains a target player’s movement. It also predicts only future `(x, y)`, not body orientation, acceleration, or decision intent. This limits its suitability for automated personnel decisions.

Third, there is potential labor and fairness risk. If used in scouting, contract negotiation, or player evaluation, the model could reinforce historical biases in the data. For example, players in different schemes, roles, or coverage responsibilities may be penalized because their movement differs from the learned league distribution. Analysts should stratify results by position, role, route, coverage, and team context before drawing player-level conclusions.

Fourth, there are security and integrity concerns. A model like this could become valuable competitive intelligence. If used by a team, access to trained checkpoints, processed tracking data, and dashboard outputs should be controlled. Model outputs should not be exposed in a way that leaks proprietary strategy or player tendencies.

Finally, uncertainty should be communicated clearly. Longer-horizon Ghost forecasts are less reliable, and forecasts beyond available labels are not scored. The dashboard’s separation between scored Ghost path and unverified forecast is therefore not just a design feature; it is an ethical requirement for preventing overclaiming.

## 6. Conclusion

GhostPlayer demonstrates that relational, temporal modeling is useful for NFL pass-play movement prediction. The one-frame ST-GAT substantially outperforms a player-only baseline, suggesting that graph context matters. The upgraded trajectory model extends the system from single-point prediction to full future Ghost paths, making the output more interpretable and more aligned with football analysis workflows.

The current system is a strong research prototype, not a production grading tool. The main next steps are to persist training curves, add richer metadata-based error breakdowns, compare against a trajectory baseline, and calibrate uncertainty over future frames. With those additions, the project would provide a clearer scientific evaluation of when Ghost trajectories are reliable and where the dataset limits the claims.

## References

[1] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. “Graph Attention Networks.” ICLR 2018. https://openreview.net/forum?id=rJXMpikCZ

[2] Cho, K. et al. “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.” EMNLP 2014 / arXiv:1406.1078. https://huggingface.co/papers/1406.1078

[3] Kingma, D. P. and Ba, J. “Adam: A Method for Stochastic Optimization.” arXiv:1412.6980. https://arxiv.org/abs/1412.6980

[4] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.” JMLR 2014. https://www.jmlr.org/papers/v15/srivastava14a.html
