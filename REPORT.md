# GhostPlayer: Predicting Full Player Trajectories in NFL Passing Plays

## 1. Project Description and Motivation

For this project, I built **GhostPlayer**, a model that predicts where NFL players are expected to move during passing plays. The idea is simple: given the last few frames before the future play segment, the model draws a “ghost” path for each target player. We can then compare the real player path to the predicted Ghost path.

This is useful because football movement is hard to judge from one still frame. A receiver might look covered at the catch point but may have created separation earlier. A defender might start in a good position but lose leverage after a route break. A model that predicts a full path can show these differences better than a single predicted dot.

The goal is **not** to say what the player “should” have done in a perfect football sense. The model is not reading the playbook, and it does not know the coach’s exact assignment. Instead, this is an imitation learning problem: the model learns movement patterns from real NFL data and predicts what movement looks likely based on similar historical situations.

## 2. Data and Learning Problem

The project uses the NFL Big Data Bowl 2026 Analytics dataset. The local files are stored at:

```text
data/114239_nfl_competition_files_published_analytics_final/
```

The important files are:

```text
supplementary_data.csv
train/input_2023_w*.csv
train/output_2023_w*.csv
```

One important thing I had to adjust for is that the 2026 dataset is not a full 22-player tracking dataset. It gives a focused subset of players around the passing play. The input files contain the past/context frames, and the output files contain future `(x, y)` labels only for players marked as `player_to_predict`.

So the learning problem is:

```text
Input: 10 past tracking frames for selected players in a pass play
Output: future x/y trajectory for player_to_predict nodes
Learn: f_theta(X) -> Y
```

Each play is converted into a fixed graph with 23 nodes:

- Up to 22 player-context nodes.
- One ball-landing node using `ball_land_x` and `ball_land_y`.
- Padding when fewer than 22 player-context nodes exist.
- A mask that says which nodes should count toward the loss.

Each player node uses these continuous features from the previous 10 frames:

```text
x, y, s, a, o, dir
```

I also added categorical information through embeddings, including player position and team type. I normalized field direction so every play is represented as moving left-to-right. Without this step, the model would need to learn the same football pattern twice, once in each direction.

The processed dataset sizes are:

| Split | Examples |
|---|---:|
| Train | 9,727 |
| Validation | 2,187 |
| Test | 2,181 |

The final graph tensors look like this:

```text
history_continuous: (10 frames, 23 nodes, 6 features)
target_trajectories: (94 future frames, 23 nodes, 2 coordinates)
target_trajectory_mask: (94 future frames, 23 nodes)
```

The model predicts 94 future frames, but not every play has actual labels for all 94 frames. This is why the mask is important. We only score the model where actual future labels exist. In the dashboard, I still show the model’s forecast beyond the actual labels, but those parts are shown as unscored forecasts.

Some data issues that mattered:

- Plays have different numbers of context players.
- Future labels have different lengths depending on the play.
- Only `player_to_predict` nodes have labels, not every context player.
- Splitting must happen by `game_id` to avoid leakage.
- The code still uses the old name `defender_mask`, but in this version it really means `player_to_predict` mask.

## 3. Methodology

I used two main models: a simple baseline and a graph neural network.

The baseline is an MLP that predicts one target player at a time. It only sees that player’s own 10-frame history. This is useful because it tells us how much performance we can get without any relational context. If the graph model cannot beat this baseline, then the graph structure is not helping.

The main model is a **Spatio-Temporal Graph Attention Network (ST-GAT)**. This model fits the problem better because football movement depends on both space and time. Spatially, a player reacts to teammates, opponents, route roles, and ball location. Temporally, the last 10 frames tell us about speed, direction, and route development.

The model works in three main stages:

1. Build node features for each player and frame.
2. Apply graph attention layers within each frame.
3. Use a GRU over time to summarize the 10-frame history.

For each frame, the model combines:

```text
tracking features
ball/context flag
player-position embedding
team-type embedding
```

Then the model applies dense multi-head graph attention. I used graph attention because every play has a different relationship structure. A defender might need to pay attention to a receiver on one play, but to a different route runner or ball-landing point on another. Graph attention lets the model learn these weights instead of using fixed neighbor weights [1].

After the graph attention layers, each node has a sequence of 10 embeddings. A GRU processes this sequence and creates a final hidden state for each node [2]. The output head then predicts:

```text
94 future frames x 2 coordinates
```

One design choice I made was to predict the whole future trajectory directly instead of repeatedly predicting one frame at a time. Repeated one-step prediction sounds natural, but it is not a clean fit for this dataset. The future output files only give future `(x, y)` labels. They do not give future speed, acceleration, orientation, direction, or future context-player states. To roll the model forward step by step, we would need to invent all of those missing features. Direct trajectory prediction avoids that problem:

```text
10-frame context -> full future x/y path
```

The loss is masked coordinate regression. In words, the model only gets penalized for valid future labels on `player_to_predict` nodes. The objective is:

```math
\min_{\theta}
\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{|M_i|}
\sum_{(t,v)\in M_i}
\left\| f_{\theta}(X_i)_{t,v} - Y_{i,t,v} \right\|_2^2
+ \lambda \|\theta\|_2^2
```

Here, `M_i` is the mask of valid target-player labels. I used masked MSE for training and Average Displacement Error (ADE) for reporting because ADE is measured in yards and is easier to understand.

For optimization, I used AdamW. Adam-style optimization works well for neural networks because it adapts learning rates based on gradient statistics [3]. AdamW also gives weight decay in a cleaner way. For regularization, I used dropout, weight decay, validation checkpointing, and gradient clipping. Hidden layers use ReLU activations.

The advanced part of the project is the combination of **graph attention** and **direct multi-horizon trajectory prediction**. Graph attention handles relational football context, and the direct trajectory head lets the model output a full Ghost path instead of just one future point.

## 4. Results and Discussion

First, I evaluated the earlier one-frame model against the baseline. This compares both models on the first future frame only:

| Model | Count | ADE | Median Error | P90 | P95 | Max Error |
|---|---:|---:|---:|---:|---:|---:|
| Baseline MLP | 7,192 | 2.347 | 2.122 | 3.811 | 4.721 | 17.959 |
| ST-GAT, frame 1 | 7,192 | 1.403 | 1.203 | 2.604 | 3.157 | 9.078 |

The graph model clearly improves over the baseline. It lowers ADE by about 0.945 yards and wins on 86.5% of held-out plays. This suggests that relational context is actually useful for this task.

Then I evaluated the full-trajectory model:

| Model | Count | Full-Trajectory ADE | Median Error | P90 | P95 | Max Error |
|---|---:|---:|---:|---:|---:|---:|
| ST-GAT trajectory | 88,144 | 2.238 | 1.823 | 4.211 | 5.376 | 24.161 |

The best validation ADE for the trajectory model was 2.218 yards, and the test ADE was 2.238 yards. These numbers are close, which is a good sign that the model is not obviously failing on the test split.

The error increases farther into the future, which makes sense. The longer we predict, the more uncertainty there is. Also, there are fewer actual labels at later future frames, so late-frame metrics are based on fewer examples.

| Output Frame | Label Count | ADE |
|---:|---:|---:|
| 1 | 7,192 | 1.766 |
| 5 | 7,192 | 1.687 |
| 10 | 4,443 | 2.234 |
| 15 | 1,768 | 3.319 |
| 19 | 943 | 4.050 |

One interesting result is that the trajectory model’s frame-1 ADE is worse than the separate one-frame ST-GAT checkpoint. This is not too surprising because the trajectory model has a harder job. It is trained to predict many future frames, not just the first one. The tradeoff is that it gives us a full Ghost path, which is more useful for visualization and analysis.

The Streamlit dashboard makes this easier to understand. It shows:

- Actual target-player path in black.
- Scored Ghost path in cyan.
- Unscored future forecast as a lighter dotted cyan line.
- Error over future frames.
- A frame scrubber and full animation.

This helped me catch an important visualization issue: at first, the app stopped the Ghost path when the actual labels stopped. But the model still predicts all 94 frames. I changed the app so the Ghost forecast can continue past the labeled frames, while clearly marking that part as unscored.

One thing still missing from the current report is a proper training/validation curve. The training script prints epoch metrics and saves the best validation ADE, but it does not yet save a CSV history. For the final version, I would either rerun training with logging or save stdout so I can plot training loss and validation ADE over time. Based on the validation/test numbers, the model does not look severely overfit, but the curve would make that diagnosis stronger.

## 5. Safety, Security, and Ethics

If this type of model were used by a team, broadcaster, or betting company, it would need to be used carefully.

The biggest issue is interpretation. A player being far from the Ghost path does not automatically mean the player made a mistake. The model does not know the exact play call, assignment, coaching instruction, or communication on the field. A player might move differently from the Ghost because they saw something the model cannot see.

There are also fairness concerns. Players in different schemes or roles may naturally move differently. If the model is used for player evaluation, it could accidentally punish players for being in unusual systems or assignments. For this reason, I would not use this model as a direct grading tool without much more context and validation.

The dataset also limits what the model can claim. Since the 2026 data only includes selected player subsets, the model may miss important off-screen or excluded context. It predicts future `(x, y)` positions, but not intent, body control, or assignment correctness.

There are also security concerns. A system like this could reveal competitive insights if used by a team. The model checkpoints, processed data, and dashboard outputs should be protected if used in a real football organization.

Finally, uncertainty needs to be shown clearly. Longer-horizon predictions are less reliable, and forecasts beyond available labels are not scored. This is why the dashboard separates scored Ghost paths from unscored forecasts. Without that separation, it would be easy to overclaim what the model actually proved.

## 6. Conclusion

GhostPlayer shows that graph-based context helps predict NFL pass-play movement. The one-frame ST-GAT beats the player-only baseline, and the trajectory version extends the idea from a single Ghost point to a full Ghost path.

The current system is best viewed as a research prototype. It is useful for visual analysis and for studying movement deviations, but it is not a production player-grading system. The next improvements would be to save training curves, add more detailed error breakdowns by role and position, compare against a trajectory baseline, and add uncertainty estimates for long-horizon forecasts.

Overall, the project supports the main idea: football movement is relational and temporal, so a spatio-temporal graph model is a reasonable approach for predicting expected player trajectories.

## References

[1] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. “Graph Attention Networks.” ICLR 2018. https://openreview.net/forum?id=rJXMpikCZ

[2] Cho, K. et al. “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.” EMNLP 2014 / arXiv:1406.1078. https://huggingface.co/papers/1406.1078

[3] Kingma, D. P. and Ba, J. “Adam: A Method for Stochastic Optimization.” arXiv:1412.6980. https://arxiv.org/abs/1412.6980

[4] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.” JMLR 2014. https://www.jmlr.org/papers/v15/srivastava14a.html
