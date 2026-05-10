This is a comprehensive **AI-Agent Execution Blueprint**. You can paste these sections directly into a `PLAN.md` file in your repository or use them as high-level prompts for your coding agent (Cursor, Windsurf, or Cascade).

---

# 📜 Project Master Plan: NFL "Ghost Player" GNN Optimizer

**Course:** INDENG 1/242B (Spring 2026)

**Objective:** Use a Spatio-Temporal Graph Neural Network (GNN) to predict the "ideal" positioning of NFL defensive players during passing plays.

---

## 🏗 1. Technical Architecture

We will treat each play as a dynamic graph.

* **Nodes:** 22 players + the ball.
* **Features:** $(x, y)$ coordinates, speed ($s$), acceleration ($a$), orientation ($o$), and direction ($dir$).
* **Edges:** Fully connected graph with "Attention Weights" to determine player influence.
* **The "Advanced" Component:** A **Spatio-Temporal GNN**.
* **Spatial:** Graph Attention Layers (GAT) to model player-to-player relationships.
* **Temporal:** A Gated Recurrent Unit (GRU) or LSTM to model movement over time.



---

## 📂 2. File Structure (Standardize this first!)

Tell your AI agent to initialize this structure:

```text
nfl-ghosting-gnn/
├── data/                  # Raw Kaggle CSVs (gitignored)
├── processed/             # Cleaned .pt files for PyTorch Geometric
├── src/
│   ├── data_pipeline.py   # Data cleaning & Graph construction
│   ├── models/
│   │   ├── baseline.py    # Simple MLP/LSTM (for comparison)
│   │   └── gnn_model.py   # Advanced GAT + GRU architecture
│   ├── training.py        # Training & Validation loops
│   └── evaluation.py      # Metrics (ADE, FDE) and visualization
├── app/
│   └── streamlit_viz.py   # Interactive dashboard (Bonus 5%)
├── PLAN.md                # This file
└── requirements.txt

```

---

## 🚀 3. Phase-by-Phase AI Prompts

### Phase 1: Data Engineering (The Foundation)

**Prompt for AI Agent:**

> "Initialize the project structure. Create `data_pipeline.py`. Download the NFL Big Data Bowl 2025 dataset. Write a script to:
> 1. Filter for 'pass' plays only.
> 2. Normalize coordinates so the offense always moves from left to right.
> 3. Convert each frame into a PyTorch Geometric `Data` object where nodes are players and features include position, speed, and orientation.
> 4. Save the output as a processed dataset."
> 
> 

### Phase 2: The Baseline (Required for Report)

**Prompt for AI Agent:**

> "In `models/baseline.py`, implement a simple Multi-Layer Perceptron (MLP) and a standard LSTM. These will serve as our comparison models. The goal is to predict a defensive player's next $(x, y)$ coordinate based on the previous 10 frames of all players. Write the training loop in `training.py`."

### Phase 3: The Advanced GNN (The "A" Grade)

**Prompt for AI Agent:**

> "In `models/gnn_model.py`, implement a Spatio-Temporal GNN using `torch_geometric`.
> 1. Use `GATConv` layers to allow players to 'attend' to relevant opponents.
> 2. Wrap the GNN output in a `GRU` layer to capture the momentum of the play.
> 3. Ensure the model is permutation-invariant (the order of nodes doesn't change the result)."
> 
> 

### Phase 4: Visualization & Metrics

**Prompt for AI Agent:**

> "Create `evaluation.py`. Calculate the **Average Displacement Error (ADE)** between our 'Ghost' (predicted) and the 'Human' (actual).
> Create a Matplotlib animation function that shows a top-down view of the field:
> * Blue dots = Offense
> * Red dots = Actual Defense
> * Ghost/Shadow dots = Model's predicted 'Ideal' Defense."
> 
> 

---

## ⚖️ 4. Ethics & Safety "Cheat Sheet" (For the Report)

The project description requires a section on ethics. Have your AI agent draft this based on these points:

* **Safety:** If a coach uses this model, they might demand a player move to a spot that is physically dangerous (e.g., a "suicide" route or a collision course). The model doesn't understand human physical limits.
* **Bias:** The model is trained on NFL pros. Applying it to college or high school players could cause over-exertion or injury because the "ideal" model assumes elite speed/stamina.
* **Privacy:** Discuss the ethics of tracking every micro-movement of a human being at their workplace.

---

## 🛠 5. The "Bonus 5%" Strategy

To get the extra credit, have your agent build a simple **Streamlit App**:
**Prompt for AI Agent:**

> "Create `app/streamlit_viz.py`. Build a dashboard where a user can select a specific `gameId` and `playId`. Use a slider to move through the timestamps of the play and display the GNN's 'Ghost' positioning in real-time. Show a 'Performance Score' for the defender based on how close they were to the Ghost."

---
