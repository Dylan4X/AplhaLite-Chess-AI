# Hierarchical Chess Engine: Project AlphaLite

> Bridging Classical Search and Deep Learning for Adaptive Human-AI Interaction

This project implements a **Hierarchical Chess Engine**, designed to bridge the gap between traditional heuristic search algorithms and modern Deep Learning approaches. By offering a spectrum of difficulty levels—from rule-based heuristics to AlphaZero-style neural networks—this system provides an adaptive environment for both education and competitive play.

The system features a custom **"Green AI"** neural architecture trained via Supervised Learning on the CCRL database, achieving high-level play on consumer hardware without the massive computational cost of reinforcement learning.

---

## 🌟 Key Features

- **Hierarchical Architecture**: Four distinct difficulty tiers ranging from naive material search to grandmaster-level neural inference
- **AlphaLite Engine**: A lightweight implementation of AlphaZero, combining Monte Carlo Tree Search (MCTS) with a ResNet backbone
- **Efficient Inference**: Utilizes a compact $16 \times 8 \times 8$ input representation and a ResNet-10 model to achieve an 82% non-loss rate against Stockfish Level 6
- **Interactive GUI**: A 120 FPS capable interface built with Flet, featuring real-time neural evaluation bars, automatic opening recognition, and move explainability

---

## 🧠 Engine Levels

The project includes four engines, each representing a different era of chess AI development:

| Level | Engine Name | Algorithm | Evaluation Method | Strength |
|:-----:|:------------|:----------|:------------------|:---------|
| **1** | Novice | Minimax (Depth 3) | Naive Material Sum | Beginner |
| **2** | Club | Minimax + MVV-LVA | Piece-Square Tables (PST) | Intermediate |
| **3** | Expert | Negamax + Iterative Deepening | PeSTO Tapered Eval + TT | Strong Amateur |
| **4** | AlphaLite | MCTS + ResNet | Neural Value & Policy Head | Master |

### Performance Highlights

- **Level 2 vs Level 1**: 100% Win Rate (50-0)
- **Level 3 vs Level 2**: 90% Win Rate
- **AlphaLite (10x128)**: Outperforms traditional heuristics and competes effectively with Stockfish Level 7

---

## 📂 Project Structure

```
Project Root
├── alphalite/              # AlphaLite Core Package
│   ├── AlphaLiteNetwork.py # ResNet Architecture (Policy & Value Heads)
│   ├── MCTS.py             # Monte Carlo Tree Search Implementation
│   └── encoder.py          # Board State Encoding (16x8x8 Tensor)
├── weights/                # Pre-trained Model Weights
│   ├── 10x128.pt           # Lightweight Model (ResNet-10)
│   └── 20x256.pt           # High-Capacity Model (ResNet-20)
├── engine_level1.py        # Classical Engine (Material)
├── engine_level2.py        # Classical Engine (PST + Quiescence)
├── engine_level3.py        # Classical Engine (PeSTO + Transposition Table)
├── engine_alphalite.py     # Neural Engine Interface
├── main_gui.py             # Application Entry Point (Royal Arena GUI)
└── requirements.txt        # Python Dependencies
```

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (Highly recommended for Level 4 inference)

### 1. Install Dependencies

```bash
pip install torch torchvision numpy python-chess flet
```

### 2. Setup Weights

Ensure your trained `.pt` model files are placed in the `weights/` directory. The engine expects:

- `weights/10x128.pt`
- `weights/20x256.pt`

### 3. Run the Application

Launch the Royal Arena graphical interface:

```bash
python main_gui.py
```

---

## 🎮 Usage Guide

1. **Select Difficulty Level**: Choose from Levels 1-4 based on your skill level
2. **Play**: Make moves by clicking on pieces and target squares
3. **Analyze**: View real-time evaluation scores and neural network confidence
4. **Learn**: Use move explainability features to understand engine decisions

---

## 📊 Technical Details

### Neural Architecture

- **Input Representation**: 16-channel × 8×8 board encoding
- **Backbone**: ResNet-10 or ResNet-20 with residual connections
- **Output Heads**:
  - Policy Head: 1968-dimensional move probability distribution
  - Value Head: Single scalar win/loss/draw prediction

### Training Data

- **Source**: CCRL database (Computer Chess Rating Lists)
- **Method**: Supervised Learning on expert games
- **Optimization**: Green AI approach—minimal computational resources


---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

[Your contact information here]

---

## 🙏 Acknowledgments

- AlphaZero paper by DeepMind
- CCRL for providing high-quality game databases
- Python-Chess library maintainers
- Flet framework developers
