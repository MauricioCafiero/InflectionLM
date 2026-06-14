# InflectionLM: Beam Token Visualization

InflectionLM is a tool designed to visualize "inflection points" in Large Language Model (LLM) generation. By leveraging beam search, the tool generates multiple alternative output paths (beams) and identifies tokens where the model's confidence was low, highlighting these as potential points where the generation could have diverged.

## 🌟 Features

- **Multi-Beam Generation**: Generates 5 alternative beams for every prompt using the GEMMA 4 model.
- **Confidence Highlighting**: Automatically highlights tokens with a probability score below **0.6** in red, marking them as "inflection points."
- **Interactive Visualization**: 
  - A Gradio-based GUI to input prompts and view results.
  - Ability to switch between the 5 generated beams.
  - Toggleable detailed view showing exact probability scores for every token in a beam.
- **Customizable Generation**: Adjustable temperature settings to control the randomness and diversity of the output.
- **UI Preferences**: Support for both light and dark modes.

## 🛠️ Tech Stack

- **Language**: Python
- **LLM**: [GEMMA 4](https://huggingface.co/google/gemma-4-31B-it) (via Hugging Face `transformers`)
- **Deep Learning Framework**: PyTorch
- **UI Framework**: Gradio
- **Compute**: Optimized for GPU usage (via `@spaces.GPU` for ZeroGPU environments)

## 📁 Project Structure

```text
InflectionLM/
├── app.py               # Gradio web application and UI logic
├── inflections_funcs.py # Core logic for model loading, beam generation, and scoring
├── requirements.txt     # Project dependencies
└── Stochastic_parrot.JPG # UI asset image
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A GPU with sufficient VRAM to load GEMMA 4 (31B)
- Hugging Face account and access to the GEMMA 4 model

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd InflectionLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Ensure your environment variables for Hugging Face are set (e.g., `HUGGING_FACE_HUB_TOKEN`), then run:

```bash
python app.py
```

The application will start a Gradio server. Open the provided URL in your browser to begin experimenting.

## 📖 How it Works

The tool uses **Beam Search** to maintain several top-scoring sequences simultaneously. After generation, it passes these sequences back through the model to extract the precise softmax probability of each chosen token. 

Tokens with a probability $< 0.6$ are considered "unconfident." In linguistic terms, these are the "inflections"—moments where the model's internal probability distribution was flatter, meaning alternative tokens were almost as likely, leading to potential variations in the final text.
