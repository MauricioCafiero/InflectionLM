# InflectionLM: Output and Token Visualization

<img src="Stochastic_parrot.JPG" width="25%">

InflectionLM is a tool designed to visualize "inflection points" in Large Language Model (LLM) generation. By leveraging independent sampling, the tool generates multiple alternative output paths and identifies tokens where the model's confidence was low, highlighting these as potential points where the generation could have diverged.

[See the Demo on HuggingFace!](https://huggingface.co/spaces/build-small-hackathon/InflectionLM)

or

[Watch a video demo!](https://www.linkedin.com/posts/mauricio-cafiero-5481259b_buildsmallhackathon-gemma4-huggingface-ugcPost-7472245848584683521-L4hx)

## 🌟 Features

- **Diverse Response Generation**: Generates 3 independent responses for every prompt using the GEMMA 4 model, utilizing Top-P (Nucleus) and Top-K sampling to maximize diversity.
- **Confidence Highlighting**: Automatically highlights tokens with a probability score below **0.6** in red, marking them as "inflection points."
- **Interactive Visualization**: 
  - A Gradio-based GUI to input prompts and view results.
  - Ability to switch between the 3 generated responses.
  - Toggleable detailed view showing exact probability scores for every token in a response.
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
├── inflections_funcs.py # Core logic for model loading, generation, and scoring
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

The tool uses **Multinomial Sampling** with Top-P and Top-K filtering to generate distinct sequences. Instead of a deterministic beam search, it samples from the probability distribution to find diverse but high-quality responses. After generation, it extracts the softmax probability of each chosen token from the model's output scores.

Tokens with a probability $< 0.6$ are considered "unconfident." In linguistic terms, these are the "inflections"—moments where the model's internal probability distribution was flatter, meaning alternative tokens were almost as likely, leading to potential variations in the final text.
