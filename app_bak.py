import gradio as gr
import spaces
import math
from inflections_funcs import start_model, make_beams, get_beam_tokens, calculate_score_vectors

# Global model initialization
print("Loading model...")
model, processor = start_model()
print("Model loaded.")

def generate_beam_html(index, all_beams_data, dark_mode):
    """
    Helper to construct the main and detailed HTML for a specific beam index.
    """
    beam_tokens = all_beams_data["beam_tokens"][index]
    beam_scores = all_beams_data["score_vectors"][index]
    beam_overall_score = math.exp(all_beams_data["sequences_scores"][index])

    # Construct Main HTML output
    main_container_style = "border: 2px solid lightblue; padding: 15px; border-radius: 8px; font-family: sans-serif;"
    title_style = "font-weight: bold; margin-bottom: 10px; font-size: 1.1em;"

    main_html = f'<div style="{main_container_style}">'
    main_html += f'<div style="{title_style}">Beam {index + 1} | Score: {beam_overall_score:.4f}</div>'
    main_html += '<div style="font-family: monospace; white-space: pre-wrap; line-height: 1.5; overflow-wrap: break-word; word-wrap: break-word;">'

    normal_color = "white" if dark_mode else "black"

    for token, score in zip(beam_tokens, beam_scores):
        # Comprehensive replacement of SentencePiece space (U+2581),
        # literal underscores (U+005F), and non-breaking spaces (U+00A0).
        display_token = token.replace('▁', ' ').replace('_', ' ').replace(' ', ' ')
        if score < 0.6:
            color = "red"
            bg_color = "#441111" if dark_mode else "#ffe6e6"
            style = f"color: {color}; background-color: {bg_color};"
        else:
            color = normal_color
            style = f"color: {color};"
        main_html += f'<span style="{style}" title="Score: {score:.4f}">{display_token}</span>'
    main_html += '</div></div>'

    # Construct Detailed HTML output
    detail_table_style = "border-collapse: collapse; width: 100%; max-width: 500px; font-family: monospace;"
    th_style = "border: 1px solid #ccc; padding: 8px; text-align: left; background-color: #f2f2f2;" if not dark_mode else "border: 1px solid #444; padding: 8px; text-align: left; background-color: #333;"
    td_style = "border: 1px solid #ccc; padding: 8px; text-align: left;" if not dark_mode else "border: 1px solid #444; padding: 8px; text-align: left;"

    detail_html = f'<table style="{detail_table_style}"><thead><tr><th style="{th_style}">Token</th><th style="{th_style}">Score</th></tr></thead><tbody>'
    for token, score in zip(beam_tokens, beam_scores):
        display_token = token.replace(' ', ' ').replace('_', ' ').replace(' ', ' ')
        if score < 0.6:
            color = "red"
            bg_color = "#441111" if dark_mode else "#ffe6e6"
            token_cell_style = f"{td_style} color: {color}; background-color: {bg_color};"
        else:
            color = normal_color
            token_cell_style = f"{td_style} color: {color};"
        detail_html += f'<tr><td style="{token_cell_style}">{display_token}</td><td style="{td_style}">{score:.4f}</td></tr>'
    detail_html += '</tbody></table>'

    return main_html, detail_html

@spaces.GPU
def predict(prompt, dark_mode, temperature):
    """
    Generates responses for 5 beams and returns the first beam's visualization and visibility for controls.
    """
    # Generate beams
    generated_dicts, transcription = make_beams(model, processor, prompt, temperature=temperature)

    # Get tokens and scores for all beams
    beam_tokens = get_beam_tokens(generated_dicts, processor)
    score_vectors = calculate_score_vectors(model, generated_dicts)

    if not beam_tokens or not score_vectors:
        return "<span style='color: grey'>No tokens generated.</span>", "", gr.update(visible=False), gr.update(visible=False), None, 0

    # Initialize state with all beam data
    # Convert tensors to lists to avoid ZeroGPU serialization errors
    all_beams_data = {
        "beam_tokens": beam_tokens,
        "score_vectors": score_vectors,
        "sequences_scores": generated_dicts.sequences_scores.tolist() if hasattr(generated_dicts.sequences_scores, 'tolist') else generated_dicts.sequences_scores
    }

    # Generate HTML for the first beam (index 0)
    main_html, detail_html = generate_beam_html(0, all_beams_data, dark_mode)

    return main_html, detail_html, gr.update(visible=True), gr.update(visible=True), all_beams_data, 0

def switch_beam(current_index, all_beams_data, dark_mode):
    """
    Increments the beam index and returns the updated HTML.
    """
    if all_beams_data is None:
        return None, None, 0

    new_index = (current_index + 1) % 5
    main_html, detail_html = generate_beam_html(new_index, all_beams_data, dark_mode)
    return main_html, detail_html, new_index

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("Stochastic_parrot.JPG")
        with gr.Column(scale=3):
            gr.Markdown("# InflectionLM: Beam Token Visualization")
            gr.Markdown('''Input a prompt. The model will:
                            - Generate multiple beams (a beam is a possible output generated by the model).
                            - Display the first output beam, highlighting any word or parts of words (tokens) with a probability score < 0.6 in red.
                            - These unconfident words/tokens can be considered inflection points, or places where the output could change easily.
                            - Provide a toggle button to view detailed word/token probabilities for the beam.
                            - Switch between beams to see different possible outputs and their associated token probabilities.
                        ''')

    with gr.Column():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=1, show_label=True)
        with gr.Row():
            dark_mode_toggle = gr.Checkbox(label="Dark Mode", value=True)
            temp_radio = gr.Radio(
                label="Temperature",
                choices=[0.0, 0.5, 1.0, 1.5],
                value=1.0
            )
        submit_btn = gr.Button("Generate")

        output_html = gr.HTML(label="Highlighted Beam")

        with gr.Row():
            toggle_btn = gr.Button("Show Token Details", visible=False)
            next_beam_btn = gr.Button("Next Beam", visible=False)

        detail_html = gr.HTML(label="Detailed Token Probabilities", visible=False)

    # State components
    visibility_state = gr.State(value=False)
    all_beams_state = gr.State(value=None)
    current_beam_state = gr.State(value=0)

    def toggle_view(visible):
        return not visible, gr.update(visible=not visible)

    toggle_btn.click(fn=toggle_view, inputs=visibility_state, outputs=[visibility_state, detail_html])

    # Switch beam logic
    next_beam_btn.click(
        fn=switch_beam,
        inputs=[current_beam_state, all_beams_state, dark_mode_toggle],
        outputs=[output_html, detail_html, current_beam_state]
    )

    # Trigger generation on both button click and Enter key in textbox
    submit_btn.click(
        fn=predict,
        inputs=[prompt_input, dark_mode_toggle, temp_radio],
        outputs=[output_html, detail_html, toggle_btn, next_beam_btn, all_beams_state, current_beam_state]
    )
    prompt_input.submit(
        fn=predict,
        inputs=[prompt_input, dark_mode_toggle, temp_radio],
        outputs=[output_html, detail_html, toggle_btn, next_beam_btn, all_beams_state, current_beam_state]
    )

if __name__ == "__main__":
    demo.launch()
