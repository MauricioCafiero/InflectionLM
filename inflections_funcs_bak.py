from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch import exp
from typing import Any, Tuple, List

def start_model(model_id: str = "google/gemma-4-31B-it"):
    '''
    Initializes and returns the processor and model.
    '''
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto"
    )
    print(f'Model {model_id} has been installed.')
    return model, processor

def make_beams(model: AutoModelForCausalLM, processor: AutoProcessor, initial_prompt: str, temperature: float = 1.0) -> Tuple[Any, List[str]]:
    '''
    Generates 5 beams in response to a prompt.
    '''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    # Generate output
    generated_dicts = model.generate(**inputs,
                                    max_new_tokens=1024,
                                    num_beams=5,          # Enables beam search with 5 active beams
                                    num_return_sequences=5, # Returns the top result
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    temperature=temperature,
                                    do_sample=True if temperature > 0 else False)

    transcription = processor.batch_decode(generated_dicts.sequences, skip_special_tokens=True)

    print('Keys in model output -------------------')
    for key in generated_dicts:
        print(key)
    print('----------------------------------------')

    print('Beam scores ----------------------------')
    for score in generated_dicts.sequences_scores:
        print(exp(score).item())
    print('----------------------------------------')

    return generated_dicts, transcription

def parse_beams(transcription: List[str]) -> List[str]:
    '''
    Parses beams to extract only the response after 'model\nthought'.
    '''
    beam_text = []
    for beam in transcription:
        parts = beam.split('''model\nthought''')
        if len(parts) > 1:
            response = parts[1].strip('\n')
        else:
            response = beam
        beam_text.append(response)

    print('Beams have been parsed. --------------')
    return beam_text

def get_beam_tokens(generated_dicts: Any, processor: AutoProcessor) -> List[List[str]]:
    '''
    Decodes the generated sequences into individual tokens for each beam.
    '''
    beam_tokens = []
    # The number of generated tokens is the length of the scores list
    gen_len = len(generated_dicts.scores)

    for sequence in generated_dicts.sequences:
        # Calculate input length for this specific sequence
        total_len = sequence.shape[0]
        input_len = total_len - gen_len

        # Extract only the generated token IDs
        generated_ids = sequence[input_len:].tolist()

        # Convert IDs to tokens (e.g., ' Hello', ' world')
        tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids)
        beam_tokens.append(tokens)

    return beam_tokens

def calculate_score_vectors(model: AutoModelForCausalLM, generated_dicts: Any) -> List[List[float]]:
    '''
    Creates a score vector for each beam containing the probability of each
    token that was chosen.
    '''
    # Number of sequences generated
    num_sequences = generated_dicts.sequences.shape[0]
    # Number of generated tokens (excluding prompt)
    gen_len = len(generated_dicts.scores)
    # Total length of sequences (including prompt)
    total_len = generated_dicts.sequences.shape[1]
    # Input length (prompt length)
    input_len = total_len - gen_len

    # To get the accurate probabilities for the final beams,
    # we pass the final sequences back through the model.
    with torch.no_grad():
        outputs = model(generated_dicts.sequences)
        logits = outputs.logits # shape: (num_sequences, total_len, vocab_size)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1) # shape: (num_sequences, total_len, vocab_size)

        score_vectors = []
        for i in range(num_sequences):
            beam_probs = []
            # The generated tokens are at positions [input_len, total_len)
            # The probability for the token at position t is found at logits index t-1
            for t in range(input_len, total_len):
                token_id = generated_dicts.sequences[i, t]
                # Probability for token at position t is at index t-1 in the logits
                prob = probs[i, t - 1, token_id].item()
                beam_probs.append(prob)
            score_vectors.append(beam_probs)

    return score_vectors
