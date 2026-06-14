from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch import exp

class inflections():
  def __init__(self, model_id = "google/gemma-4-31B-it"):
    '''
    '''
    self.model_id = model_id

  def start_model(self):
    '''
    '''
    self.processor = AutoProcessor.from_pretrained(self.model_id)
    self.model = AutoModelForCausalLM.from_pretrained(
    self.model_id,
    dtype="auto",
    device_map="auto"
    )
    print(f'Model {self.model_id} has been installed.')
  
  def make_beams(self, intitial_prompt: str):
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": initial_prompt},
    ]

    # Process input
    text = self.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    self.generated_dicts = self.model.generate(**inputs,
                            max_new_tokens=1024,
                            num_beams=5,          # Enables beam search with 5 active beams
                            num_return_sequences=5, # Returns the top result
                            return_dict_in_generate=True,
                            output_scores=True)
    
    self.transcription = self.processor.batch_decode(self.generated_dicts.sequences, skip_special_tokens=True)

    print('Keys in model output -------------------')
    for key in self.generated_dicts:
      print(key)
    print('----------------------------------------')

    print('Beam scores ----------------------------')
    for score in self.generated_dicts.sequences_scores:
      print(exp(score).item())
    print('----------------------------------------')

    return self.generated_dicts
  
  def parse_beams(self):
    '''
    '''
    beam_text = []
    for beam in self.transcription:
    
      parts = beam.split('''model\nthought''')
      response = parts[1].strip('\n')
      beam_text.append(response)

      print('Beams have been parsed. --------------')

    self.beam_text = beam_text
    return beam_text

  def get_beam_tokens(self):
    '''
    Decodes the generated sequences into individual tokens for each beam.
    '''
    beam_tokens = []
    # The number of generated tokens is the length of the scores list
    gen_len = len(self.generated_dicts.scores)

    for sequence in self.generated_dicts.sequences:
        # Calculate input length for this specific sequence
        total_len = sequence.shape[0]
        input_len = total_len - gen_len

        # Extract only the generated token IDs
        generated_ids = sequence[input_len:].tolist()

        # Convert IDs to tokens (e.g., ' Hello', ' world')
        tokens = self.processor.tokenizer.convert_ids_to_tokens(generated_ids)
        beam_tokens.append(tokens)

    self.beam_tokens = beam_tokens
    return beam_tokens

  def calculate_score_vectors(self):
    '''
    Creates a score vector for each beam containing the probability of each
    token that was chosen.
    '''
    # Number of sequences generated
    num_sequences = self.generated_dicts.sequences.shape[0]
    # Number of generated tokens (excluding prompt)
    gen_len = len(self.generated_dicts.scores)
    # Total length of sequences (including prompt)
    total_len = self.generated_dicts.sequences.shape[1]
    # Input length (prompt length)
    input_len = total_len - gen_len

    # To get the accurate probabilities for the final beams,
    # we pass the final sequences back through the model.
    with torch.no_grad():
        outputs = self.model(self.generated_dicts.sequences)
        logits = outputs.logits # shape: (num_sequences, total_len, vocab_size)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1) # shape: (num_sequences, total_len, vocab_size)

        score_vectors = []
        for i in range(num_sequences):
            beam_probs = []
            # The generated tokens are at positions [input_len, total_len)
            # The probability for the token at position t is found at logits index t-1
            for t in range(input_len, total_len):
                token_id = self.generated_dicts.sequences[i, t]
                # Probability for token at position t is at index t-1 in the logits
                prob = probs[i, t - 1, token_id].item()
                beam_probs.append(prob)
            score_vectors.append(beam_probs)

    self.score_vectors = score_vectors
    return score_vectors


