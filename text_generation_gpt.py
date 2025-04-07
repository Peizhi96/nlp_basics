from transfomers import GPT2LMHeadModel, GPT2Tokenizer

def gpt_generate_text(prompt, model_name='gpt2', max_length=100):
    """
    Generate text using GPT-2 model.

    Args:
        prompt (str): The input text to generate from.
        model_name (str): The name of the pre-trained model.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    # num_beams: Number of beams for beam search (default: 1)
    # no_repeat_ngram_size: Size of n-grams to avoid repetition, ngram means the sequence of n words (default: 0)
    # early_stopping: Whether to stop the beam search when at least num_beams sentences are finished per batch or not (default: False)
    # max_length: Maximum length of the sequence to be generated
    

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text