from transformer import AutoModelForQuestionAnswering, AutoTokenizer
import torch

def bert_qa(question, context):
    """
    This function uses a pre-trained BERT model for question answering.
    
    Args:
    question (str): The input question from the user.
    context (str): The context in which to find the answer.
    
    Returns:
    str: The answer to the question if found, otherwise a default message.
    """
    
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors='pt')
    
    # Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the start and end scores
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Get the most likely beginning of answer with the argmax of the score
    start_index = torch.argmax(start_scores)
    
    # Get the most likely end of answer with the argmax of the score
    end_index = torch.argmax(end_scores) + 1  # +1 because end index is inclusive
    
    # Convert token indices to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index])
    
    # Join tokens to form the answer string
    answer = tokenizer.convert_tokens_to_string(tokens)
    
    return answer if answer else "I'm sorry, I don't have an answer for that question."
if __name__ == "__main__":
    # Example usage
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    
    answer = bert_qa(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")