def rule_based_qa(question, knowledge_base):
    """
    This function implements a rule-based question-answering system.
    It uses a simple keyword matching approach to find the answer in the knowledge base.
    
    Args:
    question (str): The input question from the user.
    knowledge_base (dict): A dictionary containing questions as keys and answers as values.
    
    Returns:
    str: The answer to the question if found, otherwise a default message.
    """
    
    # Normalize the question by converting it to lowercase
    normalized_question = question.lower()
    
    # Check if the normalized question is in the knowledge base
    if normalized_question in knowledge_base:
        return knowledge_base[normalized_question]
    
    # If not found, return a default message
    return "I'm sorry, I don't have an answer for that question."