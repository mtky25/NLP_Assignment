from .client import MillionaireClient

__all__ = ["MillionaireClient"]

def quick_start():
    """example usage of the library."""
    import os
    
    api_url = "http://131.175.15.22:51111/"
    username = os.getenv("POLI_USERNAME", "")
    password = os.getenv("POLI_PASSWORD", "")
    
    client = MillionaireClient(api_url)
    client.login(username, password)
    
    # start a game
    game = client.game.start(competition_id=1)
    
    # answer questions
    while game.in_progress:
        question = game.current_question
        print(f"q: {question.text}")
        for opt in question.options:
            print(f"  {opt.id}: {opt.text}")
            
        # your answering logic here
        answer_id = question.options[0].id
        result = game.answer(answer_id)
        print(f"correct: {result.correct}")
