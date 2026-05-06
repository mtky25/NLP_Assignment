import os
from dotenv import load_dotenv
from millionaire_client import AuthenticationError, MillionaireClient
from src.guesser.guesser import Guesser
from src.prompts.guesser_instruction import guesser_instructions
from src.game.game import Game

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join("Marcelo", ".env"))

# GLOBALS from environment (using more specific names to avoid system conflicts)
API_URL =  "http://131.175.15.22:51111/"
MODEL_ID = "llama3.2"
USERNAME = os.getenv("MILLIONAIRE_USERNAME")
PASSWORD = os.getenv("MILLIONAIRE_PASSWORD")

def run_games(client: MillionaireClient, guesser: Guesser, n: int, competition_id=1):
    for i in range(n):
        print(f"\n=== Starting Game {i+1}/{n} ===")
        game = Game(client, guesser)
        game.init_game(competition_id)
        game.play_game()

def main():
    # Verify environment variables
    if not all([API_URL, USERNAME, PASSWORD]):
        print("Error: Missing required environment variables in .env file.")
        print(f"Make sure MILLIONAIRE_USERNAME and MILLIONAIRE_PASSWORD are set.")
        return

    # Model config
    guesser = Guesser(
        guesser_instructions,
        MODEL_ID
    )

    # Client config
    client = MillionaireClient(API_URL)
    try:
        user = client.login(USERNAME, PASSWORD)
        print(f"\nWelcome, {user.username}! (Role: {user.role})")
    except AuthenticationError as e:
        print(f"Login failed: {e}")
        return

    # List available competitions
    print("\n=== Available Competitions ===")
    competitions = client.competitions.list_all()
    for comp in competitions:
        print(f"  {comp.id}: {comp.name} ({comp.max_levels} questions)")
    
    N_GAMES = 10
    COMP_ID = 1
    run_games(client, guesser, N_GAMES, COMP_ID)

if __name__ == "__main__":
    main()
