#!/usr/bin/env python3
"""Interactive CLI demo for the Chess Tutor."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.agents.tutor_orchestrator import TutorOrchestrator
from src.utils.config import settings


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("  ♔ INTERACTIVE CHESS TUTOR ♚")
    print("=" * 60)
    print("\nCommands:")
    print("  - Enter a move in SAN (e.g., 'e4', 'Nf3', 'O-O') or UCI (e.g., 'e2e4')")
    print("  - Type 'board' to see the current position")
    print("  - Type 'status' to see game status")
    print("  - Type 'new' to start a new game")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type anything else to chat with the tutor")
    print("=" * 60 + "\n")


def print_board(orchestrator: TutorOrchestrator):
    """Print the current board."""
    print("\n" + orchestrator.get_board_display())
    print(f"FEN: {orchestrator.current_fen}\n")


def print_status(orchestrator: TutorOrchestrator):
    """Print game status."""
    status = orchestrator.get_game_status()
    print(f"\nGame Status: {status['status']}")
    print(f"Turn: {status['turn'].capitalize()}")
    print(f"Move Number: {status['move_number']}")
    if status['is_game_over']:
        winner = status.get('winner')
        if winner:
            print(f"Winner: {winner.capitalize()}")
        else:
            print("Result: Draw")
    print()


def format_tutor_response(plan, is_move_mode: bool) -> str:
    """Format tutor plan for display."""
    lines = []

    # Move evaluation - only show in move mode
    if is_move_mode and plan.move_evaluation:
        eval_emoji = {
            "brilliant": "💎",
            "great": "⭐",
            "best": "✅",
            "excellent": "👍",
            "good": "👌",
            "inaccuracy": "⚠️",
            "mistake": "❌",
            "blunder": "💀",
            "book": "📖",
        }
        emoji = eval_emoji.get(plan.move_evaluation.value, "")
        lines.append(f"Move Quality: {emoji} {plan.move_evaluation.value.upper()}")

    # Explanation
    lines.append(f"\n{plan.explain}")

    # Tutor's move - only in move mode
    if plan.opponent_reply:
        lines.append(f"\n🎯 My move: {plan.opponent_reply}")

    # Suggested moves
    if plan.user_candidates:
        lines.append(f"\n💡 Candidates to consider: {', '.join(plan.user_candidates)}")

    # Question
    if plan.ask_user:
        lines.append(f"\n❓ {plan.ask_user}")

    # Grounding (debug info)
    if plan.grounding_citations:
        citation = plan.grounding_citations[0]
        lines.append(f"\n📊 Analysis: {citation[:80]}...")

    return "\n".join(lines)


def select_level() -> str:
    """Let user select their level."""
    print("\nWhat's your chess level?")
    print("  1. Beginner (learning the basics)")
    print("  2. Intermediate (know tactics, learning strategy)")
    print("  3. Advanced (rated 1800+)")

    while True:
        choice = input("\nEnter 1, 2, or 3 [2]: ").strip() or "2"
        if choice == "1":
            return "beginner"
        elif choice == "2":
            return "intermediate"
        elif choice == "3":
            return "advanced"
        print("Please enter 1, 2, or 3")


def main():
    """Run the interactive chess tutor demo."""
    # Load environment variables
    load_dotenv()

    # Check API key
    if not settings.validate_api_key():
        print("\n⚠️  ERROR: OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file or environment.")
        print("Copy .env.example to .env and add your key.")
        sys.exit(1)

    print_header()

    # Show vLLora status
    if settings.use_local_gateway:
        print(f"🔗 Using vLLora gateway: {settings.llm_base_url}")
    else:
        print("🔗 Using OpenAI API directly")

    # Select level
    user_level = select_level()
    print(f"\nGreat! Starting as {user_level} player.\n")

    # Initialize orchestrator
    try:
        orchestrator = TutorOrchestrator(user_level=user_level)
        print(f"Thread ID: {orchestrator.thread_id}\n")
    except RuntimeError as e:
        print(f"\n⚠️  ERROR: {e}")
        print("Make sure Stockfish is installed:")
        print("  - Linux: sudo apt install stockfish")
        print("  - macOS: brew install stockfish")
        print("  - Or set STOCKFISH_PATH in .env")
        sys.exit(1)

    # Initial greeting
    print("Tutor: Welcome! I'm your chess tutor. You'll play as White.")
    print("       Make your first move (e.g., 'e4' or 'd4') or ask a question.\n")

    print_board(orchestrator)

    # Main loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            lower_input = user_input.lower()

            if lower_input in ("quit", "exit", "q"):
                print("\nThanks for playing! Goodbye. ♔")
                break

            if lower_input == "board":
                print_board(orchestrator)
                continue

            if lower_input == "status":
                print_status(orchestrator)
                continue

            if lower_input == "new":
                orchestrator.reset_game()
                print("\n--- New Game Started ---")
                print(f"New Thread ID: {orchestrator.thread_id}\n")
                print_board(orchestrator)
                continue

            # Process turn
            print("\nTutor: (thinking...)")
            try:
                # Store FEN before to detect if move was made
                fen_before = orchestrator.current_fen

                plan = orchestrator.process_turn(user_input)

                # Detect mode: if FEN changed or opponent replied, it was move mode
                is_move_mode = (orchestrator.current_fen != fen_before) or (plan.opponent_reply is not None)

                if is_move_mode:
                    print("[MOVE MODE]")
                else:
                    print("[CHAT MODE]")

                response = format_tutor_response(plan, is_move_mode)
                print(f"\nTutor: {response}\n")

                # Show board after moves (only in move mode)
                if is_move_mode:
                    print_board(orchestrator)

                # Check for game over
                status = orchestrator.get_game_status()
                if status['is_game_over']:
                    print("\n🏁 GAME OVER!")
                    print_status(orchestrator)
                    print("Type 'new' to start a new game or 'quit' to exit.\n")

            except Exception as e:
                print(f"\nTutor: I encountered an error: {e}")
                print("       Please try again or type 'new' to restart.\n")

    finally:
        # Clean up
        orchestrator.close()


if __name__ == "__main__":
    main()
