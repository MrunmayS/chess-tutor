#!/usr/bin/env python3
"""
Simulated conversations for generating Chess Tutor tracing data.

This script runs predefined conversation scenarios to generate x-thread-id and
x-run-id tracing data via vLLora gateway for RL fine-tuning.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.agents.tutor_orchestrator import TutorOrchestrator
from src.utils.config import settings


@dataclass
class ConversationScenario:
    """A predefined conversation scenario with moves and chat."""
    name: str
    skill_level: str
    user_inputs: list[str]
    description: str


def create_scenarios() -> list[ConversationScenario]:
    """Create diverse conversation scenarios with varied openings and interactions."""

    scenarios = [
        # Scenario 1: King's Pawn Opening - Beginner
        ConversationScenario(
            name="King's Pawn Opening - Beginner",
            skill_level="beginner",
            description="e4 opening with basic tactical questions",
            user_inputs=[
                "e4",
                "Nf3",
                "Why is developing knights early important?",
                "d4",
                "Bd3",
                "What should I be thinking about now?",
                "O-O",
                "c3",
                "How do I create an attack?",
                "Re1",
            ],
        ),

        # Scenario 2: Queen's Pawn - Intermediate
        ConversationScenario(
            name="Queen's Pawn Defense - Intermediate",
            skill_level="intermediate",
            description="d4 opening with positional questions",
            user_inputs=[
                "d4",
                "c4",
                "What's the main idea behind this Queen's Gambit approach?",
                "Nc3",
                "Nf3",
                "How should I evaluate the pawn structure here?",
                "e3",
                "Bd3",
                "What are my key strategic goals in this position?",
            ],
        ),

        # Scenario 3: Reti Opening - Advanced
        ConversationScenario(
            name="Reti Opening - Advanced",
            skill_level="advanced",
            description="Nf3 hypermodern opening with deep strategic analysis",
            user_inputs=[
                "Nf3",
                "g3",
                "Can you analyze the imbalances in this hypermodern setup?",
                "Bg2",
                "O-O",
                "What are the critical pawn breaks I should be considering?",
                "d3",
                "c4",
                "How does this compare to traditional d4 systems?",
            ],
        ),

        # Scenario 4: English Opening - Intermediate
        ConversationScenario(
            name="English Opening - Intermediate",
            skill_level="intermediate",
            description="c4 flank opening with strategy questions",
            user_inputs=[
                "c4",
                "Nc3",
                "What's the strategic difference between c4 and e4?",
                "g3",
                "Bg2",
                "What should I know about this fianchetto structure?",
                "Nf3",
                "d3",
                "How flexible is my position right now?",
            ],
        ),

        # Scenario 5: King's Gambit - Beginner
        ConversationScenario(
            name="King's Gambit - Beginner",
            skill_level="beginner",
            description="f4 gambit with tactical focus",
            user_inputs=[
                "e4",
                "f4",
                "Is this pawn sacrifice safe?",
                "Nf3",
                "Bc4",
                "What am I trying to achieve with this opening?",
                "d4",
                "O-O",
                "Should I be worried about my king?",
            ],
        ),

        # Scenario 6: Mixed Strategy - Advanced
        ConversationScenario(
            name="Positional e4 Game - Advanced",
            skill_level="advanced",
            description="Strategic e4 game with complex questions",
            user_inputs=[
                "e4",
                "d3",
                "Why choose d3 over d4 here?",
                "Nd2",
                "Ngf3",
                "Evaluate the knight positioning and pawn tension.",
                "g3",
                "Bg2",
                "What are the key squares to control in this structure?",
                "O-O",
            ],
        ),

        # Scenario 7: Quick Development - Intermediate
        ConversationScenario(
            name="Italian Game Development - Intermediate",
            skill_level="intermediate",
            description="Classical development with tactical opportunities",
            user_inputs=[
                "e4",
                "Nf3",
                "What's the plan after developing the knight?",
                "Bc4",
                "Nc3",
                "How do I decide between aggressive and quiet play?",
                "d3",
                "Be3",
                "What tactical patterns should I watch for?",
            ],
        ),

        # Scenario 8: London System - Beginner
        ConversationScenario(
            name="London System - Beginner",
            skill_level="beginner",
            description="Solid d4 system with fundamental questions",
            user_inputs=[
                "d4",
                "Nf3",
                "What makes the London System good for beginners?",
                "Bf4",
                "e3",
                "How should I continue developing?",
                "Nbd2",
                "Bd3",
                "What's my typical plan in the middlegame?",
            ],
        ),

        # Scenario 9: Chat-Only - Intermediate
        ConversationScenario(
            name="Chess Concepts Discussion - Intermediate",
            skill_level="intermediate",
            description="Pure chat scenario without moves",
            user_inputs=[
                "Can you explain the concept of weak squares?",
                "How do I know when to trade pieces?",
                "What's the difference between tactics and strategy?",
                "When should I start thinking about the endgame?",
                "How important is piece activity compared to material?",
                "Can you explain pawn chains and how to attack them?",
            ],
        ),

        # Scenario 10: Catalan-style - Advanced
        ConversationScenario(
            name="Catalan Setup - Advanced",
            skill_level="advanced",
            description="Sophisticated d4 + g3 system with deep analysis",
            user_inputs=[
                "d4",
                "c4",
                "How does the Catalan differ from the Queen's Gambit?",
                "g3",
                "Bg2",
                "Analyze the long diagonal and central tension.",
                "Nf3",
                "O-O",
                "What are the typical pawn breaks and piece maneuvers?",
            ],
        ),
    ]

    return scenarios


def run_scenario(scenario: ConversationScenario, verbose: bool = True) -> dict:
    """
    Run a single conversation scenario.

    Args:
        scenario: The scenario to run
        verbose: If True, print progress information

    Returns:
        Dictionary with statistics about the run
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Running: {scenario.name}")
        print(f"Skill Level: {scenario.skill_level}")
        print(f"Inputs: {len(scenario.user_inputs)} turns")
        print(f"{'=' * 70}")

    stats = {
        "scenario": scenario.name,
        "skill_level": scenario.skill_level,
        "total_turns": 0,
        "move_turns": 0,
        "chat_turns": 0,
        "thread_id": None,
        "success": False,
        "error": None,
    }

    try:
        # Initialize orchestrator with scenario's skill level
        orchestrator = TutorOrchestrator(user_level=scenario.skill_level)
        stats["thread_id"] = orchestrator.thread_id

        if verbose:
            print(f"Thread ID: {orchestrator.thread_id}\n")

        # Process each user input in the scenario
        for i, user_input in enumerate(scenario.user_inputs, 1):
            try:
                # Track FEN before to detect move vs chat mode
                fen_before = orchestrator.current_fen

                # Process the turn
                plan = orchestrator.process_turn(user_input)
                stats["total_turns"] += 1

                # Detect if it was a move or chat turn
                is_move = (orchestrator.current_fen != fen_before) or (plan.opponent_reply is not None)

                if is_move:
                    stats["move_turns"] += 1
                    mode = "MOVE"
                else:
                    stats["chat_turns"] += 1
                    mode = "CHAT"

                if verbose:
                    # Show brief summary
                    input_preview = user_input[:40] + "..." if len(user_input) > 40 else user_input
                    print(f"  [{i}/{len(scenario.user_inputs)}] [{mode}] {input_preview}")

                    # Show tutor's move if applicable
                    if plan.opponent_reply:
                        print(f"      → Tutor played: {plan.opponent_reply}")

            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Error on turn {i}: {e}")
                stats["error"] = f"Turn {i}: {str(e)}"
                break

        # Clean up
        orchestrator.close()

        # Mark as successful if we completed all turns
        if stats["total_turns"] == len(scenario.user_inputs):
            stats["success"] = True

        if verbose:
            print(f"\n✓ Completed: {stats['total_turns']} turns ({stats['move_turns']} moves, {stats['chat_turns']} chat)")
            if not stats["success"]:
                print(f"  ⚠️  {stats['error']}")

    except Exception as e:
        stats["error"] = f"Initialization error: {str(e)}"
        if verbose:
            print(f"\n⚠️  Failed to initialize: {e}")

    return stats


def list_scenarios(scenarios: list[ConversationScenario]) -> None:
    """List all available scenarios."""
    print("\n" + "=" * 70)
    print("AVAILABLE SCENARIOS")
    print("=" * 70)

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i}] {scenario.name}")
        print(f"    Skill Level: {scenario.skill_level}")
        print(f"    Turns: {len(scenario.user_inputs)}")
        print(f"    Description: {scenario.description}")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(scenarios)} scenarios")
    print("=" * 70 + "\n")


def main():
    """CLI entry point for conversation simulator."""
    parser = argparse.ArgumentParser(
        description="Run simulated chess tutor conversations for tracing data generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all scenarios once
  %(prog)s --count 5          # Run 5 random scenarios
  %(prog)s --scenario 0 2 4   # Run specific scenarios by index
  %(prog)s --repeat 3         # Run all scenarios 3 times each
  %(prog)s --list             # List all available scenarios
        """
    )

    parser.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="Run N random scenarios (instead of all)",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="Run specific scenario(s) by index (see --list)",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Repeat each scenario N times (default: 1)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios and exit",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Create scenarios
    scenarios = create_scenarios()

    # Handle --list
    if args.list:
        list_scenarios(scenarios)
        return

    # Validate API configuration
    if not settings.validate_api_key():
        print("\n⚠️  ERROR: OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file or environment.")
        print("Copy .env.example to .env and add your key.")
        sys.exit(1)

    # Show configuration
    if not args.quiet:
        print("\n" + "=" * 70)
        print("CHESS TUTOR CONVERSATION SIMULATOR")
        print("=" * 70)

        if settings.use_local_gateway:
            print(f"🔗 Using vLLora gateway: {settings.llm_base_url}")
        else:
            print("🔗 Using OpenAI API directly")

        print(f"📊 Total scenarios available: {len(scenarios)}")

    # Determine which scenarios to run
    scenarios_to_run = []

    if args.scenario is not None:
        # Run specific scenarios
        for idx in args.scenario:
            if 0 <= idx < len(scenarios):
                scenarios_to_run.append(scenarios[idx])
            else:
                print(f"⚠️  Warning: Scenario index {idx} out of range (0-{len(scenarios)-1}), skipping")
    elif args.count is not None:
        # Run N random scenarios
        count = min(args.count, len(scenarios))
        scenarios_to_run = random.sample(scenarios, count)
    else:
        # Run all scenarios
        scenarios_to_run = scenarios

    if not args.quiet:
        print(f"🎯 Running {len(scenarios_to_run)} scenario(s), {args.repeat} time(s) each")
        print("=" * 70)

    # Run scenarios
    all_stats = []
    total_runs = len(scenarios_to_run) * args.repeat
    current_run = 0

    for repeat_num in range(args.repeat):
        if args.repeat > 1 and not args.quiet:
            print(f"\n{'#' * 70}")
            print(f"# REPEAT {repeat_num + 1} / {args.repeat}")
            print(f"{'#' * 70}")

        for scenario in scenarios_to_run:
            current_run += 1

            if not args.quiet:
                print(f"\n[Run {current_run}/{total_runs}]")

            stats = run_scenario(scenario, verbose=not args.quiet)
            all_stats.append(stats)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = sum(1 for s in all_stats if s["success"])
    failed = len(all_stats) - successful
    total_turns = sum(s["total_turns"] for s in all_stats)
    total_moves = sum(s["move_turns"] for s in all_stats)
    total_chats = sum(s["chat_turns"] for s in all_stats)

    print(f"Total Conversations: {len(all_stats)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nTotal Turns: {total_turns}")
    print(f"  Move Turns: {total_moves}")
    print(f"  Chat Turns: {total_chats}")

    if failed > 0:
        print("\nFailed Conversations:")
        for s in all_stats:
            if not s["success"]:
                print(f"  - {s['scenario']}: {s['error']}")

    print("=" * 70 + "\n")

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
