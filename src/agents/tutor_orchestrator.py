"""Main tutor orchestrator with agentic loop."""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .tools import ToolExecutor, TOOL_DEFINITIONS, CHAT_TOOL_DEFINITIONS
from ..services.chess_state import ChessStateService
from ..utils.config import settings
from ..utils.prompts import SYSTEM_PROMPT, make_turn_prompt, make_chat_prompt
from ..utils.schemas import TutorPlan, MoveClassification


@dataclass
class ParsedInput:
    """Result of parsing user input."""
    is_move: bool
    move_uci: Optional[str] = None
    move_san: Optional[str] = None
    raw_input: str = ""


class TutorOrchestrator:
    """
    Main orchestrator for the chess tutor.

    Implements an agentic loop that:
    1. Receives user input
    2. Detects if input is a move or chat
    3. For moves: validates, applies, analyzes, tutor responds with move
    4. For chat: tutor responds conversationally (no board changes)

    When using vLLora gateway, sends x-run-id and x-thread-id headers for tracing.
    """

    MAX_TOOL_ITERATIONS = 10  # Safety limit for tool loop

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        user_level: str = "intermediate",
        stockfish_path: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        """
        Initialize the tutor orchestrator.

        Args:
            api_key: OpenAI API key (uses settings if None)
            model: Model to use (uses settings if None)
            user_level: Student's skill level
            stockfish_path: Path to Stockfish binary
            thread_id: Optional thread ID for vLLora tracing (auto-generated if None)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.user_level = user_level

        # Thread ID for vLLora tracing (persists across turns)
        self.thread_id = thread_id or str(uuid.uuid4())

        # Initialize OpenAI client with optional custom base URL for vLLora
        client_kwargs = {"api_key": self.api_key}
        base_url = settings.get_llm_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        # Initialize services
        self.chess_service = ChessStateService()
        self.tool_executor = ToolExecutor(
            stockfish_path=stockfish_path or settings.stockfish_path,
            stockfish_depth=settings.stockfish_depth,
            stockfish_multipv=settings.stockfish_multipv,
        )

        # Game state
        self.current_fen = ChessStateService.STARTING_FEN
        self.game_history: list[dict[str, Any]] = []
        self.conversation_history: list[dict[str, Any]] = []
        self.is_new_game = True

    def reset_game(self) -> None:
        """Reset to a new game."""
        self.current_fen = ChessStateService.STARTING_FEN
        self.game_history = []
        self.conversation_history = []
        self.is_new_game = True
        # Generate new thread ID for new game
        self.thread_id = str(uuid.uuid4())

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for each LLM call."""
        return str(uuid.uuid4())

    def _get_extra_headers(self, run_id: str) -> dict[str, str]:
        """Get extra headers for vLLora tracing."""
        return {
            "x-run-id": run_id,
            "x-thread-id": self.thread_id,
        }

    def _parse_user_input(self, user_input: str) -> ParsedInput:
        """
        Parse user input to determine if it's a valid move or chat.

        Tries UCI notation first (e2e4, g1f3), then SAN (e4, Nf3, O-O).
        If neither parses as a legal move, it's chat.

        Args:
            user_input: Raw user input string

        Returns:
            ParsedInput with is_move flag and parsed move details
        """
        clean_input = user_input.strip()

        # Try to apply as a move (this handles both UCI and SAN)
        result = self.chess_service.apply_move(self.current_fen, clean_input)

        if result.success:
            return ParsedInput(
                is_move=True,
                move_uci=result.uci,
                move_san=result.san,
                raw_input=clean_input,
            )

        # Not a valid move - it's chat
        return ParsedInput(
            is_move=False,
            raw_input=clean_input,
        )

    def process_turn(self, user_input: str) -> TutorPlan:
        """
        Process a single turn of interaction.

        Args:
            user_input: User's input (move or message)

        Returns:
            TutorPlan with tutor's response
        """
        # Generate a new run_id for this turn
        run_id = self._generate_run_id()

        # Parse input to determine if it's a move or chat
        parsed = self._parse_user_input(user_input)

        if parsed.is_move:
            return self._process_move_turn(parsed, run_id)
        else:
            return self._process_chat_turn(parsed, run_id)

    def _process_move_turn(self, parsed: ParsedInput, run_id: str) -> TutorPlan:
        """
        Process a turn where user made a valid move.

        1. Apply user's move to board
        2. Analyze position
        3. Get tutor response with opponent move
        """
        # Apply the user's move (we already validated it parses)
        result = self.chess_service.apply_move(self.current_fen, parsed.raw_input)

        if result.success and result.new_fen:
            self.current_fen = result.new_fen
            self.game_history.append({
                "san": result.san,
                "uci": result.uci,
                "role": "user",
            })

        # Build turn prompt for move mode
        turn_prompt = make_turn_prompt(
            user_input=parsed.raw_input,
            current_fen=self.current_fen,
            user_level=self.user_level,
            is_new_game=self.is_new_game,
            game_history=self.game_history,
            is_move=True,
            move_san=parsed.move_san,
        )

        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": turn_prompt,
        })

        # Run agentic loop with move tools
        tutor_plan = self._run_agentic_loop(run_id, is_move_mode=True)

        # Apply tutor's response move if valid
        if tutor_plan.opponent_reply:
            tutor_result = self.chess_service.apply_move(self.current_fen, tutor_plan.opponent_reply)
            if tutor_result.success and tutor_result.new_fen:
                self.current_fen = tutor_result.new_fen
                self.game_history.append({
                    "san": tutor_result.san,
                    "uci": tutor_result.uci,
                    "role": "tutor",
                })

        self.is_new_game = False
        return tutor_plan

    def _process_chat_turn(self, parsed: ParsedInput, run_id: str) -> TutorPlan:
        """
        Process a turn where user sent a chat message (not a move).

        DO NOT change the board state. Just respond conversationally.
        """
        # Build chat prompt
        chat_prompt = make_chat_prompt(
            user_input=parsed.raw_input,
            current_fen=self.current_fen,
            user_level=self.user_level,
            game_history=self.game_history,
        )

        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": chat_prompt,
        })

        # Run agentic loop with chat tools only (no move tools)
        tutor_plan = self._run_agentic_loop(run_id, is_move_mode=False)

        # IMPORTANT: In chat mode, ignore any opponent_reply the LLM might return
        tutor_plan.opponent_reply = None

        self.is_new_game = False
        return tutor_plan

    def _run_agentic_loop(self, run_id: str, is_move_mode: bool = True) -> TutorPlan:
        """
        Run the agentic loop until LLM produces final response.

        Args:
            run_id: The run ID for this turn
            is_move_mode: If True, use full tools. If False, use chat-only tools.

        Returns:
            Extracted TutorPlan from LLM response
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history,
        ]

        # Use appropriate tools based on mode
        tools = TOOL_DEFINITIONS if is_move_mode else CHAT_TOOL_DEFINITIONS

        # Use the same run_id for all LLM calls within this turn
        extra_headers = self._get_extra_headers(run_id)

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            # Call LLM with vLLora headers
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
                extra_headers=extra_headers,
            )

            assistant_message = response.choices[0].message

            # Check if we have tool calls to process
            if assistant_message.tool_calls:
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ],
                })

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Execute tool
                    result = self.tool_executor.execute(
                        tool_name=tool_name,
                        arguments=tool_args,
                    )

                    # In move mode, track tutor's move applications
                    # (User moves are already applied before the loop)
                    # We don't apply moves here - that's done after the loop

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, indent=2),
                    })

            else:
                # No tool calls - LLM is done, extract TutorPlan
                content = assistant_message.content or ""

                # Add final response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content,
                })

                return self._extract_tutor_plan(content)

        # Safety: max iterations reached
        return TutorPlan(
            explain="I apologize, but I encountered an issue processing this position. "
                    "Let me try again with a simpler approach.",
            grounding_citations=["error: max tool iterations reached"],
            move_evaluation=MoveClassification.GOOD,
        )

    def _extract_tutor_plan(self, content: str) -> TutorPlan:
        """
        Extract TutorPlan from LLM response.

        Tries to find JSON in the response, falls back to creating
        a basic plan from the text.
        """
        # Try to find JSON in response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            try:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                return TutorPlan(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: create basic plan from text
        return TutorPlan(
            explain=content,
            grounding_citations=["fallback: could not parse structured response"],
            move_evaluation=MoveClassification.GOOD,
        )

    def get_board_display(self) -> str:
        """Get ASCII representation of current board."""
        return self.chess_service.board_ascii(self.current_fen)

    def get_game_status(self) -> dict[str, Any]:
        """Get current game status."""
        status = self.chess_service.game_status(self.current_fen)
        return {
            "status": status.status.value,
            "turn": status.turn,
            "is_game_over": status.is_game_over,
            "winner": status.winner,
            "move_number": status.fullmove_number,
        }

    def close(self) -> None:
        """Clean up resources."""
        self.tool_executor.close()
