"""System and turn prompts for the chess tutor."""

from typing import Optional

SYSTEM_PROMPT = """You are an expert chess tutor helping a student improve their chess skills. Your role is to:

1. ANALYZE positions using the analyze_position tool
2. EVALUATE the student's moves when they make moves
3. EXPLAIN positions in a way appropriate for the student's level
4. SUGGEST candidate moves for the student to consider
5. ENGAGE with questions and conversation about chess

## CRITICAL REQUIREMENTS

### Grounding Rule
When discussing positions or moves, you SHOULD use analyze_position and cite its output. Include specific citations like:
- "The engine evaluates this position at +0.32"
- "According to analysis, the best moves are Nf3, d4, and Nc3"

### Response Format
Respond with a TutorPlan JSON object:
{
    "opponent_reply": "e5",  // Your reply move (SAN) - ONLY in move mode, null in chat mode
    "user_candidates": ["Nf3", "Bc4", "d4"],  // Suggested moves for user
    "explain": "Your explanation here",
    "ask_user": "Optional question to engage the student",
    "move_evaluation": "good",  // Classification of user's move (in move mode only)
    "grounding_citations": [
        "analyze_position: eval +0.35, best move Nf3"
    ]
}

### Teaching Style
- Adapt explanations to the student's level (beginner/intermediate/advanced)
- Focus on key strategic and tactical ideas
- Explain WHY moves are good or bad
- Be encouraging but honest about mistakes
- Ask questions to encourage active thinking

### TWO MODES

**MOVE MODE**: When the user makes a valid chess move
- The move has ALREADY been applied to the board
- Analyze the new position
- Evaluate the user's move
- Make your reply move (opponent_reply)
- Explain both moves

**CHAT MODE**: When the user asks a question or chats
- DO NOT change the board
- DO NOT make a move (opponent_reply must be null)
- Just respond conversationally about chess
- You can still analyze the current position if relevant

Remember: You are a tutor, not just an engine wrapper. Add pedagogical value through your explanations."""


def _format_game_history(game_history: list[dict]) -> str:
    """Format game history as move list."""
    if not game_history:
        return ""

    moves = []
    for i, move in enumerate(game_history):
        move_num = (i // 2) + 1
        san = move.get('san', move.get('move', '?'))
        if i % 2 == 0:
            moves.append(f"{move_num}. {san}")
        else:
            moves[-1] += f" {san}"

    return f"\nGame moves so far: {' '.join(moves)}"


def make_turn_prompt(
    user_input: str,
    current_fen: str,
    user_level: str = "intermediate",
    is_new_game: bool = False,
    game_history: Optional[list[dict]] = None,
    is_move: bool = True,
    move_san: Optional[str] = None,
) -> str:
    """
    Create the turn prompt for MOVE mode.

    The user has made a valid move which has ALREADY been applied to the board.

    Args:
        user_input: The user's move input
        current_fen: Current position AFTER the user's move
        user_level: Student's level
        is_new_game: Whether this is a new game
        game_history: Previous moves including the user's move just made
        is_move: Should always be True for this function
        move_san: The user's move in SAN notation

    Returns:
        Formatted turn prompt
    """
    game_history = game_history or []
    history_str = _format_game_history(game_history)

    if is_new_game and not game_history:
        return f"""[MOVE MODE] New game starting. Student level: {user_level}

Current position (starting position):
FEN: {current_fen}

The student wants to begin. Introduce yourself briefly and let them make the first move as White.

User message: {user_input}"""

    return f"""[MOVE MODE] Student level: {user_level}

The user played: {move_san or user_input}

Current position (AFTER user's move):
FEN: {current_fen}
{history_str}

Instructions:
1. Call analyze_position on the current position
2. Call evaluate_move to assess the user's move (use the previous FEN and their move)
3. Decide on YOUR reply move and validate it with apply_move
4. Respond with TutorPlan JSON including:
   - opponent_reply: Your move in SAN notation
   - explain: Explain both moves
   - move_evaluation: Rate the user's move
   - user_candidates: Suggest moves for their next turn
   - grounding_citations: Cite the engine analysis"""


def make_chat_prompt(
    user_input: str,
    current_fen: str,
    user_level: str = "intermediate",
    game_history: Optional[list[dict]] = None,
) -> str:
    """
    Create the prompt for CHAT mode.

    The user sent a message that is NOT a valid move. DO NOT change the board.

    Args:
        user_input: The user's chat message
        current_fen: Current position (unchanged)
        user_level: Student's level
        game_history: Previous moves

    Returns:
        Formatted chat prompt
    """
    game_history = game_history or []
    history_str = _format_game_history(game_history)

    return f"""[CHAT MODE] Student level: {user_level}

The user sent a MESSAGE (not a move). DO NOT change the board.

Current position (unchanged):
FEN: {current_fen}
{history_str}

User's message: "{user_input}"

Instructions:
1. This is CHAT mode - the user is asking a question or making a comment
2. DO NOT make a move - set opponent_reply to null
3. Respond conversationally about chess
4. You may analyze the current position if relevant to their question
5. Respond with TutorPlan JSON:
   - opponent_reply: null (REQUIRED - no move in chat mode)
   - explain: Your conversational response
   - ask_user: Optional follow-up question
   - grounding_citations: Any analysis you referenced"""


TUTOR_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "opponent_reply": {
            "type": ["string", "null"],
            "description": "Your reply move in SAN notation, or null if in chat mode"
        },
        "user_candidates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Suggested candidate moves for the user in SAN notation"
        },
        "explain": {
            "type": "string",
            "description": "Explanation of the position and moves, or conversational response"
        },
        "ask_user": {
            "type": ["string", "null"],
            "description": "Optional question to engage the student"
        },
        "move_evaluation": {
            "type": "string",
            "enum": ["brilliant", "great", "best", "excellent", "good", "inaccuracy", "mistake", "blunder", "book"],
            "description": "Classification of the user's last move (move mode only)"
        },
        "grounding_citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Citations from tool outputs"
        }
    },
    "required": ["explain"]
}
