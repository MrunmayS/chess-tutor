"""Pydantic schemas for chess tutor."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Chess Domain Models
# ============================================================================

class GameStatusType(str, Enum):
    """Game status types."""
    ONGOING = "ongoing"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    FIFTY_MOVE_RULE = "fifty_move_rule"
    THREEFOLD_REPETITION = "threefold_repetition"


class MoveClassification(str, Enum):
    """Classification of move quality."""
    BRILLIANT = "brilliant"
    GREAT = "great"
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"
    BOOK = "book"


class MoveResult(BaseModel):
    """Result of applying a move."""
    success: bool
    new_fen: Optional[str] = None
    san: Optional[str] = None
    uci: Optional[str] = None
    error: Optional[str] = None
    is_capture: bool = False
    is_check: bool = False
    is_checkmate: bool = False


class GameStatus(BaseModel):
    """Current game status."""
    status: GameStatusType
    turn: str
    is_game_over: bool
    winner: Optional[str] = None
    fullmove_number: int = 1
    halfmove_clock: int = 0


class LineAnalysis(BaseModel):
    """Analysis of a single variation line."""
    move: str
    move_san: Optional[str] = None
    centipawn: Optional[int] = None
    mate_in: Optional[int] = None
    pv: list[str] = Field(default_factory=list)
    pv_san: list[str] = Field(default_factory=list)

    @property
    def eval_string(self) -> str:
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        if self.centipawn is not None:
            return f"{self.centipawn / 100:+.2f}"
        return "?"


class PositionAnalysis(BaseModel):
    """Full analysis of a position."""
    fen: str
    depth: int
    lines: list[LineAnalysis] = Field(default_factory=list)
    best_move: str
    best_move_san: Optional[str] = None
    evaluation: Optional[float] = None
    mate_in: Optional[int] = None
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False

    @property
    def eval_string(self) -> str:
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        if self.evaluation is not None:
            return f"{self.evaluation:+.2f}"
        return "?"


class MoveEvaluation(BaseModel):
    """Evaluation of a specific move."""
    move: str
    move_san: Optional[str] = None
    prev_eval: Optional[float] = None
    new_eval: Optional[float] = None
    delta: Optional[float] = None
    classification: MoveClassification = MoveClassification.GOOD
    best_move: Optional[str] = None
    best_move_san: Optional[str] = None
    explanation: str = ""


# ============================================================================
# Tutor Plan Schema (LLM Output)
# ============================================================================

class TutorPlan(BaseModel):
    """
    Structured output from the tutor LLM.

    This is the main response schema that the LLM produces after analyzing
    the user's move and position. It must include grounding citations.
    """
    opponent_reply: Optional[str] = Field(
        None,
        description="The tutor's reply move in SAN notation (e.g., 'c5', 'Nf6'). "
                    "None if waiting for user or game is over."
    )
    user_candidates: list[str] = Field(
        default_factory=list,
        description="Suggested candidate moves for the user to consider, in SAN notation."
    )
    explain: str = Field(
        ...,
        description="Explanation of the position, the user's move quality, "
                    "and strategic ideas. MUST cite engine analysis."
    )
    ask_user: Optional[str] = Field(
        None,
        description="Optional question to engage the user (e.g., 'Do you prefer "
                    "sharp tactical play or solid positional play?')"
    )
    move_evaluation: MoveClassification = Field(
        MoveClassification.GOOD,
        description="Classification of the user's last move."
    )
    grounding_citations: list[str] = Field(
        default_factory=list,
        description="REQUIRED: Citations from tool outputs that ground the explanation. "
                    "Format: 'tool_name: relevant_data' (e.g., 'analyze_position: eval +0.32, best Nf3')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "opponent_reply": "c5",
                    "user_candidates": ["Nf3", "d4", "Bc4"],
                    "explain": "You played the classic King's Pawn opening (1.e4). "
                               "This controls the center and opens lines for your bishop and queen. "
                               "I'll respond with the Sicilian Defense (1...c5), the most popular "
                               "response at the top level. The engine evaluates this as roughly equal.",
                    "ask_user": "Do you prefer sharp tactical battles or quieter positional play?",
                    "move_evaluation": "good",
                    "grounding_citations": [
                        "analyze_position: eval +0.32, depth 15",
                        "analyze_position: top moves Nf3 (+0.35), d4 (+0.30), Nc3 (+0.28)"
                    ]
                }
            ]
        }
    }
