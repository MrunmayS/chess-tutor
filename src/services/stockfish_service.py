"""Stockfish chess engine service."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import shutil

import chess
from stockfish import Stockfish


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


@dataclass
class LineAnalysis:
    """Analysis of a single variation line."""
    move: str  # Best move in UCI
    move_san: Optional[str] = None  # Best move in SAN (if available)
    centipawn: Optional[int] = None  # Evaluation in centipawns
    mate_in: Optional[int] = None  # Mate in N moves (positive = winning)
    pv: list[str] = field(default_factory=list)  # Principal variation (UCI)
    pv_san: list[str] = field(default_factory=list)  # Principal variation (SAN)

    @property
    def eval_string(self) -> str:
        """Human-readable evaluation."""
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        if self.centipawn is not None:
            score = self.centipawn / 100
            return f"{score:+.2f}"
        return "?"


@dataclass
class PositionAnalysis:
    """Full analysis of a position."""
    fen: str
    depth: int
    lines: list[LineAnalysis]
    best_move: str  # UCI
    best_move_san: Optional[str] = None
    evaluation: Optional[float] = None  # In pawns (centipawns / 100)
    mate_in: Optional[int] = None
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False

    @property
    def eval_string(self) -> str:
        """Human-readable evaluation."""
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        if self.evaluation is not None:
            return f"{self.evaluation:+.2f}"
        return "?"


@dataclass
class MoveEvaluation:
    """Evaluation of a specific move."""
    move: str
    move_san: Optional[str] = None
    prev_eval: Optional[float] = None  # Eval before move
    new_eval: Optional[float] = None  # Eval after move
    delta: Optional[float] = None  # Change in evaluation
    classification: MoveClassification = MoveClassification.GOOD
    best_move: Optional[str] = None  # What was the best move
    best_move_san: Optional[str] = None
    explanation: str = ""


class StockfishService:
    """Service for chess analysis using Stockfish engine."""

    # Thresholds for move classification (in centipawns)
    THRESHOLDS = {
        "blunder": 200,    # Loses >= 2 pawns
        "mistake": 100,    # Loses >= 1 pawn
        "inaccuracy": 50,  # Loses >= 0.5 pawns
    }

    def __init__(
        self,
        path: Optional[str] = None,
        depth: int = 15,
        multipv: int = 3,
    ):
        """
        Initialize Stockfish service.

        Args:
            path: Path to Stockfish binary (auto-detect if None)
            depth: Default analysis depth
            multipv: Default number of lines to analyze
        """
        self.default_depth = depth
        self.default_multipv = multipv

        # Find Stockfish binary
        if path is None:
            path = shutil.which("stockfish")
            if path is None:
                raise RuntimeError(
                    "Stockfish not found. Install with: "
                    "apt install stockfish (Linux) or brew install stockfish (macOS)"
                )

        self._engine = Stockfish(path=path)
        self._engine.set_depth(depth)

    def analyze(
        self,
        fen: str,
        depth: Optional[int] = None,
        multipv: Optional[int] = None,
    ) -> PositionAnalysis:
        """
        Analyze a chess position.

        Args:
            fen: Position in FEN notation
            depth: Analysis depth (uses default if None)
            multipv: Number of lines to analyze (uses default if None)

        Returns:
            PositionAnalysis with top moves and evaluations
        """
        depth = depth or self.default_depth
        multipv = multipv or self.default_multipv

        self._engine.set_fen_position(fen)
        self._engine.set_depth(depth)

        # Get top N lines
        top_moves = self._engine.get_top_moves(multipv)

        lines = []
        for move_info in top_moves:
            centipawn = move_info.get("Centipawn")
            mate = move_info.get("Mate")

            line = LineAnalysis(
                move=move_info["Move"],
                centipawn=centipawn,
                mate_in=mate,
            )
            lines.append(line)

        # Get best move
        best_move = self._engine.get_best_move()

        # Get evaluation
        eval_info = self._engine.get_evaluation()
        evaluation = None
        mate_in = None

        if eval_info["type"] == "cp":
            evaluation = eval_info["value"] / 100
        elif eval_info["type"] == "mate":
            mate_in = eval_info["value"]

        # Check game state (use python-chess; wrapper APIs vary by version)
        board = chess.Board(fen)
        is_check = board.is_check()
        is_checkmate = board.is_checkmate()
        is_stalemate = board.is_stalemate()

        return PositionAnalysis(
            fen=fen,
            depth=depth,
            lines=lines,
            best_move=best_move or "",
            evaluation=evaluation,
            mate_in=mate_in,
            is_check=is_check,
            is_checkmate=is_checkmate,
            is_stalemate=is_stalemate,
        )

    def evaluate_move(
        self,
        prev_fen: str,
        move: str,
        depth: Optional[int] = None,
    ) -> MoveEvaluation:
        """
        Evaluate a specific move.

        Args:
            prev_fen: Position before the move
            move: Move in UCI notation
            depth: Analysis depth

        Returns:
            MoveEvaluation with classification and delta
        """
        depth = depth or self.default_depth

        # Analyze position before move
        self._engine.set_fen_position(prev_fen)
        self._engine.set_depth(depth)

        best_move = self._engine.get_best_move()
        eval_before = self._engine.get_evaluation()

        # Apply the move and analyze
        if not self._engine.is_move_correct(move):
            return MoveEvaluation(
                move=move,
                classification=MoveClassification.BLUNDER,
                explanation=f"Invalid move: {move}",
            )

        self._engine.make_moves_from_current_position([move])
        eval_after = self._engine.get_evaluation()

        # Calculate evaluation delta
        prev_eval = self._extract_eval(eval_before)
        new_eval = self._extract_eval(eval_after)

        # Flip sign since evaluation is from the opponent's perspective after move
        if new_eval is not None:
            new_eval = -new_eval

        delta = None
        if prev_eval is not None and new_eval is not None:
            delta = new_eval - prev_eval

        # Classify the move
        classification = self._classify_move(move, best_move, delta, prev_eval, new_eval)

        explanation = self._generate_explanation(
            classification, delta, best_move, move
        )

        return MoveEvaluation(
            move=move,
            prev_eval=prev_eval,
            new_eval=new_eval,
            delta=delta,
            classification=classification,
            best_move=best_move,
            explanation=explanation,
        )

    def _extract_eval(self, eval_info: dict) -> Optional[float]:
        """Extract evaluation as float (in pawns)."""
        if eval_info["type"] == "cp":
            return eval_info["value"] / 100
        elif eval_info["type"] == "mate":
            # Convert mate to a large value
            mate_val = eval_info["value"]
            return 100 if mate_val > 0 else -100
        return None

    def _classify_move(
        self,
        move: str,
        best_move: Optional[str],
        delta: Optional[float],
        prev_eval: Optional[float],
        new_eval: Optional[float],
    ) -> MoveClassification:
        """Classify a move based on evaluation change."""
        # Best move check
        if move == best_move:
            if delta is not None and delta > 0.5:
                return MoveClassification.BRILLIANT
            return MoveClassification.BEST

        if delta is None:
            return MoveClassification.GOOD

        # Convert delta to centipawns for threshold comparison
        delta_cp = abs(delta * 100)

        if delta_cp >= self.THRESHOLDS["blunder"]:
            return MoveClassification.BLUNDER
        elif delta_cp >= self.THRESHOLDS["mistake"]:
            return MoveClassification.MISTAKE
        elif delta_cp >= self.THRESHOLDS["inaccuracy"]:
            return MoveClassification.INACCURACY
        elif delta >= 0:
            return MoveClassification.EXCELLENT
        else:
            return MoveClassification.GOOD

    def _generate_explanation(
        self,
        classification: MoveClassification,
        delta: Optional[float],
        best_move: Optional[str],
        played_move: str,
    ) -> str:
        """Generate explanation for move classification."""
        explanations = {
            MoveClassification.BRILLIANT: "An exceptional move that creates winning chances!",
            MoveClassification.BEST: "This is the engine's top choice.",
            MoveClassification.EXCELLENT: "A very strong move.",
            MoveClassification.GREAT: "A strong move that maintains advantage.",
            MoveClassification.GOOD: "A solid move.",
            MoveClassification.INACCURACY: f"Slightly imprecise. Better was {best_move}.",
            MoveClassification.MISTAKE: f"This loses material or position. Better was {best_move}.",
            MoveClassification.BLUNDER: f"A serious error! {best_move} was much better.",
        }

        base = explanations.get(classification, "")

        if delta is not None and classification in (
            MoveClassification.INACCURACY,
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        ):
            base += f" (eval change: {delta:+.2f})"

        return base

    def close(self):
        """Close the Stockfish engine."""
        if hasattr(self, '_engine'):
            del self._engine
