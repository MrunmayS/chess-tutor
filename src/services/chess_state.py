"""Chess state service using python-chess library."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import chess
import chess.pgn
import io


class GameStatusType(str, Enum):
    """Game status types."""
    ONGOING = "ongoing"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    FIFTY_MOVE_RULE = "fifty_move_rule"
    THREEFOLD_REPETITION = "threefold_repetition"


@dataclass
class MoveResult:
    """Result of applying a move."""
    success: bool
    new_fen: Optional[str] = None
    san: Optional[str] = None  # Standard Algebraic Notation
    uci: Optional[str] = None  # Universal Chess Interface notation
    error: Optional[str] = None
    is_capture: bool = False
    is_check: bool = False
    is_checkmate: bool = False


@dataclass
class GameStatus:
    """Current game status."""
    status: GameStatusType
    turn: str  # "white" or "black"
    is_game_over: bool
    winner: Optional[str] = None  # "white", "black", or None for draw
    fullmove_number: int = 1
    halfmove_clock: int = 0


class ChessStateService:
    """Service for managing chess game state using python-chess."""

    STARTING_FEN = chess.STARTING_FEN

    def apply_move(self, fen: str, move: str) -> MoveResult:
        """
        Apply a move to the given position.

        Args:
            fen: Current position in FEN notation
            move: Move in UCI (e.g., 'e2e4') or SAN (e.g., 'e4', 'Nf3') notation

        Returns:
            MoveResult with new FEN or error message
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return MoveResult(success=False, error=f"Invalid FEN: {e}")

        # Try parsing as UCI first, then SAN
        chess_move = None
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move not in board.legal_moves:
                chess_move = None
        except ValueError:
            pass

        if chess_move is None:
            try:
                chess_move = board.parse_san(move)
            except (chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                return MoveResult(success=False, error=f"Invalid move '{move}': {e}")

        if chess_move not in board.legal_moves:
            return MoveResult(
                success=False,
                error=f"Illegal move '{move}'. Legal moves: {self._format_legal_moves(board)}"
            )

        # Get move info before applying
        san = board.san(chess_move)
        uci = chess_move.uci()
        is_capture = board.is_capture(chess_move)

        # Apply the move
        board.push(chess_move)

        return MoveResult(
            success=True,
            new_fen=board.fen(),
            san=san,
            uci=uci,
            is_capture=is_capture,
            is_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
        )

    def legal_moves(self, fen: str, format: str = "both") -> dict:
        """
        Get all legal moves for the current position.

        Args:
            fen: Position in FEN notation
            format: "uci", "san", or "both"

        Returns:
            Dictionary with legal moves in requested format(s)
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}", "moves": []}

        moves_uci = [m.uci() for m in board.legal_moves]
        moves_san = [board.san(m) for m in board.legal_moves]

        result = {"count": len(moves_uci)}

        if format in ("uci", "both"):
            result["uci"] = moves_uci
        if format in ("san", "both"):
            result["san"] = moves_san
        if format == "both":
            result["pairs"] = [
                {"uci": u, "san": s} for u, s in zip(moves_uci, moves_san)
            ]

        return result

    def game_status(self, fen: str) -> GameStatus:
        """
        Get the current game status.

        Args:
            fen: Position in FEN notation

        Returns:
            GameStatus with current state information
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            return GameStatus(
                status=GameStatusType.ONGOING,
                turn="white",
                is_game_over=False,
                fullmove_number=1,
                halfmove_clock=0,
            )

        turn = "white" if board.turn == chess.WHITE else "black"
        winner = None

        if board.is_checkmate():
            status = GameStatusType.CHECKMATE
            winner = "black" if board.turn == chess.WHITE else "white"
        elif board.is_stalemate():
            status = GameStatusType.STALEMATE
        elif board.is_insufficient_material():
            status = GameStatusType.INSUFFICIENT_MATERIAL
        elif board.can_claim_fifty_moves():
            status = GameStatusType.FIFTY_MOVE_RULE
        elif board.can_claim_threefold_repetition():
            status = GameStatusType.THREEFOLD_REPETITION
        elif board.is_check():
            status = GameStatusType.CHECK
        else:
            status = GameStatusType.ONGOING

        return GameStatus(
            status=status,
            turn=turn,
            is_game_over=board.is_game_over(),
            winner=winner,
            fullmove_number=board.fullmove_number,
            halfmove_clock=board.halfmove_clock,
        )

    def fen_from_pgn(self, pgn: str) -> dict:
        """
        Parse PGN and return the final position FEN.

        Args:
            pgn: Game in PGN notation

        Returns:
            Dictionary with final FEN and move list
        """
        try:
            pgn_io = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_io)

            if game is None:
                return {"error": "Could not parse PGN", "fen": None}

            board = game.board()
            moves = []

            for move in game.mainline_moves():
                san = board.san(move)
                moves.append({"san": san, "uci": move.uci()})
                board.push(move)

            return {
                "fen": board.fen(),
                "moves": moves,
                "headers": dict(game.headers),
            }
        except Exception as e:
            return {"error": f"PGN parsing error: {e}", "fen": None}

    def board_ascii(self, fen: str) -> str:
        """
        Get ASCII representation of the board.

        Args:
            fen: Position in FEN notation

        Returns:
            ASCII art representation of the board
        """
        try:
            board = chess.Board(fen)
            return str(board)
        except ValueError:
            return "Invalid FEN"

    def _format_legal_moves(self, board: chess.Board, max_moves: int = 10) -> str:
        """Format legal moves for error messages."""
        moves = [board.san(m) for m in list(board.legal_moves)[:max_moves]]
        if len(list(board.legal_moves)) > max_moves:
            moves.append(f"... ({len(list(board.legal_moves))} total)")
        return ", ".join(moves)
