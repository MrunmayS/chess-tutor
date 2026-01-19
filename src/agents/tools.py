"""Tool definitions and executor for chess tutor."""

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from ..services.chess_state import ChessStateService
from ..services.stockfish_service import StockfishService


# ============================================================================
# Tool Definitions (OpenAI Function Calling Format)
# ============================================================================

# Full tools for move mode (includes apply_move for tutor's response)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "apply_move",
            "description": "Validate and apply a chess move to the current position. "
                           "Returns the new FEN if valid, or an error message if invalid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Current position in FEN notation"
                    },
                    "move": {
                        "type": "string",
                        "description": "Move in UCI (e.g., 'e2e4') or SAN (e.g., 'e4', 'Nf3') notation"
                    }
                },
                "required": ["fen", "move"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_moves",
            "description": "Get all legal moves for the current position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Current position in FEN notation"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["uci", "san", "both"],
                        "description": "Output format for moves (default: both)"
                    }
                },
                "required": ["fen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_status",
            "description": "Get the current game status (check, checkmate, stalemate, ongoing, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Current position in FEN notation"
                    }
                },
                "required": ["fen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_position",
            "description": "Analyze a position using the Stockfish chess engine. "
                           "Returns evaluation, best moves, and principal variations. "
                           "IMPORTANT: You MUST cite this analysis in your explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation to analyze"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth (default: 15, higher = more accurate but slower)"
                    },
                    "multipv": {
                        "type": "integer",
                        "description": "Number of best lines to return (default: 3)"
                    }
                },
                "required": ["fen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_move",
            "description": "Evaluate a specific move to determine if it's a blunder, mistake, "
                           "inaccuracy, or good move. Compares to the engine's best move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prev_fen": {
                        "type": "string",
                        "description": "Position BEFORE the move in FEN notation"
                    },
                    "move": {
                        "type": "string",
                        "description": "Move to evaluate in UCI notation"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth (default: 15)"
                    }
                },
                "required": ["prev_fen", "move"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_pgn",
            "description": "Parse a PGN (Portable Game Notation) string and return the final position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pgn": {
                        "type": "string",
                        "description": "Game in PGN notation"
                    }
                },
                "required": ["pgn"]
            }
        }
    }
]


# Chat-only tools (no apply_move - we don't change the board in chat mode)
# Only includes analysis tools for discussing positions
CHAT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_legal_moves",
            "description": "Get all legal moves for the current position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Current position in FEN notation"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["uci", "san", "both"],
                        "description": "Output format for moves (default: both)"
                    }
                },
                "required": ["fen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_status",
            "description": "Get the current game status (check, checkmate, stalemate, ongoing, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Current position in FEN notation"
                    }
                },
                "required": ["fen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_position",
            "description": "Analyze a position using the Stockfish chess engine. "
                           "Returns evaluation, best moves, and principal variations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation to analyze"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth (default: 15)"
                    },
                    "multipv": {
                        "type": "integer",
                        "description": "Number of best lines to return (default: 3)"
                    }
                },
                "required": ["fen"]
            }
        }
    }
]


# ============================================================================
# Tool Executor
# ============================================================================

class ToolExecutor:
    """Executes tools and manages chess services."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 15,
        stockfish_multipv: int = 3,
    ):
        """
        Initialize tool executor with chess services.

        Args:
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            stockfish_depth: Default analysis depth
            stockfish_multipv: Default number of lines to analyze
        """
        self.chess_service = ChessStateService()
        self.stockfish_service = StockfishService(
            path=stockfish_path,
            depth=stockfish_depth,
            multipv=stockfish_multipv,
        )

        # Map tool names to handlers
        self._handlers: dict[str, Callable[..., Any]] = {
            "apply_move": self._apply_move,
            "get_legal_moves": self._get_legal_moves,
            "get_game_status": self._get_game_status,
            "analyze_position": self._analyze_position,
            "evaluate_move": self._evaluate_move,
            "parse_pgn": self._parse_pgn,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        if tool_name not in self._handlers:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = self._handlers[tool_name](**arguments)
        except Exception as e:
            result = {"error": str(e)}

        return result

    def _serialize_result(self, obj: Any) -> Any:
        """Serialize dataclass or other objects to dict."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    def _apply_move(self, fen: str, move: str) -> dict[str, Any]:
        """Apply a move to the position."""
        result = self.chess_service.apply_move(fen, move)
        return self._serialize_result(result)

    def _get_legal_moves(self, fen: str, format: str = "both") -> dict[str, Any]:
        """Get legal moves for the position."""
        return self.chess_service.legal_moves(fen, format)

    def _get_game_status(self, fen: str) -> dict[str, Any]:
        """Get game status for the position."""
        result = self.chess_service.game_status(fen)
        return self._serialize_result(result)

    def _analyze_position(
        self,
        fen: str,
        depth: Optional[int] = None,
        multipv: Optional[int] = None,
    ) -> dict[str, Any]:
        """Analyze position with Stockfish."""
        result = self.stockfish_service.analyze(fen, depth, multipv)
        return self._serialize_result(result)

    def _evaluate_move(
        self,
        prev_fen: str,
        move: str,
        depth: Optional[int] = None,
    ) -> dict[str, Any]:
        """Evaluate a specific move."""
        result = self.stockfish_service.evaluate_move(prev_fen, move, depth)
        return self._serialize_result(result)

    def _parse_pgn(self, pgn: str) -> dict[str, Any]:
        """Parse PGN to final FEN."""
        return self.chess_service.fen_from_pgn(pgn)

    def close(self):
        """Clean up resources."""
        self.stockfish_service.close()
