# Interactive Chess Tutor

An AI-powered chess tutor that uses a Quick LLM architecture with deterministic chess services and structured reasoning.

## Features

- **Interactive Gameplay**: Play chess against the tutor with real-time feedback
- **Move Evaluation**: Get detailed analysis of your moves (blunder, mistake, inaccuracy, good, excellent, best)
- **Engine-Grounded Explanations**: All advice is backed by Stockfish analysis with citations
- **Adaptive Teaching**: Adjusts explanations based on your skill level
- **vLLora Tracing**: Supports `x-run-id` and `x-thread-id` headers for tracing via vLLora gateway

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tutor Orchestrator                       │
│  (Agentic loop with OpenAI function calling)               │
│  Headers: x-run-id (per turn), x-thread-id (per session)   │
├─────────────────────────────────────────────────────────────┤
│                         Tools                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ apply_move  │  │ analyze_pos │  │ evaluate_move       │ │
│  │ legal_moves │  │ game_status │  │ parse_pgn           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Services                               │
│  ┌──────────────────┐     ┌────────────────────────────┐   │
│  │ ChessStateService│     │ StockfishService           │   │
│  │ (python-chess)   │     │ (Stockfish engine)         │   │
│  └──────────────────┘     └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- OpenAI API key (or vLLora gateway)
- Stockfish chess engine

## Installation

1. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

2. **Install Stockfish** (if not already installed):
   ```bash
   # Linux (Debian/Ubuntu)
   sudo apt install stockfish

   # macOS
   brew install stockfish
   ```

3. **Configure your environment:**
   Edit `.env` and set your configuration:
   ```bash
   # For direct OpenAI API usage:
   OPENAI_API_KEY=your-key-here

   # For vLLora gateway (tracing enabled):
   USE_LOCAL_GATEWAY=true
   LLM_BASE_URL=http://localhost:9090/v1
   ```

## Usage

### Interactive Demo

```bash
source venv/bin/activate
python examples/demo.py
```

### Commands

- Enter moves in SAN (`e4`, `Nf3`, `O-O`) or UCI (`e2e4`) notation
- `board` - Display current position
- `status` - Show game status
- `new` - Start a new game
- `quit` - Exit

### Example Session

```
You: e4
Tutor: Move Quality: 👌 GOOD

You played 1.e4, the King's Pawn opening - one of the most popular and
aggressive first moves. The engine evaluates this as +0.32, slightly
favoring White. This move controls the center and opens lines for your
bishop and queen.

🎯 My move: c5

I'll respond with the Sicilian Defense (1...c5), the most popular
response at the top level. It fights for central control while
avoiding symmetry.

💡 Candidates to consider: Nf3, Nc3, d4

❓ Do you prefer tactical battles or quieter positional play?

📊 Analysis: analyze_position: eval +0.32, best move Nf3 at depth 15...
```

## vLLora Tracing

When `USE_LOCAL_GATEWAY=true`, the tutor sends tracing headers with each LLM request:

- **`x-thread-id`**: Unique per game session (persists across all turns)
- **`x-run-id`**: Unique per user turn (same for all LLM calls within one turn's agentic loop)

This enables full observability of the tutor's reasoning through vLLora.

## Project Structure

```
chess-tutor/
├── .env.example           # Environment template
├── .gitignore
├── README.md
├── pyproject.toml         # Project configuration and dependencies
├── examples/
│   └── demo.py            # Interactive CLI demo
└── src/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   ├── tutor_orchestrator.py  # Main agentic loop
    │   └── tools.py               # Tool definitions + executor
    ├── services/
    │   ├── __init__.py
    │   ├── chess_state.py         # python-chess wrapper
    │   └── stockfish_service.py   # Stockfish wrapper
    └── utils/
        ├── __init__.py
        ├── config.py              # Environment config
        ├── schemas.py             # Pydantic models
        └── prompts.py             # System/turn prompts
```

## TutorPlan Schema

The LLM outputs structured JSON:

```json
{
  "opponent_reply": "c5",
  "user_candidates": ["Nf3", "d4", "Bc4"],
  "explain": "Explanation with engine citations...",
  "ask_user": "Optional engagement question?",
  "move_evaluation": "good",
  "grounding_citations": [
    "analyze_position: eval +0.32, best Nf3",
    "evaluate_move: classification=good, delta=-0.10"
  ]
}
```

## Configuration

Environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required unless using vLLora) |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |
| `USE_LOCAL_GATEWAY` | Route through vLLora | `false` |
| `LLM_BASE_URL` | vLLora gateway URL | `http://localhost:9090/v1` |
| `STOCKFISH_PATH` | Path to Stockfish binary | (auto-detect) |
| `STOCKFISH_DEPTH` | Analysis depth | `15` |
| `STOCKFISH_MULTIPV` | Number of lines to analyze | `3` |

## Dependencies

- `openai>=1.12.0` - OpenAI API client
- `python-chess>=1.10.0` - Chess move validation and board state
- `stockfish>=3.28.0` - Stockfish engine wrapper
- `python-dotenv>=1.0.0` - Environment variable loading
- `pydantic>=2.0.0` - Data validation
- `pydantic-settings>=2.0.0` - Settings management

## License

MIT
