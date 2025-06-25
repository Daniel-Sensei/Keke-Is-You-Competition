# Keke Is You Python framework

KekeAI is a comprehensive platform for developing, testing, and evaluating intelligent agents on puzzle games inspired by Baba Is You. The platform provides a complete framework for implementing search algorithms and AI agents that can solve rule-based puzzle games.

## Project Structure

The project is organized into several core modules:

```
KekeAI/
├── app.py                  # Flask web application (frontend and API)
├── baba.py                 # Core game engine and mechanics
├── base_agent.py           # Base class for all agents
├── execution.py            # Agent execution and evaluation system
├── agents/                 # Agent implementations
│   ├── bfs_AGENT.py        # Breadth-First Search agent
│   ├── dfs_AGENT.py        # Depth-First Search agent
│   └── astar_AGENT.py      # A* Search agent with heuristic
├── json_levels/            # Game levels in JSON format
│   ├── demo_LEVELS.json    # Demo levels for testing
│   ├── train_LEVELS.json   # Training levels
│   ├── test_LEVELS.json    # Test levels for evaluation
│   └── ...                 # Additional level sets
├── img/                    # Game assets and graphics
├── static/                 # Web frontend static files
│   ├── css/                # Stylesheets
│   ├── img/                # Frontend images
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
├── results/                # Cached execution results
└── solutions/              # Solution files
```

## Core Components

### Game Engine (baba.py)

The `baba.py` module implements the complete game mechanics inspired by Baba Is You:

- **Rule System**: Dynamic rule parsing and application (e.g., "BABA IS YOU", "FLAG IS WIN")
- **Game State Management**: Complete state representation with entities, rules, and positions
- **Movement Logic**: Handles player movement, pushing mechanics, and rule interactions
- **Win Conditions**: Automatic detection of win/lose states based on active rules
- **Map Parsing**: Converts ASCII maps to internal game representations

Key functions:
- `parse_map(ascii_map)`: Converts ASCII representation to game data
- `make_level(map_data)`: Creates a GameState from parsed map data
- `advance_game_state(action, state)`: Applies an action and returns new state
- `check_win(state)`: Checks if the current state is a winning state

### Agent Framework

#### BaseAgent Class

All agents must extend the `BaseAgent` class and implement the `search` method:

```python
class BaseAgent(ABC):
    @abstractmethod
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Performs search to find a solution.
        
        Args:
            initial_state: The initial game state
            iterations: Maximum number of iterations/depth
            
        Returns:
            List of Direction enums representing the solution path
        """
        pass
```

#### Implemented Agents

1. **BFSAgent** (`bfs_AGENT.py`): Implements Breadth-First Search
   - Guarantees optimal solution (shortest path)
   - Uses queue-based exploration
   - Suitable for simpler levels

2. **DFSAgent** (`dfs_AGENT.py`): Implements Depth-First Search
   - Memory efficient
   - May find suboptimal solutions
   - Good for exploring deep solution spaces

3. **AStarAgent** (`astar_AGENT.py`): Implements A* Search with heuristic

### Execution System

The `Execution` class manages agent evaluation:

- **Dynamic Agent Loading**: Loads agents from Python files at runtime
- **Level Management**: Handles JSON level loading and parsing
- **Performance Monitoring**: Tracks execution time, iterations, and success rates
- **Result Caching**: Caches results to avoid re-running expensive computations
- **Progress Tracking**: Real-time progress updates for web interface

Key features:
- Configurable iteration limits
- Detailed error reporting and debugging
- Automatic result serialization
- Support for multiple level sets

### Web Interface

The Flask-based web application provides:

- **Agent Selection**: Choose from available agent implementations
- **Level Set Management**: Select and load different level collections
- **Real-time Execution**: Live progress updates during agent execution
- **Result Visualization**: Detailed performance metrics and solution replay
- **Solution Analysis**: Step-by-step solution visualization

## Game Level Format

Levels are stored as JSON files with the `_LEVELS.json` suffix:

```json
{
    "levels": [
        {
            "id": 1,
            "name": "Basic Movement",
            "author": "Demo",
            "ascii": "__________\n_B12..F13_\n_........_\n_.b....f._\n__________",
            "solution": "rrrrr"
        }
    ]
}
```

### ASCII Map Characters

The game uses specific ASCII characters to represent different entities:

**Objects (game entities):**
- `b`: Baba (player character)
- `f`: Flag (common win condition)
- `r`: Rock (pushable object)
- `w`: Wall (obstacle)
- `s`: Skull (dangerous object)
- `_`: Border (level boundary)
- ` `: Empty space

**Words (rule components):**
- `B`: "BABA" word
- `F`: "FLAG" word
- `R`: "ROCK" word
- `W`: "WALL" word
- `1`: "IS" word
- `2`: "YOU" word
- `3`: "WIN" word
- `4`: "STOP" word
- `5`: "PUSH" word
- `6`: "SINK" word
- `7`: "KILL" word

### Rule Formation

Words can be arranged to form rules:
- `B12` = "BABA IS YOU" (makes Baba controllable)
- `F13` = "FLAG IS WIN" (makes Flag the win condition)
- `R15` = "ROCK IS PUSH" (makes Rock pushable)
- `W14` = "WALL IS STOP" (makes Wall an obstacle)

## Adding New Agents

To create a new agent:

1. **Create the agent file**: `agents/your_algorithm_AGENT.py`
2. **Implement the agent class**:

```python
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List

class YourAlgorithmAgent(BaseAgent):
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        # Implement your search algorithm here
        # Return a list of Direction enums
        pass
```

3. **Test the agent**: The web interface will automatically detect and load your agent

## Adding New Level Sets

1. **Create level file**: `json_levels/your_levels_LEVELS.json`
2. **Follow the JSON format**: Use the structure shown above
3. **Design ASCII maps**: Create challenging and interesting puzzles
4. **Test levels**: Ensure they are solvable and properly formatted

## Installation and Setup

### Prerequisites

- Python 3.8+
- Required packages: `flask`, `pygame`, `tqdm`

### Installation

```bash
# Clone the repository
git clone https://github.com/edoardoedr/KekeIsYou_PY.git
cd KekeAI

# Install dependencies
pip install flask pygame tqdm

# Run the application
python app.py
```

The web interface will be available at `http://localhost:5000`

## Usage Examples

### Running via Web Interface

1. Open your browser to `http://localhost:5000`
2. Select an agent (e.g., "astar")
3. Choose a level set (e.g., "demo")
4. Set iteration limit (default: 1000)
5. Click "Start Simulation"
6. View real-time progress and results

### Programmatic Usage

```python
from execution import Execution

# Create execution instance
executor = Execution(
    agent_path="agents/astar_AGENT.py",
    level_path="json_levels/demo_LEVELS.json",
    iter_cap=1000
)

# Run all levels
results = executor.run_all_levels()

# Access results
for level_result in results['levels']:
    print(f"Level: {level_result['id']}")
    print(f"Status: {level_result['status']}")
    print(f"Solution: {level_result['solution']}")
```

## Performance Evaluation

The system provides comprehensive performance metrics:

- **Success Rate**: Percentage of levels solved
- **Average Time**: Mean execution time per level
- **Solution Quality**: Solution length and optimality
- **Efficiency**: Iterations used vs. limit
- **Scalability**: Performance across different level complexities

## Architecture Design

### Modular Design

- **Separation of Concerns**: Clear boundaries between game engine, agents, and interface
- **Plugin Architecture**: Easy addition of new agents and level sets
- **Caching System**: Efficient result storage and retrieval
- **Error Handling**: Robust error recovery and reporting

### Extensibility

- **Agent Framework**: Simple interface for implementing new algorithms
- **Level Format**: Flexible JSON-based level definition
- **Web API**: RESTful endpoints for external integration
- **Visualization**: Extensible rendering system

## Contributing

When contributing to KekeAI:

1. **Follow the agent naming convention**: `algorithm_AGENT.py`
2. **Implement proper error handling**: Handle edge cases gracefully
3. **Add comprehensive documentation**: Document algorithm specifics
4. **Test thoroughly**: Verify agent works on multiple level sets
5. **Consider performance**: Optimize for the iteration limits

## License

This project is designed for educational and research purposes in artificial intelligence and game-solving algorithms.

## Acknowledgments

This implementation is inspired by the official GitHub repository of the competition: [KekeCompetition](https://github.com/MasterMilkX/KekeCompetition). The rules are derived from the original GitHub repository, but the graphical interface implementation has been developed by the author of this repository.

The game itself is inspired by the official paper: ["Baba Is You: A Rule-Based Puzzle Game"](https://ieeexplore.ieee.org/document/9893650/footnotes#footnotes-id-fn1).
