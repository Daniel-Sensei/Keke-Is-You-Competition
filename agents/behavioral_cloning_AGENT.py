"""
Behavioral Cloning Agent for KekeAI.

This agent is designed to use a pre-trained model (e.g., a neural network)
to decide which action to take based on the current game state. The model
itself would be trained offline using supervised learning on a dataset of
(state, expert_action) pairs derived from solved KekeAI levels.

This file defines the agent structure, including state vectorization,
and how it would interact with a loaded model for inference.
The actual model training pipeline and model file are external to this agent.

# -----------------------------------------------------------------------------
# Conceptual Offline Data Preprocessing and Model Training Pipeline:
# -----------------------------------------------------------------------------
# 1. Data Collection:
#    - Use `train_LEVELS.json` (or any set of levels with provided solutions).
#    - For each level:
#        - Parse the `ascii` map to get the `initial_state` using `baba.parse_map` and `baba.make_level`.
#        - Convert the `solution` string (e.g., "RRDLU") into a sequence of `Direction` enums.
#
# 2. Trajectory Unrolling:
#    - For each level and its solution action sequence:
#        - Start with the `initial_state`.
#        - Iteratively apply each action from the solution sequence using `baba.advance_game_state`.
#        - At each step `t`:
#            - Store the pair: (`current_state_t`, `action_taken_at_t`).
#            - The `current_state_t` is the state *before* `action_taken_at_t` is applied.
#        - This creates a list of (state, action) pairs representing expert moves.
#
# 3. State Vectorization & Data Formatting:
#    - For every `(state_t, action_t)` pair collected:
#        - Convert `state_t` into its numerical vector representation using `_vectorize_state(state_t)`.
#        - Convert `action_t` into a numerical label (e.g., an integer index from `ACTION_TO_INDEX`).
#    - This results in a dataset `D = [(vectorized_state_1, label_1), ..., (vectorized_state_N, label_N)]`.
#
# 4. Model Architecture (Example for an MLP or CNN->MLP):
#    - Input Layer: Size = `FEATURE_VECTOR_SIZE`.
#    - If using CNN for grid part: CNN layers (e.g., Conv2D, ReLU, Pooling) processing the
#      reshaped grid portion of the input vector.
#    - Hidden Layer(s): One or more fully connected layers (e.g., 256 units, 128 units) with ReLU activation.
#      These layers would process the (potentially flattened CNN output and) rule features.
#    - Output Layer: Size = `NUM_ACTIONS` (e.g., 5) with a `softmax` activation function
#      to output probabilities for each action.
#
# 5. Model Training (Supervised Learning):
#    - Use a standard machine learning library (PyTorch, TensorFlow/Keras).
#    - Split dataset D into training and validation sets.
#    - Train the model to minimize a loss function suitable for classification (e.g., cross-entropy loss).
#    - Optimizer: Adam, SGD, etc.
#    - Monitor validation accuracy/loss to prevent overfitting and for model selection.
#
# 6. Model Saving:
#    - Save the trained model's weights and architecture (e.g., to an .h5 file for Keras, .pth for PyTorch).
#
# 7. Agent Usage (Inference):
#    - The `BehavioralCloningAgent`'s `__init__` method would load these saved weights
#      into an identical model architecture.
#    - The `_get_action_from_model` method would then use this loaded model to predict
#      the best action for a given vectorized state.
# -----------------------------------------------------------------------------
"""
import random
import numpy as np # For numerical operations and state vector
from typing import List, Tuple, Optional, Any, Set # Added Set for type hints
import os # For checking model file path

# Attempt to import TensorFlow, but make it optional for agent definition
# as training happens offline. For actual use with a trained model, TF is required.
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # This print will occur at import time if TF is not found.
    # print("WARNING: TensorFlow not found. BehavioralCloningAgent will only work with a dummy model if a real model is attempted to be loaded.")

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj # GameObj for type hints

# --- State Vectorization Parameters ---
MAX_MAP_HEIGHT = 25 # Max height for the grid representation. Levels larger are cropped.
MAX_MAP_WIDTH = 25  # Max width for the grid representation. Levels larger are cropped.

# Defines a mapping from semantic object categories or specific text names to channel indices
# in the grid-based part of the state vector.
OBJECT_NAME_TO_CHANNEL = {
    # Dynamic categories based on rules
    "player_controlled": 0,  # Objects that are currently "YOU"
    "winnable_condition": 1, # Objects that are currently "WIN" (e.g., flag if FLAG IS WIN)
    "pushable_physical": 2,  # Physical objects that currently have the "PUSH" property
    "stopping_physical": 3,  # Physical objects that currently have the "STOP" property
    "dangerous_physical": 4, # Physical objects that are "KILL" (or other danger combinations)
    "general_physical": 5,   # Other physical objects not fitting special categories above
    
    # Specific Text Nouns (prefix "text_")
    "text_baba": 6, "text_flag": 7, "text_rock": 8, "text_wall": 9, 
    "text_skull": 10, "text_keke": 11, "text_lava": 12, "text_goop": 13,
    
    # Specific Text Operator
    "text_is": 14,
    
    # Specific Text Properties (prefix "text_")
    "text_you": 15, "text_win": 16, "text_push": 17, "text_stop": 18, 
    "text_kill": 19, "text_move": 20, "text_hot": 21, "text_melt": 22, "text_sink": 23,
    # Channel N-1 (NUM_OBJECT_CHANNELS - 1) is reserved for "other_text" if a text word is not in this map.
}
# Number of channels for the grid part of the state vector.
# One channel for each entry in OBJECT_NAME_TO_CHANNEL, plus one for "other_text".
NUM_OBJECT_CHANNELS = len(OBJECT_NAME_TO_CHANNEL) + 1 

# Number of additional features appended to the flattened grid features, representing global rule states.
NUM_RULE_FEATURES = 2 # Currently: [has_any_X_IS_YOU_rule, has_any_Y_IS_WIN_rule]

# Total size of the flattened feature vector fed to the model.
FEATURE_VECTOR_SIZE = (NUM_OBJECT_CHANNELS * MAX_MAP_HEIGHT * MAX_MAP_WIDTH) + NUM_RULE_FEATURES

# Mapping actions to integer indices for model output layer, and back.
ACTION_TO_INDEX = {Direction.Up: 0, Direction.Down: 1, Direction.Left: 2, Direction.Right: 3, Direction.Wait: 4}
INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}
NUM_ACTIONS = len(ACTION_TO_INDEX) # Should be 5


class BEHAVIORAL_CLONINGAgent(BaseAgent): # Renamed class to match expected convention
    """
    Agent that uses a (hypothetically) pre-trained model for Behavioral Cloning.
    It vectorizes the game state into a numerical format and queries the loaded model
    (or a dummy model if loading fails/TensorFlow is unavailable) to predict the
    next best action based on learned expert behavior.
    
    The actual model training is an offline process not handled by this class.
    This agent expects a Keras model saved in .h5 format.
    """

    def __init__(self, model_path="keke_behavioral_cloning_model.h5"):
        """
        Initializes the Behavioral Cloning agent.

        Attempts to load a pre-trained Keras model from the specified `model_path`.
        If TensorFlow is not available in the environment, or if the model file
        is not found or fails to load, the agent falls back to using a dummy model
        that predicts random actions. This ensures the agent can always be instantiated,
        even if a trained model is not yet available or the environment lacks TensorFlow.

        Args:
            model_path (str): Path to the saved Keras model file (e.g., 'my_model.h5').
                              Defaults to "keke_behavioral_cloning_model.h5", assuming the
                              model might be placed in the same directory as the script execution.
        """
        super().__init__()
        self.model: Any = None  # Will hold the loaded Keras model or a DummyModel instance.
        self.object_name_to_channel_map = OBJECT_NAME_TO_CHANNEL 
        self.num_object_channels = NUM_OBJECT_CHANNELS
        self._load_model(model_path)

    def _create_dummy_model(self):
        """
        Creates and returns an instance of a simple dummy model.
        The dummy model has a `predict` method that returns a random action index.
        This allows the agent's structure and data pipeline to be tested
        without requiring a real trained neural network model.
        """
        class DummyModel:
            def predict(self, vectorized_state: np.array) -> int:
                """Simulates model prediction by returning a random action index."""
                return random.randint(0, NUM_ACTIONS - 1)
        return DummyModel()

    def _load_model(self, model_path: str):
        """
        Attempts to load a trained Keras model from the specified `model_path`.
        
        If TensorFlow (`TF_AVAILABLE` global flag) is true and the model file exists
        at `model_path`, it tries to load it using `keras.models.load_model`.
        If loading is successful, `self.model` is set to the loaded Keras model.
        
        If TensorFlow is not available, the model file doesn't exist at the path,
        or any error occurs during loading, a warning/error message is printed to console,
        and `self.model` is set to an instance of `DummyModel` as a fallback.

        Args:
            model_path (str): The file path to the Keras model (typically .h5 format).
        """
        if TF_AVAILABLE: 
            if os.path.exists(model_path):
                try:
                    self.model = keras.models.load_model(model_path)
                    print(f"INFO: Successfully loaded trained Keras model from '{model_path}'")
                    # Optional: A "warm-up" prediction can be useful if model loading is lazy
                    # or to ensure the model is compatible with the expected input shape immediately.
                    # Example:
                    # if hasattr(self.model, 'predict') and FEATURE_VECTOR_SIZE > 0:
                    #    dummy_input = np.zeros((1, FEATURE_VECTOR_SIZE), dtype=np.float32)
                    #    self.model.predict(dummy_input, verbose=0) # verbose=0 for no Keras progress bar
                    return 
                except Exception as e:
                    print(f"ERROR: Failed to load Keras model from '{model_path}'. Error: {e}")
            else:
                 print(f"WARNING: Model file not found at '{model_path}'. Please ensure the path is correct.")
        else:
            # This warning is now printed at import time if TF is not available.
            # Adding a specific message here if model loading is attempted without TF.
            if not TF_AVAILABLE:
                 print(f"INFO: TensorFlow not available, cannot load Keras model.")
        
        print("INFO: BehavioralCloningAgent is falling back to using a DUMMY model (predicts random actions).")
        self.model = self._create_dummy_model()


    def _vectorize_state(self, state: GameState) -> np.array:
        """
        Converts a given GameState object into a flat numerical vector (NumPy array)
        suitable as input for a neural network.

        The vector consists of:
        1.  A multi-channel 2D grid representation:
            - Each channel corresponds to a specific object type (e.g., player, winnable,
              specific text words) or a dynamic property (e.g., pushable, stopping).
            - The grid cells are binary (1.0 if the object/property is present, 0.0 otherwise).
            - The grid is cropped or padded to `MAX_MAP_HEIGHT` x `MAX_MAP_WIDTH`.
        2.  Global rule features: A small vector indicating the presence of key rules
            (e.g., if any "X IS YOU" rule is active).

        This flattened vector is then returned.
        """
        grid_representation = np.zeros((self.num_object_channels, MAX_MAP_HEIGHT, MAX_MAP_WIDTH), dtype=np.float32)
        
        player_obj_names: Set[str] = set()
        winnable_obj_names: Set[str] = set()
        pushable_obj_names: Set[str] = set()
        stoppable_obj_names: Set[str] = set()
        kill_obj_names: Set[str] = set()

        for rule_str in state.rules:
            parts = rule_str.split('-is-')
            if len(parts) == 2:
                obj_name, prop_name = parts[0], parts[1]
                if prop_name == "you": player_obj_names.add(obj_name)
                elif prop_name == "win": winnable_obj_names.add(obj_name)
                elif prop_name == "push": pushable_obj_names.add(obj_name)
                elif prop_name == "stop": stoppable_obj_names.add(obj_name)
                elif prop_name == "kill": kill_obj_names.add(obj_name)

        for obj in state.phys:
            if not hasattr(obj, 'name'): continue
            y, x = obj.y, obj.x
            if 0 <= y < MAX_MAP_HEIGHT and 0 <= x < MAX_MAP_WIDTH:
                channel_idx = -1 
                if obj.name in player_obj_names: channel_idx = self.object_name_to_channel_map["player_controlled"]
                elif obj.name in winnable_obj_names: channel_idx = self.object_name_to_channel_map["winnable_condition"]
                elif obj.name in pushable_obj_names: channel_idx = self.object_name_to_channel_map["pushable_physical"]
                elif obj.name in stoppable_obj_names: channel_idx = self.object_name_to_channel_map["stopping_physical"]
                elif obj.name in kill_obj_names: channel_idx = self.object_name_to_channel_map["dangerous_physical"]
                else: channel_idx = self.object_name_to_channel_map["general_physical"]
                grid_representation[channel_idx, y, x] = 1.0
        
        all_text_objects: List[GameObj] = []
        if state.words: all_text_objects.extend(state.words)
        if state.keywords: all_text_objects.extend(state.keywords)

        for obj in all_text_objects:
            if not hasattr(obj, 'name'): continue
            y, x = obj.y, obj.x
            if 0 <= y < MAX_MAP_HEIGHT and 0 <= x < MAX_MAP_WIDTH:
                text_key = f"text_{obj.name}"
                channel_idx = self.object_name_to_channel_map.get(text_key, self.num_object_channels - 1)
                grid_representation[channel_idx, y, x] = 1.0

        flattened_grid = grid_representation.flatten()
        rule_features = np.zeros(NUM_RULE_FEATURES, dtype=np.float32)
        if player_obj_names: rule_features[0] = 1.0
        if winnable_obj_names: rule_features[1] = 1.0
        return np.concatenate((flattened_grid, rule_features))

    def _get_action_from_model(self, vectorized_state: np.array) -> Direction:
        """
        Queries the loaded model to get an action for the given vectorized state.
        
        If a Keras model is loaded, it reshapes the input for batch prediction,
        predicts action probabilities, and returns the action with the highest probability.
        If a dummy model is active, it calls its predict method.

        Args:
            vectorized_state: The numerical representation of the current game state.

        Returns:
            The Direction enum corresponding to the predicted action.
        """
        action_index: int
        if TF_AVAILABLE and isinstance(self.model, keras.Model):
            # Add batch dimension for Keras model: (FEATURE_VECTOR_SIZE,) -> (1, FEATURE_VECTOR_SIZE)
            predictions = self.model.predict(vectorized_state.reshape(1, -1), verbose=0)
            action_index = np.argmax(predictions[0])
        else: # DummyModel or other non-Keras fallback (e.g. if TF failed to import)
            action_index = self.model.predict(vectorized_state) # Dummy model predict can take flat vector

        return INDEX_TO_ACTION[action_index]

    def search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """
        Executes the learned policy (or dummy policy) for a maximum of `iterations` steps.

        At each step, the current game state is vectorized, an action is chosen by
        the model, and that action is applied. The process stops if a win condition is met,
        the player is eliminated, or the iteration (max steps) limit is reached.

        Args:
            initial_state: The starting GameState of the puzzle.
            iterations: The maximum number of actions (steps) the agent is allowed to take.

        Returns:
            A list of Direction enums representing the sequence of actions taken by the agent.
        """
        current_state = initial_state.copy()
        path_taken: List[Direction] = []

        if not current_state.players:
            return []

        for _ in range(iterations): # Use iterations as max steps
            vectorized_state = self._vectorize_state(current_state)
            chosen_action = self._get_action_from_model(vectorized_state)
            
            path_taken.append(chosen_action)
            
            current_state = advance_game_state(chosen_action, current_state.copy())

            if check_win(current_state):
                break 
            
            if not current_state.players: 
                break 
            
        return path_taken

if __name__ == '__main__':
    # This block is for potential standalone testing or utilities.
    # Example: Test vectorization
    # from baba import make_level, parse_map
    # test_map_ascii = ("__________\n"
    #                   "_B12..F13_\n"
    #                   "_........_\n"
    #                   "_.b....f._\n"
    #                   "__________")
    # game_map = parse_map(test_map_ascii)
    # if game_map:
    #     initial_gs = make_level(game_map)
    #     if initial_gs:
    #         # Check if TF is available before trying to instantiate agent that might load a model
    #         # For this test, we can pass a non-existent model path to force dummy model.
    #         agent = BehavioralCloningAgent(model_path="non_existent_model.h5")
    #         print(f"Agent using model type: {type(agent.model)}")
    #         try:
    #             vec_state = agent._vectorize_state(initial_gs)
    #             print(f"Vectorized state shape: {vec_state.shape}")
    #             print(f"Expected size: {FEATURE_VECTOR_SIZE}")
    #             if vec_state.shape[0] == FEATURE_VECTOR_SIZE:
    #                 print("Vectorization test snippet seems okay.")
    #             else:
    #                 print("ERROR: Vectorized state shape mismatch.")
    #         except Exception as e:
    #             print(f"Error during vectorization test: {e}")
    pass
