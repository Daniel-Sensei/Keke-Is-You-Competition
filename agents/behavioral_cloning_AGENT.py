"""
Behavioral Cloning Agent for KekeAI.

This agent is designed to use a pre-trained model (e.g., a neural network)
to decide which action to take based on the current game state. The model
itself is trained offline using supervised learning.
"""
import random
import numpy as np
from typing import List, Tuple, Optional, Any, Set
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj

# --- State Vectorization Parameters ---
MAX_MAP_HEIGHT = 25
MAX_MAP_WIDTH = 25

OBJECT_NAME_TO_CHANNEL = {
    "player_controlled": 0, "winnable_condition": 1, "pushable_physical": 2, 
    "stopping_physical": 3, "dangerous_physical": 4, "general_physical": 5,
    "text_baba": 6, "text_flag": 7, "text_rock": 8, "text_wall": 9,
    "text_skull": 10, "text_keke": 11, "text_lava": 12, "text_goop": 13,
    "text_is": 14, "text_you": 15, "text_win": 16, "text_push": 17,
    "text_stop": 18, "text_kill": 19, "text_move": 20, "text_hot": 21,
    "text_melt": 22, "text_sink": 23,
}
NUM_OBJECT_CHANNELS = len(OBJECT_NAME_TO_CHANNEL) + 1
NUM_RULE_FEATURES = 2
FEATURE_VECTOR_SIZE = (NUM_OBJECT_CHANNELS * MAX_MAP_HEIGHT * MAX_MAP_WIDTH) + NUM_RULE_FEATURES

# Mapping actions to integer indices for model output layer, and back.
ACTION_TO_INDEX = {Direction.Up: 0, Direction.Down: 1, Direction.Left: 2, Direction.Right: 3, Direction.Wait: 4}
INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}
NUM_ACTIONS = len(ACTION_TO_INDEX)


# The create_bc_model function has been moved to train_behavioral_cloning.py
# as its primary role is for training, not for agent inference.

class BEHAVIORAL_CLONINGAgent(BaseAgent):
    """
    Agent that uses a pre-trained model for Behavioral Cloning.
    It vectorizes the game state and queries the loaded model for the next action.
    """

    def __init__(self, model_path="keke_behavioral_cloning_model.h5"):
        """
        Initializes the agent by loading a pre-trained Keras model.
        If loading fails or TF is unavailable, it falls back to a dummy model.
        """
        super().__init__()
        self.model: Any = None
        self.object_name_to_channel_map = OBJECT_NAME_TO_CHANNEL
        self.num_object_channels = NUM_OBJECT_CHANNELS
        self._load_model(model_path)

    def _create_dummy_model(self):
        """Creates a dummy model that predicts random actions."""
        class DummyModel:
            def predict(self, vectorized_state: np.array) -> int:
                return random.randint(0, NUM_ACTIONS - 1)
        return DummyModel()

    def _load_model(self, model_path: str):
        """
        Attempts to load a trained Keras model from `model_path`.
        If TF is not available, the model file doesn't exist, or loading fails,
        it falls back to a DummyModel.
        """
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                
                # Opzionale: Costruisci le metriche per evitare il warning
                # Fai una predizione dummy per inizializzare le metriche
                if hasattr(self.model, 'built') and self.model.built:
                    dummy_input = np.zeros((1, FEATURE_VECTOR_SIZE), dtype=np.float32)
                    _ = self.model.predict(dummy_input, verbose=0)
                
                print(f"INFO: Successfully loaded trained Keras model from '{model_path}'")
                return
            except Exception as e:
                print(f"ERROR: Failed to load Keras model from '{model_path}'. Error: {e}")
        
        # Fallback to DummyModel if any of the above conditions fail.
        if not self.model:
            if not TF_AVAILABLE:
                print("INFO: TensorFlow not available.")
            elif not os.path.exists(model_path):
                print(f"WARNING: Model file not found at '{model_path}'.")
            
            print("INFO: BehavioralCloningAgent is falling back to a DUMMY model (predicts random actions).")
            self.model = self._create_dummy_model()

    def _vectorize_state(self, state: GameState) -> np.array:
        """Converts a given GameState object into a flat numerical vector."""
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
        """Queries the model to get an action for the given vectorized state."""
        action_index: int
        if TF_AVAILABLE and isinstance(self.model, keras.Model):
            predictions = self.model.predict(vectorized_state.reshape(1, -1), verbose=0)
            action_index = np.argmax(predictions[0])
        else:
            action_index = self.model.predict(vectorized_state)

        return INDEX_TO_ACTION[action_index]

    def search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """Executes the learned policy for a maximum of `iterations` steps."""
        current_state = initial_state.copy()
        path_taken: List[Direction] = []

        if not current_state.players:
            return []

        for _ in range(iterations):
            vectorized_state = self._vectorize_state(current_state)
            chosen_action = self._get_action_from_model(vectorized_state)
            
            path_taken.append(chosen_action)
            current_state = advance_game_state(chosen_action, current_state.copy())

            if check_win(current_state) or not current_state.players:
                break
            
        return path_taken

if __name__ == '__main__':
    # This block remains for potential standalone testing.
    pass