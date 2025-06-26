"""
Offline Training Script for KekeAI Behavioral Cloning Model.

This script handles the data preprocessing, model creation, training, and saving
for the behavioral cloning agent. It reads game levels and their solutions,
generates (state, action) pairs, vectorizes them, and then trains a neural network
model to predict expert actions. This version incorporates regularization to prevent overfitting.
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Set, Optional

# Keras utilities for model creation and data handling
from tensorflow import keras # Explicitly import keras from tf
from tensorflow.keras.utils import to_categorical # For one-hot encoding action labels
from tensorflow.keras.callbacks import EarlyStopping # To prevent overfitting by stopping training early
from tensorflow.keras import regularizers # To apply L2 regularization

# Imports from the KekeAI environment and agent definition
from baba import parse_map, make_level, Direction, advance_game_state, GameState, GameObj

# Import necessary constants from the agent's definition for state vectorization.
# Model creation is now handled within this script.
try:
    from agents.behavioral_cloning_AGENT import (
        FEATURE_VECTOR_SIZE,
        NUM_ACTIONS,
        ACTION_TO_INDEX,
        MAX_MAP_HEIGHT,
        MAX_MAP_WIDTH,
        OBJECT_NAME_TO_CHANNEL,
        NUM_OBJECT_CHANNELS,
        NUM_RULE_FEATURES,
        TF_AVAILABLE # To check if TensorFlow is usable
    )
except ImportError as e:
    print(f"Error importing from agents.behavioral_cloning_AGENT: {e}")
    print("Please ensure that the KekeAI environment is correctly set up and PYTHONPATH includes the project root.")
    TF_AVAILABLE = False


if not TF_AVAILABLE:
    print("WARNING: TensorFlow is not available according to behavioral_cloning_AGENT.py. Training script may not function.")

# --- Helper: Action Mapping (Mirrored from agent for clarity if used standalone) ---
SOLUTION_CHAR_TO_DIRECTION = {
    'U': Direction.Up,
    'D': Direction.Down,
    'L': Direction.Left,
    'R': Direction.Right,
    'S': Direction.Wait
}

# --- Model Definition with Regularization ---
def create_bc_model_regularized(input_size: int = FEATURE_VECTOR_SIZE, num_actions: int = NUM_ACTIONS) -> Optional[keras.Model]:
    """
    Creates and compiles a Keras neural network model for behavioral cloning
    with regularization techniques to combat overfitting.

    The model architecture includes:
    - L2 Regularization: Penalizes large weights to encourage a simpler model.
    - Dropout: Randomly sets a fraction of input units to 0 at each update
                 during training to prevent co-adaptation of neurons.

    Args:
        input_size (int): The size of the input feature vector.
        num_actions (int): The number of possible output actions.

    Returns:
        A compiled Keras model if TensorFlow is available, otherwise None.
    """
    if not TF_AVAILABLE:
        return None

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_size,)),
        
        # Hidden Layer 1 with L2 regularization and Dropout
        keras.layers.Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.001), # L2 regularization
            name='hidden_layer_1'
        ),
        keras.layers.Dropout(0.4, name='dropout_1'), # Dropout layer

        # Hidden Layer 2 with L2 regularization and Dropout
        keras.layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001), # L2 regularization
            name='hidden_layer_2'
        ),
        keras.layers.Dropout(0.4, name='dropout_2'), # Dropout layer

        # Output Layer
        keras.layers.Dense(num_actions, activation='softmax', name='output_layer')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- State Vectorization for Training ---
def _vectorize_state_for_training( # Underscore indicates it's primarily a helper for this script
    state: GameState,
    object_name_to_channel_map: Dict[str, int] = OBJECT_NAME_TO_CHANNEL,
    num_object_channels: int = NUM_OBJECT_CHANNELS,
    max_map_height: int = MAX_MAP_HEIGHT,
    max_map_width: int = MAX_MAP_WIDTH,
    num_rule_features: int = NUM_RULE_FEATURES
) -> Optional[np.array]:
    """
    Converts a given GameState object into a flat numerical vector (NumPy array)
    suitable as input for a neural network. This is an adaptation of the
    agent's _vectorize_state method for direct use in this training script.
    """
    if not object_name_to_channel_map:
        print("ERROR: object_name_to_channel_map is missing for vectorization.")
        return None

    grid_representation = np.zeros((num_object_channels, max_map_height, max_map_width), dtype=np.float32)
    
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
        if 0 <= y < max_map_height and 0 <= x < max_map_width:
            channel_idx = -1
            if obj.name in player_obj_names: channel_idx = object_name_to_channel_map["player_controlled"]
            elif obj.name in winnable_obj_names: channel_idx = object_name_to_channel_map["winnable_condition"]
            elif obj.name in pushable_obj_names: channel_idx = object_name_to_channel_map["pushable_physical"]
            elif obj.name in stoppable_obj_names: channel_idx = object_name_to_channel_map["stopping_physical"]
            elif obj.name in kill_obj_names: channel_idx = object_name_to_channel_map["dangerous_physical"]
            else: channel_idx = object_name_to_channel_map.get("general_physical", -1) 
            
            if channel_idx != -1:
                 grid_representation[channel_idx, y, x] = 1.0
    
    all_text_objects: List[GameObj] = []
    if state.words: all_text_objects.extend(state.words)
    if state.keywords: all_text_objects.extend(state.keywords)


    for obj in all_text_objects:
        if not hasattr(obj, 'name'): continue
        y, x = obj.y, obj.x
        if 0 <= y < max_map_height and 0 <= x < max_map_width:
            text_key = f"text_{obj.name}"
            channel_idx = object_name_to_channel_map.get(text_key, num_object_channels - 1)
            grid_representation[channel_idx, y, x] = 1.0

    flattened_grid = grid_representation.flatten()
    
    rule_features_array = np.zeros(num_rule_features, dtype=np.float32)
    if num_rule_features >= 1 and player_obj_names: rule_features_array[0] = 1.0
    if num_rule_features >= 2 and winnable_obj_names: rule_features_array[1] = 1.0

    return np.concatenate((flattened_grid, rule_features_array))


# --- Data Generation ---
def generate_training_data(levels_filepath: str) -> Tuple[Optional[np.array], Optional[np.array]]:
    """
    Generates training data (X, y) from game levels and their expert solutions.
    """
    vectorized_states: List[np.array] = []
    action_indices: List[int] = []

    if not os.path.exists(levels_filepath):
        print(f"ERROR: Levels file not found at '{levels_filepath}'")
        return None, None

    try:
        with open(levels_filepath, 'r') as f:
            levels_data = json.load(f)

        if isinstance(levels_data, dict) and "levels" in levels_data:
            levels_data = levels_data["levels"]
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{levels_filepath}'")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not read levels file '{levels_filepath}': {e}")
        return None, None

    num_levels_processed = 0
    num_trajectories_generated = 0
    malformed_entries = 0

    for i, level_entry in enumerate(levels_data):
        if not isinstance(level_entry, dict):
            malformed_entries += 1
            continue

        level_name = level_entry.get("name", f"Unknown Level (Entry {i})")
        ascii_map = level_entry.get("ascii")
        solution_str = level_entry.get("solution")

        if not ascii_map or not isinstance(ascii_map, str) or not solution_str or not isinstance(solution_str, str):
            continue

        game_map_objects = parse_map(ascii_map)
        if not game_map_objects: continue
        
        initial_state = make_level(game_map_objects)
        if not initial_state: continue

        current_state = initial_state.copy()
        valid_trajectory = True
        
        for action_char in solution_str:
            action_direction = SOLUTION_CHAR_TO_DIRECTION.get(action_char.upper())
            if action_direction is None:
                valid_trajectory = False
                break

            vectorized_current_state = _vectorize_state_for_training(current_state)
            if vectorized_current_state is None:
                valid_trajectory = False
                break
            
            action_idx = ACTION_TO_INDEX.get(action_direction)
            if action_idx is None:
                valid_trajectory = False
                break

            vectorized_states.append(vectorized_current_state)
            action_indices.append(action_idx)
            num_trajectories_generated += 1

            current_state = advance_game_state(action_direction, current_state.copy())
            
            if not current_state.players:
                break 
        
        if valid_trajectory:
            num_levels_processed +=1
    
    if malformed_entries > 0:
        print(f"WARNING: Encountered {malformed_entries} malformed entries in the levels file.")

    if not vectorized_states or not action_indices:
        print("INFO: No training data generated.")
        return None, None

    print(f"Successfully processed {num_levels_processed} levels, generated {num_trajectories_generated} (state, action) pairs.")

    X_train = np.array(vectorized_states, dtype=np.float32)
    y_train = to_categorical(np.array(action_indices), num_classes=NUM_ACTIONS).astype(np.float32)
    
    if X_train.shape[0] != y_train.shape[0] or (X_train.ndim > 0 and X_train.shape[1] != FEATURE_VECTOR_SIZE) or \
       (y_train.ndim > 0 and y_train.shape[1] != NUM_ACTIONS):
        print(f"ERROR: Shape mismatch in generated data.")
        return None, None

    return X_train, y_train


if __name__ == '__main__':
    print("--- Behavioral Cloning Model Training Script ---")
    
    # --- Step 0: Configuration & Pre-checks ---
    LEVELS_FILEPATH = "json_levels/train_LEVELS.json"
    MODEL_SAVE_PATH = "keke_behavioral_cloning_model.h5"
    EPOCHS = 100  # Increased epochs since EarlyStopping will find the best one
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2

    if not TF_AVAILABLE:
        print("CRITICAL ERROR: TensorFlow is not available. Cannot proceed with model training.")
        exit(1)
    
    if not os.path.exists(LEVELS_FILEPATH):
        # The dummy file creation logic is fine as is.
        print(f"WARNING: Levels file not found at '{LEVELS_FILEPATH}'.")
        print("INFO: Attempting to create a dummy file for demonstration...")
        try:
            dummy_file_dir = os.path.dirname(LEVELS_FILEPATH)
            if dummy_file_dir and not os.path.exists(dummy_file_dir): os.makedirs(dummy_file_dir)
            dummy_level_content = [{"name": "Dummy Level 1", "ascii": "B.F\n.Y.\nbiy", "solution": "R"}, {"name": "Dummy Level 2", "ascii": "B.R.F\n.Y.P.\nbiy\nrip\nfiw", "solution": "RR"}]
            with open(LEVELS_FILEPATH, 'w') as f_dummy: json.dump(dummy_level_content, f_dummy, indent=4)
            print(f"SUCCESS: Created dummy '{LEVELS_FILEPATH}'.")
        except Exception as e_dummy:
            print(f"ERROR: Failed to create dummy levels file: {e_dummy}")
            exit(1)

    # --- Step 1: Data Generation ---
    print(f"\n[Phase 1/3] Generating training data from '{LEVELS_FILEPATH}'...")
    X_train, y_train = generate_training_data(LEVELS_FILEPATH)

    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("ERROR: Failed to generate valid training data. Exiting.")
        exit(1)

    print(f"SUCCESS: Generated {X_train.shape[0]} training samples.")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")

    # --- Step 2: Model Creation ---
    print("\n[Phase 2/3] Creating the regularized Behavioral Cloning neural network model...")
    model = create_bc_model_regularized(input_size=FEATURE_VECTOR_SIZE, num_actions=NUM_ACTIONS)

    if model is None:
        print("CRITICAL ERROR: Failed to create the Keras model. Exiting.")
        exit(1)
    
    model.summary()
    print("SUCCESS: Model created.")

    # --- Step 3: Model Training with Early Stopping ---
    print(f"\n[Phase 3/3] Starting model training for up to {EPOCHS} epochs...")
    print(f"   Validation split: {VALIDATION_SPLIT*100}%")
    
    # Define the EarlyStopping callback
    # It will monitor the validation loss and stop if it doesn't improve for 'patience' epochs.
    # 'restore_best_weights=True' ensures the model has the weights from the best epoch, not the last.
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped.
        verbose=1,
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

    try:
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            verbose=1,
            #callbacks=[early_stopping_callback] # Add the callback here
        )
        print("\nSUCCESS: Model training completed.")

        # Display final metrics (from the best epoch thanks to restore_best_weights)
        if history and history.history:
            # Find the best epoch index
            best_epoch_idx = np.argmin(history.history['val_loss'])
            final_loss = history.history['loss'][best_epoch_idx]
            final_accuracy = history.history['accuracy'][best_epoch_idx]
            final_val_loss = history.history['val_loss'][best_epoch_idx]
            final_val_accuracy = history.history['val_accuracy'][best_epoch_idx]
            print(f"   Best epoch: {best_epoch_idx + 1}/{len(history.history['loss'])}")
            print(f"   Best Training Loss: {final_loss:.4f}")
            print(f"   Best Training Accuracy: {final_accuracy:.4f}")
            print(f"   Best Validation Loss: {final_val_loss:.4f}")
            print(f"   Best Validation Accuracy: {final_val_accuracy:.4f}")

        # --- Step 4: Model Saving ---
        print(f"\n[Phase 4/4] Saving the trained model to '{MODEL_SAVE_PATH}'...")
        model.save(MODEL_SAVE_PATH)
        print(f"SUCCESS: Trained model saved to '{MODEL_SAVE_PATH}'.")
        print("\n--- Behavioral Cloning Training Script Finished Successfully! ---")

    except Exception as e_train:
        print(f"\nCRITICAL ERROR: An unexpected error occurred during model training or saving: {e_train}")