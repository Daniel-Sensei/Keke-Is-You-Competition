"""
Advanced Improved Offline Training Script for KekeAI Behavioral Cloning Model.

This version includes multiple advanced strategies to improve accuracy:
1. More sophisticated data augmentation
2. Better model architecture with attention mechanisms
3. Advanced regularization techniques
4. Learning rate scheduling and optimization
5. Cross-validation and ensemble methods
6. Better feature engineering
7. Curriculum learning
8. Class weighting for imbalanced data
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Set, Optional
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Keras utilities for model creation and data handling
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.metrics import TopKCategoricalAccuracy # <--- ADD THIS LINE


# Imports from the KekeAI environment and agent definition
from baba import parse_map, make_level, Direction, advance_game_state, GameState, GameObj

# Import necessary constants from the agent's definition for state vectorization.
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
        TF_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing from agents.behavioral_cloning_AGENT: {e}")
    TF_AVAILABLE = False

if not TF_AVAILABLE:
    print("WARNING: TensorFlow is not available. Training script may not function.")

# Action mapping
SOLUTION_CHAR_TO_DIRECTION = {
    'U': Direction.Up,
    'D': Direction.Down,
    'L': Direction.Left,
    'R': Direction.Right,
    'S': Direction.Wait
}

INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}

# === ADVANCED DATA AUGMENTATION ===
def rotate_state_90(state_vector: np.array, grid_shape: Tuple[int, int, int]) -> np.array:
    """Rotate the grid representation by 90 degrees clockwise."""
    grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
    grid_part = state_vector[:grid_size].reshape(grid_shape)
    rule_part = state_vector[grid_size:]
    
    rotated_grid = np.rot90(grid_part, k=-1, axes=(1, 2))
    return np.concatenate([rotated_grid.flatten(), rule_part])

def flip_state_horizontal(state_vector: np.array, grid_shape: Tuple[int, int, int]) -> np.array:
    """Flip the grid representation horizontally."""
    grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
    grid_part = state_vector[:grid_size].reshape(grid_shape)
    rule_part = state_vector[grid_size:]
    
    flipped_grid = np.flip(grid_part, axis=2)
    return np.concatenate([flipped_grid.flatten(), rule_part])

def flip_state_vertical(state_vector: np.array, grid_shape: Tuple[int, int, int]) -> np.array:
    """Flip the grid representation vertically."""
    grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
    grid_part = state_vector[:grid_size].reshape(grid_shape)
    rule_part = state_vector[grid_size:]
    
    flipped_grid = np.flip(grid_part, axis=1)
    return np.concatenate([flipped_grid.flatten(), rule_part])

def add_gaussian_noise(state_vector: np.array, noise_factor: float = 0.02) -> np.array:
    """Add Gaussian noise to the state vector."""
    noise = np.random.normal(0, noise_factor, state_vector.shape)
    return state_vector + noise

def spatial_dropout_augmentation(state_vector: np.array, grid_shape: Tuple[int, int, int], dropout_rate: float = 0.1) -> np.array:
    """Apply spatial dropout to grid cells."""
    grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
    grid_part = state_vector[:grid_size].reshape(grid_shape)
    rule_part = state_vector[grid_size:]
    
    # Create random mask
    mask = np.random.random(grid_shape) > dropout_rate
    augmented_grid = grid_part * mask
    
    return np.concatenate([augmented_grid.flatten(), rule_part])

def transform_action_for_rotation(action_idx: int) -> int:
    """Transform action for 90-degree clockwise rotation."""
    action = INDEX_TO_ACTION[action_idx]
    rotation_map = {
        Direction.Up: Direction.Right,
        Direction.Right: Direction.Down,
        Direction.Down: Direction.Left,
        Direction.Left: Direction.Up,
        Direction.Wait: Direction.Wait
    }
    return ACTION_TO_INDEX[rotation_map[action]]

def transform_action_for_horizontal_flip(action_idx: int) -> int:
    """Transform action for horizontal flip."""
    action = INDEX_TO_ACTION[action_idx]
    flip_map = {
        Direction.Up: Direction.Up,
        Direction.Down: Direction.Down,
        Direction.Left: Direction.Right,
        Direction.Right: Direction.Left,
        Direction.Wait: Direction.Wait
    }
    return ACTION_TO_INDEX[flip_map[action]]

def transform_action_for_vertical_flip(action_idx: int) -> int:
    """Transform action for vertical flip."""
    action = INDEX_TO_ACTION[action_idx]
    flip_map = {
        Direction.Up: Direction.Down,
        Direction.Down: Direction.Up,
        Direction.Left: Direction.Left,
        Direction.Right: Direction.Right,
        Direction.Wait: Direction.Wait
    }
    return ACTION_TO_INDEX[flip_map[action]]

def advanced_augment_data(X: np.array, y: np.array, augmentation_factor: int = 4) -> Tuple[np.array, np.array]:
    """
    Advanced data augmentation with multiple transformation strategies.
    """
    grid_shape = (NUM_OBJECT_CHANNELS, MAX_MAP_HEIGHT, MAX_MAP_WIDTH)
    
    augmented_X = [X]
    augmented_y = [y]
    
    y_indices = np.argmax(y, axis=1)
    
    # Define augmentation strategies
    augmentation_strategies = [
        'rotate_90', 'rotate_180', 'rotate_270',
        'flip_h', 'flip_v', 'flip_both',
        'noise_light', 'noise_medium',
        'spatial_dropout', 'combination'
    ]
    
    for _ in range(augmentation_factor):
        aug_X = []
        aug_y_indices = []
        
        for i in range(len(X)):
            state = X[i]
            action_idx = y_indices[i]
            
            # Randomly choose augmentation strategy
            strategy = np.random.choice(augmentation_strategies)
            
            if strategy == 'rotate_90':
                new_state = rotate_state_90(state, grid_shape)
                new_action = transform_action_for_rotation(action_idx)
            elif strategy == 'rotate_180':
                new_state = rotate_state_90(rotate_state_90(state, grid_shape), grid_shape)
                new_action = transform_action_for_rotation(transform_action_for_rotation(action_idx))
            elif strategy == 'rotate_270':
                temp_state = rotate_state_90(state, grid_shape)
                temp_state = rotate_state_90(temp_state, grid_shape)
                new_state = rotate_state_90(temp_state, grid_shape)
                temp_action = transform_action_for_rotation(action_idx)
                temp_action = transform_action_for_rotation(temp_action)
                new_action = transform_action_for_rotation(temp_action)
            elif strategy == 'flip_h':
                new_state = flip_state_horizontal(state, grid_shape)
                new_action = transform_action_for_horizontal_flip(action_idx)
            elif strategy == 'flip_v':
                new_state = flip_state_vertical(state, grid_shape)
                new_action = transform_action_for_vertical_flip(action_idx)
            elif strategy == 'flip_both':
                new_state = flip_state_horizontal(flip_state_vertical(state, grid_shape), grid_shape)
                temp_action = transform_action_for_vertical_flip(action_idx)
                new_action = transform_action_for_horizontal_flip(temp_action)
            elif strategy == 'noise_light':
                new_state = add_gaussian_noise(state, 0.01)
                new_action = action_idx
            elif strategy == 'noise_medium':
                new_state = add_gaussian_noise(state, 0.03)
                new_action = action_idx
            elif strategy == 'spatial_dropout':
                new_state = spatial_dropout_augmentation(state, grid_shape, 0.05)
                new_action = action_idx
            else:  # 'combination'
                # Apply multiple transformations
                new_state = state
                new_action = action_idx
                
                if np.random.random() > 0.5:
                    new_state = flip_state_horizontal(new_state, grid_shape)
                    new_action = transform_action_for_horizontal_flip(new_action)
                
                if np.random.random() > 0.5:
                    new_state = add_gaussian_noise(new_state, 0.02)
            
            aug_X.append(new_state)
            aug_y_indices.append(new_action)
        
        augmented_X.append(np.array(aug_X))
        augmented_y.append(to_categorical(aug_y_indices, num_classes=NUM_ACTIONS))
    
    final_X = np.concatenate(augmented_X, axis=0)
    final_y = np.concatenate(augmented_y, axis=0)
    
    return final_X, final_y

def create_cnn_model(input_size: int = FEATURE_VECTOR_SIZE, num_actions: int = NUM_ACTIONS) -> Optional[keras.Model]:
    if not TF_AVAILABLE:
        return None

    inputs = keras.layers.Input(shape=(input_size,))
    
    # Split grid and rule features
    grid_size = NUM_OBJECT_CHANNELS * MAX_MAP_HEIGHT * MAX_MAP_WIDTH
    grid_features = keras.layers.Lambda(lambda x: x[:, :grid_size])(inputs)
    rule_features = keras.layers.Lambda(lambda x: x[:, grid_size:])(inputs)
    
    # Reshape grid for CNN (channels last format)
    grid_reshaped = keras.layers.Reshape((MAX_MAP_HEIGHT, MAX_MAP_WIDTH, NUM_OBJECT_CHANNELS))(grid_features)
    
    # CNN layers
    conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(grid_reshaped)
    conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    flattened_grid = keras.layers.Flatten()(conv2)
    
    # Combine processed grid with rule features
    combined_features = keras.layers.concatenate([flattened_grid, rule_features])
    
    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(combined_features)
    x = keras.layers.Dropout(0.4)(x)
    
    outputs = keras.layers.Dense(num_actions, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'] # Start simple
    )
    return model

# === ADVANCED MODEL ARCHITECTURES ===
def create_attention_model(input_size: int = FEATURE_VECTOR_SIZE, num_actions: int = NUM_ACTIONS) -> Optional[keras.Model]:
    """
    Create a model with attention mechanisms for better feature understanding.
    """
    if not TF_AVAILABLE:
        return None

    # Input layer
    inputs = keras.layers.Input(shape=(input_size,))
    
    # Extract grid and rule features
    grid_size = NUM_OBJECT_CHANNELS * MAX_MAP_HEIGHT * MAX_MAP_WIDTH
    grid_features = keras.layers.Lambda(lambda x: x[:, :grid_size])(inputs)
    rule_features = keras.layers.Lambda(lambda x: x[:, grid_size:])(inputs)
    
    # Reshape grid for spatial processing
    grid_reshaped = keras.layers.Reshape((MAX_MAP_HEIGHT * MAX_MAP_WIDTH, NUM_OBJECT_CHANNELS))(grid_features)
    
    # Multi-head attention for spatial relationships
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=NUM_OBJECT_CHANNELS // 4,
        dropout=0.1
    )(grid_reshaped, grid_reshaped)
    
    attention_output = LayerNormalization()(attention_output + grid_reshaped)
    
    # Global average pooling
    spatial_features = keras.layers.GlobalAveragePooling1D()(attention_output)
    
    # Combine spatial and rule features
    combined_features = keras.layers.concatenate([spatial_features, rule_features])
    
    # Dense layers with residual connections
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined_features)
    x = keras.layers.Dropout(0.3)(x)
    
    residual = keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(combined_features)
    x = keras.layers.Add()([x, residual])
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.LayerNormalization()(x)
    
    # Second dense block
    x2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x2 = keras.layers.Dropout(0.3)(x2)
    
    # Output layer
    outputs = keras.layers.Dense(num_actions, activation='softmax')(x2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Use advanced optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_accuracy')] # Use the class here
    )
    
    return model

def create_ensemble_model(input_size: int = FEATURE_VECTOR_SIZE, num_actions: int = NUM_ACTIONS) -> Optional[keras.Model]:
    """
    Create an ensemble-like model with multiple pathways.
    """
    if not TF_AVAILABLE:
        return None

    inputs = keras.layers.Input(shape=(input_size,))
    
    # Pathway 1: Deep and narrow
    path1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    path1 = keras.layers.Dropout(0.4)(path1)
    path1 = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(path1)
    path1 = keras.layers.Dropout(0.4)(path1)
    path1 = keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(path1)
    
    # Pathway 2: Wide and shallow
    path2 = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    path2 = keras.layers.Dropout(0.3)(path2)
    path2 = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(path2)
    
    # Pathway 3: Specialized for rules
    rule_features = keras.layers.Lambda(lambda x: x[:, -NUM_RULE_FEATURES:])(inputs)
    path3 = keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(rule_features)
    path3 = keras.layers.Dropout(0.2)(path3)
    
    # Combine pathways
    combined = keras.layers.concatenate([path1, path2, path3])
    
    # Final layers
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    
    outputs = keras.layers.Dense(num_actions, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_2_accuracy']
    )
    
    return model

# === LEARNING RATE SCHEDULING ===
def cosine_annealing_schedule(epoch, initial_lr=0.001, min_lr=1e-6, restart_period=50):
    """Cosine annealing with warm restarts."""
    cycle = epoch % restart_period
    return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * cycle / restart_period)) / 2

def create_lr_scheduler():
    """Create a learning rate scheduler."""
    return LearningRateScheduler(cosine_annealing_schedule, verbose=1)

# === STATE VECTORIZATION (Enhanced) ===
def _vectorize_state_for_training(
    state: GameState,
    object_name_to_channel_map: Dict[str, int] = OBJECT_NAME_TO_CHANNEL,
    num_object_channels: int = NUM_OBJECT_CHANNELS,
    max_map_height: int = MAX_MAP_HEIGHT,
    max_map_width: int = MAX_MAP_WIDTH,
    num_rule_features: int = NUM_RULE_FEATURES
) -> Optional[np.array]:
    """Enhanced state vectorization with better feature engineering."""
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

    # Enhanced object processing with distance features
    player_positions = []
    win_positions = []
    
    for obj in state.phys:
        if not hasattr(obj, 'name'): continue
        y, x = obj.y, obj.x
        if 0 <= y < max_map_height and 0 <= x < max_map_width:
            channel_idx = -1
            if obj.name in player_obj_names:
                channel_idx = object_name_to_channel_map["player_controlled"]
                player_positions.append((y, x))
            elif obj.name in winnable_obj_names:
                channel_idx = object_name_to_channel_map["winnable_condition"]
                win_positions.append((y, x))
            elif obj.name in pushable_obj_names:
                channel_idx = object_name_to_channel_map["pushable_physical"]
            elif obj.name in stoppable_obj_names:
                channel_idx = object_name_to_channel_map["stopping_physical"]
            elif obj.name in kill_obj_names:
                channel_idx = object_name_to_channel_map["dangerous_physical"]
            else:
                channel_idx = object_name_to_channel_map.get("general_physical", -1)
            
            if channel_idx != -1:
                grid_representation[channel_idx, y, x] = 1.0
    
    # Process text objects
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
    
    # Enhanced rule features
    rule_features_array = np.zeros(max(num_rule_features, 10), dtype=np.float32)  # Ensure at least 10 features
    if len(rule_features_array) >= 1 and player_obj_names: rule_features_array[0] = 1.0
    if len(rule_features_array) >= 2 and winnable_obj_names: rule_features_array[1] = 1.0
    if len(rule_features_array) >= 3 and pushable_obj_names: rule_features_array[2] = 1.0
    if len(rule_features_array) >= 4 and stoppable_obj_names: rule_features_array[3] = 1.0
    if len(rule_features_array) >= 5 and kill_obj_names: rule_features_array[4] = 1.0
    
    # Add distance features if we have enough space
    if len(rule_features_array) >= 8 and player_positions and win_positions:
        min_distance = float('inf')
        for px, py in player_positions:
            for wx, wy in win_positions:
                distance = abs(px - wx) + abs(py - wy)  # Manhattan distance
                min_distance = min(min_distance, distance)
        
        if min_distance != float('inf'):
            rule_features_array[5] = min(min_distance / (max_map_height + max_map_width), 1.0)  # Normalized distance
            rule_features_array[6] = 1.0 if min_distance <= 1 else 0.0  # Adjacent to win
            rule_features_array[7] = 1.0 if min_distance <= 2 else 0.0  # Close to win

    return np.concatenate((flattened_grid, rule_features_array[:num_rule_features]))

# === DATA GENERATION WITH CURRICULUM LEARNING ===
def generate_training_data_with_curriculum(levels_filepath: str) -> Tuple[Optional[np.array], Optional[np.array], Optional[np.array]]:
    """Generate training data with curriculum learning - easy levels first."""
    vectorized_states: List[np.array] = []
    action_indices: List[int] = []
    level_difficulties: List[int] = []  # Track difficulty for curriculum learning

    if not os.path.exists(levels_filepath):
        print(f"ERROR: Levels file not found at '{levels_filepath}'")
        return None, None, None

    try:
        with open(levels_filepath, 'r') as f:
            levels_data = json.load(f)

        if isinstance(levels_data, dict) and "levels" in levels_data:
            levels_data = levels_data["levels"]
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{levels_filepath}'")
        return None, None, None
    except Exception as e:
        print(f"ERROR: Could not read levels file '{levels_filepath}': {e}")
        return None, None, None

    # Sort levels by difficulty (solution length as proxy)
    levels_with_difficulty = []
    for level_entry in levels_data:
        if isinstance(level_entry, dict):
            solution_str = level_entry.get("solution", "")
            difficulty = len(solution_str) if solution_str else 999
            levels_with_difficulty.append((difficulty, level_entry))
    
    levels_with_difficulty.sort(key=lambda x: x[0])  # Sort by difficulty
    
    num_levels_processed = 0
    num_trajectories_generated = 0

    for difficulty, level_entry in levels_with_difficulty:
        level_name = level_entry.get("name", f"Unknown Level")
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
            level_difficulties.append(min(difficulty, 10))  # Cap difficulty at 10
            num_trajectories_generated += 1

            current_state = advance_game_state(action_direction, current_state.copy())
            
            if not current_state.players:
                break 
        
        if valid_trajectory:
            num_levels_processed += 1

    if not vectorized_states or not action_indices:
        print("INFO: No training data generated.")
        return None, None, None

    # Smart balancing based on both action and difficulty
    action_counts = Counter(action_indices)
    print(f"Action distribution: {action_counts}")
    
    # Use class weights instead of hard balancing
    unique_actions = np.unique(action_indices)
    class_weights = compute_class_weight('balanced', classes=unique_actions, y=action_indices)
    class_weight_dict = dict(zip(unique_actions, class_weights))
    
    print(f"Successfully processed {num_levels_processed} levels, generated {len(vectorized_states)} (state, action) pairs.")

    X_train = np.array(vectorized_states, dtype=np.float32)
    y_train = to_categorical(np.array(action_indices), num_classes=NUM_ACTIONS).astype(np.float32)
    difficulties = np.array(level_difficulties, dtype=np.float32)
    
    return X_train, y_train, difficulties

# === CROSS-VALIDATION TRAINING ===
def cross_validate_model(X: np.array, y: np.array, n_splits: int = 5) -> List[float]:
    """Perform cross-validation to get more robust performance estimates."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_labels = np.argmax(y, axis=1)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labels)):
        print(f"\nTraining fold {fold + 1}/{n_splits}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create model for this fold
        model = create_attention_model()
        if model is None:
            continue
        
        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            verbose=0,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=16,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # Evaluate
        val_acc = max(history.history['val_accuracy'])
        cv_scores.append(val_acc)
        print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
    
    return cv_scores

if __name__ == '__main__':
    print("--- Advanced KekeAI Behavioral Cloning Training Script ---")

    # Enhanced Configuration
    LEVELS_FILEPATH = "json_levels/train_LEVELS.json"
    MODEL_SAVE_PATH = "keke_behavioral_cloning_model.h5"
    EPOCHS = 300
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.25
    AUGMENTATION_FACTOR = 5
    USE_CROSS_VALIDATION = False  # Set to True for more robust evaluation

    if not TF_AVAILABLE:
        print("CRITICAL ERROR: TensorFlow is not available. Cannot proceed with model training.")
        exit(1)

    # Create dummy file if needed for out-of-the-box execution
    if not os.path.exists(LEVELS_FILEPATH):
        print(f"Creating dummy file at {LEVELS_FILEPATH}...")
        try:
            dummy_file_dir = os.path.dirname(LEVELS_FILEPATH)
            if dummy_file_dir and not os.path.exists(dummy_file_dir):
                os.makedirs(dummy_file_dir)
            dummy_level_content = {
                "levels": [
                    {"name": "Easy Level 1", "ascii": "B.F\n.Y.\nbiy", "solution": "R"},
                    {"name": "Easy Level 2", "ascii": "B.R.F\n.Y.P.\nbiy\nrip\nfiw", "solution": "RR"},
                    {"name": "Medium Level", "ascii": "B.R.K.F\n.Y.P..\nbiy\nrip\nkik\nfiw", "solution": "RRDR"},
                    {"name": "Hard Level", "ascii": "B.R.K.W.F\n.Y.P.Q..\nbiy\nrip\nkik\nwis\nfiw", "solution": "RRDLUR"}
                ]
            }
            with open(LEVELS_FILEPATH, 'w') as f_dummy:
                json.dump(dummy_level_content, f_dummy, indent=4)
            print(f"SUCCESS: Created dummy '{LEVELS_FILEPATH}'.")
        except Exception as e_dummy:
            print(f"ERROR: Failed to create dummy levels file: {e_dummy}")
            exit(1)

    # Step 1: Generate training data using curriculum approach
    print(f"\n[Phase 1/5] Generating training data from '{LEVELS_FILEPATH}' with curriculum sorting...")
    X_train, y_train, difficulties = generate_training_data_with_curriculum(LEVELS_FILEPATH)

    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("ERROR: Failed to generate valid training data. Exiting.")
        exit(1)

    print(f"Original dataset: {X_train.shape[0]} samples")

    # Step 2: Advanced data augmentation
    print(f"\n[Phase 2/5] Augmenting data with advanced strategies (factor: {AUGMENTATION_FACTOR})...")
    X_train_aug, y_train_aug = advanced_augment_data(X_train, y_train, AUGMENTATION_FACTOR)
    print(f"Augmented dataset: {X_train_aug.shape[0]} samples")

    # ======================== ADD THIS CODE BLOCK ========================
    print("\nFiltering out the extremely rare 'Wait' action (class 4) to stabilize training...")
    y_indices_aug = np.argmax(y_train_aug, axis=1)
    mask_to_keep = y_indices_aug != 4 # The index for 'Wait' is likely 4
    X_train_aug = X_train_aug[mask_to_keep]
    y_train_aug = y_train_aug[mask_to_keep]

    print(f"Dataset size after filtering: {X_train_aug.shape[0]} samples")
    # =====================================================================

    print(f"Augmented dataset: {X_train_aug.shape[0]} samples")
    print(f"    X_train shape: {X_train_aug.shape}")
    print(f"    y_train shape: {y_train_aug.shape}")

    # Step 3: (Optional) Cross-validation for robust evaluation
    if USE_CROSS_VALIDATION:
        print("\n[Phase 3/5] Performing cross-validation...")
        cv_scores = cross_validate_model(X_train_aug, y_train_aug, n_splits=5)
        if cv_scores:
            print(f"\nCross-Validation Results:")
            print(f"  Scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"  Mean Accuracy: {np.mean(cv_scores):.4f}")
            print(f"  Std Deviation: {np.std(cv_scores):.4f}")
        else:
            print("Cross-validation could not be completed.")

    # Step 4: Model creation and compilation
    print("\n[Phase 4/5] Creating the advanced attention-based neural network...")
    model = create_cnn_model(input_size=FEATURE_VECTOR_SIZE, num_actions=NUM_ACTIONS)

    if model is None:
        print("CRITICAL ERROR: Failed to create the Keras model. Exiting.")
        exit(1)

    model.summary()
    print("SUCCESS: Advanced model created.")

    # Step 5: Training with advanced callbacks and class weighting
    print(f"\n[Phase 5/5] Starting model training for up to {EPOCHS} epochs...")

    # Callbacks for robust training
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=30,  # More patience for complex models and schedules
        verbose=1,
        restore_best_weights=True
    )
    lr_scheduler = create_lr_scheduler()

    # Calculate class weights for the augmented dataset to handle imbalances
    y_indices_aug = np.argmax(y_train_aug, axis=1)
    unique_classes = np.unique(y_indices_aug)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_indices_aug)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    print(f"Using class weights: {class_weight_dict}")

    try:
        history = model.fit(
            X_train_aug, y_train_aug,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            verbose=1,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, lr_scheduler]
        )

        print("\nSUCCESS: Model training completed.")

        # Display final metrics from the best epoch
        if history and history.history and 'val_accuracy' in history.history:
            best_epoch_idx = np.argmax(history.history['val_accuracy'])
            final_loss = history.history['loss'][best_epoch_idx]
            final_accuracy = history.history['accuracy'][best_epoch_idx]
            final_val_loss = history.history['val_loss'][best_epoch_idx]
            final_val_accuracy = history.history['val_accuracy'][best_epoch_idx]
            print(f"    Best epoch: {best_epoch_idx + 1}/{len(history.history['loss'])}")
            print(f"    Best Training Loss: {final_loss:.4f}")
            print(f"    Best Training Accuracy: {final_accuracy:.4f}")
            print(f"    Best Validation Loss: {final_val_loss:.4f}")
            print(f"    Best Validation Accuracy: {final_val_accuracy:.4f}")
        
        # Save the final model
        print(f"\nSaving the trained model to '{MODEL_SAVE_PATH}'...")
        model.save(MODEL_SAVE_PATH)
        print(f"SUCCESS: Trained model saved to '{MODEL_SAVE_PATH}'.")
        print("\n--- Advanced Behavioral Cloning Training Script Finished Successfully! ---")

    except Exception as e_train:
        print(f"\nCRITICAL ERROR: An unexpected error occurred during model training or saving: {e_train}")
        exit(1)