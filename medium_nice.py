import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import os

# --- Configuration ---
# Set to None for random seed, integer for reproducible results, or string for custom seed text
CUSTOM_SEED = "the quick brown"  # Change this as needed
SEQUENCE_LENGTH = 60  # Longer sequences for complex data
EPOCHS = 200  # Much more training for larger dataset
BATCH_SIZE = 64  # Larger batches for stability with more data
GENERATION_LENGTH = 300  # Longer generation

# Set numpy random seed for reproducibility if integer seed provided
if isinstance(CUSTOM_SEED, int) and CUSTOM_SEED is not None:
    np.random.seed(CUSTOM_SEED)

# --- 1. Data Collection & Preprocessing ---

# Try to load external data file first, fallback to internal corpus
filename = "data.txt"
if os.path.exists(filename):
    print(f"Loading external data from '{filename}'...")
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print("External data loaded successfully.")
else:
    print(f"'{filename}' not found. Using internal text corpus...")
    # Enhanced internal corpus with more diverse content
    text_list = [
        "The quick brown fox jumps over the lazy dog. The lazy dog barks loudly.",
        "Once upon a time, in a land far away, there lived a wise old wizard.",
        "Parth is a good boy, he loves to play cricket with his friends every evening.",
        "Arya loves to play with her toys, especially her colorful dolls and teddy bears.",
        "Laita loves to eat mangos, she eats them every day during summer season.",
        "Amol is in merchant navy, he is a good sailor who travels across oceans.",
        "Kiran lives in the village and is a good farmer, he grows rice and wheat.",
        "Ravi is a good student, he studies hard and gets excellent grades in school.",
        "Sita is a good cook, she makes delicious food for her large family.",
        "Ram is a good friend, he helps others and is always there for them.",
        "The sun rises in the east and sets in the west every single day.",
        "Birds fly high in the sky, singing beautiful songs in the morning.",
        "Children love to play in the park, running and laughing together.",
        "Books contain knowledge and wisdom that can change our lives forever.",
        "Music has the power to heal hearts and bring people together.",
        "Technology is advancing rapidly, changing how we live and work.",
        "Nature is beautiful and we must protect it for future generations.",
        "Education is the key to success and personal growth in life.",
        "Friendship is one of the most valuable treasures in human life.",
        "Dreams give us hope and motivation to achieve our goals.",
        "Hard work and dedication always lead to success in the end.",
        "Love and kindness make the world a better place for everyone.",
        "Science helps us understand the mysteries of the universe around us.",
        "Art and creativity express the deepest emotions of human soul.",
        "Time is precious and we should use it wisely every day."
    ]
    raw_text = ' '.join(text_list)
    print("Internal corpus loaded.")

# Convert to lowercase for consistency
raw_text = raw_text.lower()

# Get all unique characters in the text
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

print(f"Total Characters: {n_chars}")
print(f"Total Unique Characters (Vocabulary Size): {n_vocab}")
print("-" * 50)

# Prepare input sequences and their corresponding next characters
X_data = []
y_data = []

# Create overlapping sequences for better training
for i in range(0, n_chars - SEQUENCE_LENGTH, 1):
    seq_in = raw_text[i:i + SEQUENCE_LENGTH]
    seq_out = raw_text[i + SEQUENCE_LENGTH]
    X_data.append([char_to_int[char] for char in seq_in])
    y_data.append(char_to_int[seq_out])

n_patterns = len(X_data)
print(f"Total Sequences (patterns): {n_patterns}")
print("-" * 50)

# Reshape the data for the LSTM model
# The LSTM layer expects a 3D input: [samples, timesteps, features]
X = np.reshape(X_data, (n_patterns, SEQUENCE_LENGTH, 1))
# Normalize the input data to a range between 0 and 1
X = X / float(n_vocab)

# One-hot encode the output variable
y = to_categorical(y_data)

# --- 2. Define the Model (Best of Both Worlds) ---

print("Building optimized LSTM model...")
model = Sequential()

# First LSTM layer with return_sequences=True for stacking
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))  # Higher dropout for complex data

# Second LSTM layer with return_sequences=True for deeper learning
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))

# Third LSTM layer (final layer, no return_sequences)
model.add(LSTM(128))  # Smaller final layer for efficiency
model.add(Dropout(0.2))

# Output layer with softmax for probability distribution
model.add(Dense(y.shape[1], activation='softmax'))

# Display model architecture
model.summary()
print("-" * 50)

# --- 3. Training the Model ---

print("Compiling model...")
# Use a lower learning rate for better convergence with complex data
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)  # Lower learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(f"Starting model training for {EPOCHS} epochs...")
print("This may take a while depending on your hardware...")

# Add learning rate reduction callback
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

# Train the model with validation split for monitoring
history = model.fit(X, y, 
                   epochs=EPOCHS, 
                   batch_size=BATCH_SIZE, 
                   verbose=1,
                   validation_split=0.1,  # Use 10% for validation
                   callbacks=[reduce_lr])  # Add learning rate scheduler

print("Training complete!")
print("-" * 50)

# --- 4. Advanced Text Generation ---

# Handle different types of seeds (enhanced from simple_nice.py)
if isinstance(CUSTOM_SEED, str) and len(CUSTOM_SEED) >= SEQUENCE_LENGTH:
    # Use the provided string seed if it's long enough
    seed_text = CUSTOM_SEED[-SEQUENCE_LENGTH:].lower()
    # Handle characters not in vocabulary
    pattern = []
    for char in seed_text:
        if char in char_to_int:
            pattern.append(char_to_int[char])
        else:
            # Replace unknown characters with space
            pattern.append(char_to_int.get(' ', 0))
    print(f"Using custom text seed: \"{seed_text}\"")
    
elif isinstance(CUSTOM_SEED, int) and 0 <= CUSTOM_SEED < n_patterns:
    # Use the integer seed to select a specific pattern
    start_index = CUSTOM_SEED
    pattern = X_data[start_index]
    seed_text = ''.join([int_to_char[value] for value in pattern])
    print(f"Using custom seed index: {CUSTOM_SEED}")
    
else:
    # Use a random seed from the training data
    start_index = np.random.randint(0, n_patterns - 1)
    pattern = X_data[start_index]
    seed_text = ''.join([int_to_char[value] for value in pattern])
    
    if CUSTOM_SEED is not None:
        print(f"Custom seed '{CUSTOM_SEED}' is invalid or too short.")
        print(f"Using random seed instead: \"{seed_text}\"")
    else:
        print(f"Using random seed: \"{seed_text}\"")

print("\n" + "="*60)
print("GENERATED TEXT")
print("="*60)
print(f"Seed: \"{seed_text}\"")
print("-" * 60)

generated_text = []

# Generate text with improved sampling
for i in range(GENERATION_LENGTH):
    # Prepare the input sequence for prediction
    x_predict = np.reshape(pattern, (1, len(pattern), 1))
    x_predict = x_predict / float(n_vocab)

    # Predict probabilities for the next character
    prediction = model.predict(x_predict, verbose=0)[0]
    
    # Use temperature-based sampling for more interesting results
    # Temperature = 1.0 means use raw probabilities
    # Temperature < 1.0 makes it more conservative (better for coherent text)
    # Temperature > 1.0 makes it more creative
    temperature = 0.3  # Very conservative for complex data
    prediction = np.log(prediction + 1e-8) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    
    # Sample from the probability distribution instead of always taking argmax
    index = np.random.choice(len(prediction), p=prediction)
    
    result = int_to_char[index]
    generated_text.append(result)

    # Update the pattern for next prediction
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# Display the generated text
output_text = ''.join(generated_text)
print(output_text)
print("-" * 60)
print("="*60)

# --- 5. Save the Model ---

model_filename = "medium_nice_model.h5"
model.save(model_filename)
print(f"\nModel saved as '{model_filename}'")
print("You can load this model later using:")
print("from keras.models import load_model")
print(f"model = load_model('{model_filename}')")

# Save training history for analysis
import pickle
history_filename = "medium_nice_training_history.pkl"
with open(history_filename, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Training history saved as '{history_filename}'")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)