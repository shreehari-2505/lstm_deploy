import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import json

# Load data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define hyperparameter grid (LIMITED for speed)
param_grid = {
    'lstm_units': [50, 100],           # Test 2 options
    'dropout_rate': [0.2, 0.3],        # Test 2 options
    'batch_size': [32, 64],            # Test 2 options
}

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"🔥 Testing {len(combinations)} hyperparameter combinations...\n")

best_loss = float('inf')
best_params = None
results = []

# Test each combination
for idx, params in enumerate(combinations, 1):
    print(f"[{idx}/{len(combinations)}] Testing: {params}")
    
    # Build model with current params
    model = Sequential()
    model.add(LSTM(units=params['lstm_units'], return_sequences=True, 
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=params['lstm_units'], return_sequences=False))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train with early stopping (max 15 epochs for speed)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        batch_size=params['batch_size'], 
                        epochs=15,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=0)
    
    # Evaluate
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    results.append({**params, 'test_loss': test_loss})
    
    print(f"   → Test Loss: {test_loss:.4f}")
    
    # Track best
    if test_loss < best_loss:
        best_loss = test_loss
        best_params = params
        model.save('lstm_model_tuned.h5')
        print(f"   ✅ NEW BEST MODEL SAVED!\n")
    else:
        print()

# Summary
print("="*60)
print("🏆 TUNING COMPLETE!")
print("="*60)
print(f"Best Params: {best_params}")
print(f"Best Test Loss: {best_loss:.4f}")
print("\nAll Results:")
for r in sorted(results, key=lambda x: x['test_loss']):
    print(f"  Loss: {r['test_loss']:.4f} | {r}")

# Save results
with open('tuning_results.json', 'w') as f:
    json.dump({
        'best_params': best_params,
        'best_loss': float(best_loss),
        'all_results': results
    }, f, indent=2)
print("\n📊 Results saved to tuning_results.json")
