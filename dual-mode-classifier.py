import os
import sys
import json
import time
import pickle
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import gc # Import garbage collector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad # <-- CHANGED: Use Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # <-- CHANGED: Removed ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Androguard for APK analysis
try:
    from androguard.core.apk import APK
    from androguard.misc import AnalyzeAPK
    ANDROGUARD_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: Androguard not installed. APK analysis will be limited.")
    ANDROGUARD_AVAILABLE = False

# Streamlit for UI
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# --- NEW IMPORTS for SHAP, TFLite, and ONNX ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: SHAP not installed. Explainability will be disabled. (pip install shap)")
    SHAP_AVAILABLE = False

try:
    import tf2onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: tf2onnx not installed. ONNX conversion will be disabled. (pip install tf2onnx onnxruntime)")
    ONNX_AVAILABLE = False
# --- END NEW IMPORTS ---


warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # File paths (Update these paths before running)
    BENIGN_DIR = 'CCCS-CIC-Benign-CSVs'
    MALICIOUS_DIR = 'CCCS-CIC-Malicious-CSVs'

    MODELS_DIR = "models_practical"

    # Model artifacts
    MODEL_PATH = os.path.join(MODELS_DIR, "malware_classifier_practical.keras") 
    MODEL_PATH_LEGACY = os.path.join(MODELS_DIR, "malware_classifier_practical.h5") 
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
    FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.npy")
    CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.npy")
    METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
    PATTERN_RULES_PATH = os.path.join(MODELS_DIR, "malware_patterns.json")
    
    # --- NEW: Paths for SHAP/TFLite data ---
    X_TEST_DATA_PATH = os.path.join(MODELS_DIR, "X_test_scaled.npy")
    Y_TEST_DATA_PATH = os.path.join(MODELS_DIR, "Y_test.npy")
    X_TRAIN_SAMPLE_PATH = os.path.join(MODELS_DIR, "X_train_sample.npy")
    
    # Converted model paths
    TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "malware_classifier.tflite")
    ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "malware_classifier.onnx")
    # --- END NEW ---

    # Model hyperparameters
    MODEL_VERSION = "v3.1.0-Adagrad" # Updated version
    LEARNING_RATE = 0.001 # Initial LR for Adagrad (often 0.01 or 0.001)
    BATCH_SIZE = 128
    EPOCHS = 150
    PATIENCE_EARLY_STOP = 20
    # PATIENCE_LR_REDUCE = 8 # <-- REMOVED: Using fixed LR with Adagrad
    L2_LAMBDA = 0.0005
    DROPOUT_RATE_1 = 0.4
    DROPOUT_RATE_2 = 0.3
    DROPOUT_RATE_3 = 0.25

    # Data processing
    TEST_SIZE = 0.20
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    # Memory Efficiency: DATA SAMPLING LIMITS
    CHUNK_SIZE = 10000 
    MAX_TOTAL_SAMPLES = 500000 
    DATA_DTYPE = np.float32 
    
    # Proportional sampling rate (1/8 = 0.125)
    SAMPLING_PROBABILITY = 0.125 

    # APK analysis
    MAX_APK_SIZE_MB = 150
    APK_ANALYSIS_TIMEOUT = 120

    # Malware behavior patterns (remains the same)
    SUSPICIOUS_PERMISSIONS = [
        'SEND_SMS', 'RECEIVE_SMS', 'READ_SMS', 'WRITE_SMS',
        'CALL_PHONE', 'READ_CALL_LOG', 'WRITE_CALL_LOG',
        'READ_CONTACTS', 'WRITE_CONTACTS',
        'ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION',
        'RECORD_AUDIO', 'CAMERA',
        'SYSTEM_ALERT_WINDOW', 'WRITE_SETTINGS'
    ]
    MALICIOUS_PERMISSIONS = [
        'INSTALL_PACKAGES', 'DELETE_PACKAGES',
        'REQUEST_INSTALL_PACKAGES',
        'MOUNT_UNMOUNT_FILESYSTEMS',
        'WRITE_SECURE_SETTINGS',
        'CHANGE_COMPONENT_ENABLED_STATE'
    ]
    MALICIOUS_API_PATTERNS = [
        'Runtime.exec',   'ProcessBuilder', 'DexClassLoader',   'sendTextMessage',
        'getSubscriberId', 'getDeviceId', 'getLine1Number', 'HttpURLConnection',
        'URLConnection', 'Socket', 'encrypt', 'decrypt',
        'getRunningTasks', 'killBackgroundProcesses'
    ]


# ============================================================================
# CUSTOM EXCEPTIONS (Remain the same)
# ============================================================================

class APKAnalysisError(Exception):
    """Raised when APK analysis fails"""
    pass

class DataLoadError(Exception):
    """Raised when dataset loading fails"""
    pass

class ModelNotTrainedError(Exception):
    """Raised when model hasn't been trained yet"""
    pass


# ============================================================================
# MEMORY OPTIMIZED DATA LOADING (UPDATED WITH FIX)
# ============================================================================

def map_dtypes(chunk, data_dtype):
    """Downcasts numerical columns in a chunk to save memory (Proactive Downcasting)."""
    for col in chunk.select_dtypes(include=[np.number]).columns:
        if str(chunk[col].dtype).startswith('int'):
            c_min = chunk[col].min()
            c_max = chunk[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                chunk[col] = chunk[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                chunk[col] = chunk[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                chunk[col] = chunk[col].astype(np.int32)
            else:
                chunk[col] = chunk[col].astype(np.int64) 
        elif str(chunk[col].dtype).startswith('float'):
            chunk[col] = chunk[col].astype(data_dtype)
            
    return chunk

def load_real_datasets():
    """Load and merge datasets from benign and malicious CSV directories using chunking and PROPORTIONAL RANDOM SAMPLING."""
    print("\n" + "="*70)
    print("📥 LOADING DATASETS (CHUNKED, DOWNCATSED, & 1/8 RANDOMLY SAMPLED)")
    print("="*70)

    all_chunks = []
    all_columns = set()
    total_loaded_samples = 0
    
    # Function to process a directory
    def process_dir(directory, is_benign=False):
        nonlocal total_loaded_samples
        if os.path.exists(directory):
            print(f"\n✅ Found directory: {directory}")
            files = [f for f in os.listdir(directory) if f.endswith(".csv")]
            print(f"  Loading data from {len(files)} files (chunk size: {Config.CHUNK_SIZE})...")
            print(f"  Sampling probability: {Config.SAMPLING_PROBABILITY*100:.1f}% per chunk.")
            
            for filename in files:
                if total_loaded_samples >= Config.MAX_TOTAL_SAMPLES:
                    return True # Signal to stop
                
                file_path = os.path.join(directory, filename)
                label = 'Benign' if is_benign else filename.replace('.csv', '').strip()
                
                try:
                    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=Config.CHUNK_SIZE):
                        
                        if total_loaded_samples >= Config.MAX_TOTAL_SAMPLES:
                            break
                        
                        # Proportional Random Sampling
                        sample_mask = np.random.rand(len(chunk)) < Config.SAMPLING_PROBABILITY
                        chunk_sampled = chunk[sample_mask].copy()
                        
                        # If the sampled chunk is empty, skip to next chunk
                        if chunk_sampled.empty:
                            del chunk, chunk_sampled
                            gc.collect()
                            continue
                        
                        # Assign the label
                        chunk_sampled['Label'] = label

                        # --- NEW FIX: Auto-detect and drop non-numeric columns ---
                        # This prevents the ValueError during type casting
                        non_numeric_cols = chunk_sampled.select_dtypes(exclude=[np.number]).columns
                        cols_to_drop = [col for col in non_numeric_cols if col != 'Label']
                        
                        if cols_to_drop:
                            # print(f"   Dropping {len(cols_to_drop)} non-numeric cols: {cols_to_drop[:2]}...") # Optional: for debugging
                            chunk_sampled = chunk_sampled.drop(columns=cols_to_drop)
                        # --- END FIX ---

                        # Memory Optimization 1: Enforce DTYPE and Downcast
                        chunk_sampled = map_dtypes(chunk_sampled, Config.DATA_DTYPE)
                        
                        all_chunks.append(chunk_sampled)
                        all_columns.update(set(chunk_sampled.columns)) # Now only contains numeric cols + 'Label'
                        
                        total_loaded_samples += len(chunk_sampled)
                        
                        # Aggressive Garbage Collection after processing chunk
                        del chunk, chunk_sampled 
                        gc.collect() 

                    print(f"   Sampled chunks from: {filename}. Current samples: {total_loaded_samples:,}")
                except Exception as e:
                    print(f"❌ Error loading/sampling chunks from {filename}: {e}")
            return False # Continue loading
        else:
            print(f"⚠️ Directory {directory} not found")
            return False # Continue loading

    # --- Load Benign Datasets ---
    if process_dir(Config.BENIGN_DIR, is_benign=True): 
        print(f"🛑 Maximum sample limit ({Config.MAX_TOTAL_SAMPLES:,}) reached. Stopping data loading.")
    
    # --- Load Malicious Datasets ---
    if total_loaded_samples < Config.MAX_TOTAL_SAMPLES:
        if process_dir(Config.MALICIOUS_DIR, is_benign=False): 
            print(f"🛑 Maximum sample limit ({Config.MAX_TOTAL_SAMPLES:,}) reached. Stopping data loading.")


    if len(all_chunks) == 0:
        raise DataLoadError(
            "No data loaded from the specified directories or sampling resulted in zero rows."
        )

    # Align and concatenate chunks
    print(f"\n🔗 Aligning and concatenating {len(all_chunks)} sampled chunks (Total samples: {total_loaded_samples:,})...")
    
    # Create the final column list for alignment (always include 'Label')
    # all_columns is now guaranteed to be clean (only numeric + 'Label')
    final_cols = list(all_columns - {'Label'}) + ['Label']
    
    # Concatenation remains memory-efficient using the generator and dtype map
    df_merged = pd.concat(
        (chunk.reindex(columns=final_cols, fill_value=0).astype(
            # This cast will now succeed as all non-'Label' cols are numeric
            {col: Config.DATA_DTYPE for col in final_cols if col != 'Label'}
        ) for chunk in all_chunks), 
        ignore_index=True
    )
    
    print(f"✅ Merged dataset shape: {df_merged.shape}")

    # Drop any temporary/unnecessary columns that might have been carried over
    possible_junk_cols = ['class', 'Type', 'target']
    df_merged = df_merged.drop(columns=[col for col in possible_junk_cols if col in df_merged.columns and col != 'Label'], errors='ignore')

    # Explicitly delete the list of chunks and aligned dfs
    del all_chunks
    gc.collect()

    return df_merged


def simplify_labels(df):
    """Simplify malware classification to 3 practical classes"""
    print("\n" + "="*70)
    print("🔄 SIMPLIFYING CLASSIFICATION LABELS")
    print("="*70)

    # Map original labels to simplified classes
    label_mapping = {
        'Benign': 'Benign', 
        'benign': 'Benign',
        '0': 'Benign',

        # Malicious - High confidence threats (from the new filenames)
        'Adware': 'Malicious',
        'Banker': 'Malicious', 
        'SMS': 'Malicious',
        'SMS_Malware': 'Malicious',
        'Ransomware': 'Malicious',
        'Riskware': 'Malicious',
        'Trojan': 'Malicious',
        'Backdoor': 'Malicious',
        'malware': 'Malicious',
        '1': 'Malicious',
        'Dropper': 'Malicious', 
        'FileInfector': 'Malicious', 
        'Spy': 'Malicious', 
        'Zeroday': 'Malicious', 

        # Suspicious - Uncertain or low-risk threats
        'Scareware': 'Suspicious',
        'Unknown': 'Suspicious',
        'PUA': 'Suspicious',
        'Greyware': 'Suspicious',
        'NoCategory': 'Suspicious', 
    }

    original_labels = df['Label'].unique()
    print(f"Original labels found: {len(original_labels)}")

    #Apply mapping
    df['Label'] = df['Label'].astype(str).str.strip()
    df['SimplifiedLabel'] = df['Label'].map(label_mapping)
    df['SimplifiedLabel'] = df['SimplifiedLabel'].fillna('Suspicious')

    # Show distribution
    print(f"\n✅ Simplified Classification:")
    for label in ['Benign', 'Suspicious', 'Malicious']:
        count = (df['SimplifiedLabel'] == label).sum()
        percentage = count / len(df) * 100 if len(df) > 0 else 0
        print(f"   {label}: {count} samples ({percentage:.1f}%)")

    # Replace original label with simplified
    df['Label'] = df['SimplifiedLabel']
    df = df.drop(columns=['SimplifiedLabel'])

    return df


def preprocess_dataset(df):
    """Clean, encode, and prepare dataset for training (Optimized dtype)"""
    print("\n" + "="*70)
    print("🧹 PREPROCESSING DATASET")
    print("="*70)

    # Simplify labels first
    df = simplify_labels(df)

    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    print(f"\nRemoved {initial_shape - df.shape[0]} duplicate rows")

    # Separate features and labels
    if 'Label' not in df.columns:
        raise DataLoadError("'Label' column not found after preprocessing")

    X = df.drop(columns=['Label'])
    Y = df['Label'].astype(str).str.strip()

    # Remove non-numeric columns (This is now redundant but safe to keep)
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Removing {len(non_numeric_cols)} non-numeric columns (preprocessing step)")
        X = X.select_dtypes(include=[np.number])

    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"Filling {missing_count} missing values with 0")
        X = X.fillna(0)

    # Convert to configured float type (for final tensor consistency)
    X = X.astype(Config.DATA_DTYPE)

    # Store feature names
    feature_names = X.columns.tolist()
    print(f"✅ Total features: {len(feature_names)}")

    # Encode labels
    label_encoder = LabelEncoder()
    Y_int = label_encoder.fit_transform(Y)
    Y_categorical = to_categorical(Y_int)

    class_names = label_encoder.classes_
    print(f"\n✅ Final Classes: {len(class_names)}")
    for i, cls in enumerate(class_names):
        count = np.sum(Y_int == i)
        print(f"   {cls}: {count} samples ({count/len(Y)*100:.1f}%)")

    # Clear original DataFrame to free up memory before the final train/test splits
    del df
    gc.collect()

    return X.values, Y_categorical, Y_int, feature_names, class_names, label_encoder


# ============================================================================
# 2. MODEL ARCHITECTURE (OPTIMIZED FOR 3 CLASSES)
# ============================================================================

def build_practical_model(input_dim, num_classes=3):
    """
    Build optimized model for practical 3-class classification
    """
    print("\n" + "="*70)
    print("🏗️ BUILDING PRACTICAL MODEL (Using Adagrad)")
    print("="*70)

    regularizer = l2(Config.L2_LAMBDA)

    model = Sequential([
        # Layer 1: 256 units
        Dense(256, kernel_initializer='he_normal',
              kernel_regularizer=regularizer, input_shape=(input_dim,)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(Config.DROPOUT_RATE_1),

        # Layer 2: 128 units 
        Dense(128, kernel_initializer='he_normal',
              kernel_regularizer=regularizer),
        BatchNormalization(),
        Activation('relu'),
        Dropout(Config.DROPOUT_RATE_2),

        # Layer 3: 64 units
        Dense(64, kernel_initializer='he_normal',
              kernel_regularizer=regularizer),
        BatchNormalization(),
        Activation('relu'),
        Dropout(Config.DROPOUT_RATE_3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # --- CHANGED: Use Adagrad optimizer as requested ---
    optimizer = Adagrad(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model architecture (Practical):")
    print(f"   Input: {input_dim} features")
    print(f"   Hidden: 256 → 128 → 64")
    print(f"   Output: {num_classes} classes")
    print(f"   Optimizer: Adagrad (LR={Config.LEARNING_RATE})")
    print(f"   Total parameters: {model.count_params():,}")

    return model


# ============================================================================
# 3. MEMORY-EFFICIENT TRAINING
# ============================================================================

def to_tf_dataset(X, Y, batch_size, shuffle=False):
    """Creates a memory-efficient tf.data.Dataset with prefetching."""
    # Ensure inputs are float32 for consistency
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X, tf.float32), Y))
    
    if shuffle:
        # Shuffle buffer size is an important parameter for performance/memory tradeoff
        dataset = dataset.shuffle(buffer_size=10000, seed=Config.RANDOM_STATE) 
    
    # Optimization: Use prefetch to keep the GPU/CPU busy
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) 
    return dataset


def train_practical_model(X, Y, Y_int, feature_names, class_names, label_encoder):
    """Train model with practical 3-class classification using tf.data.Dataset"""
    print("\n" + "="*70)
    print("🚀 TRAINING PRACTICAL MODEL (WITH TF.DATA OPTIMIZATION)")
    print("="*70)

    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Split data
    X_train_val, X_test, Y_train_val, Y_test, Y_int_train_val, Y_int_test = train_test_split(
        X, Y, Y_int,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=Y_int
    )

    val_size_adjusted = Config.VAL_SIZE / (1 - Config.TEST_SIZE)
    X_train, X_val, Y_train, Y_val, Y_int_train, Y_int_val = train_test_split(
        X_train_val, Y_train_val, Y_int_train_val, 
        test_size=val_size_adjusted,
        random_state=Config.RANDOM_STATE,
        stratify=Y_int_train_val 
    )

    # Aggressive Memory Cleanup: Delete large intermediate splits
    del X, Y, Y_int, X_train_val, Y_train_val, Y_int_train_val
    gc.collect() 
    
    print(f"✅ Data split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- NEW: Save a sample of training data for SHAP background ---
    print(f"\n💾 Saving data samples for SHAP and Conversion...")
    sample_size = min(200, X_train_scaled.shape[0]) # Use 200 samples for background
    train_sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
    X_train_sample = X_train_scaled[train_sample_indices]
    np.save(Config.X_TRAIN_SAMPLE_PATH, X_train_sample)
    print(f"   Saved {len(X_train_sample)} training samples for SHAP.")

    # Aggressive Memory Cleanup: Delete original split arrays after scaling
    del X_train, X_val
    gc.collect()
    
    # Compute class weights
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(Y_int_train),
        y=Y_int_train
    )
    class_weights = dict(enumerate(class_weights_array))

    print(f"\n✅ Class weights:")
    for i, (cls, weight) in enumerate(zip(class_names, class_weights_array)):
        print(f"   {cls}: {weight:.3f}")

    # Build model
    model = build_practical_model(X_train_scaled.shape[1], len(class_names))

    # Optimization: Convert arrays to tf.data.Dataset for high-throughput I/O
    train_dataset = to_tf_dataset(X_train_scaled, Y_train, Config.BATCH_SIZE, shuffle=True)
    val_dataset = to_tf_dataset(X_val_scaled, Y_val, Config.BATCH_SIZE)
    
    # Aggressive Memory Cleanup: Delete scaled arrays after creating Datasets
    del X_train_scaled, Y_train, X_val_scaled, Y_val, Y_int_train, Y_int_val
    gc.collect() 

    # Setup callbacks
    checkpoint_path = os.path.join(Config.MODELS_DIR, "checkpoint_best.keras")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.PATIENCE_EARLY_STOP,
        restore_best_weights=True,
        verbose=1
    )

    # --- REMOVED: ReduceLROnPlateau to keep LR fixed as requested ---
    # lr_scheduler = ReduceLROnPlateau(...)

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # --- UPDATED: Removed lr_scheduler ---
    callbacks = [early_stop, checkpoint]

    # Train (Using tf.data.Dataset)
    print(f"\n🎯 Training...")
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=Config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks, # <-- Use updated callbacks list
        verbose=1
    )

    # Aggressive Memory Cleanup: Clear datasets
    del train_dataset, val_dataset
    gc.collect() 

    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATION ON TEST SET")
    print("="*70)

    # X_test_scaled is small enough to keep for final evaluation
    Y_pred_prob = model.predict(X_test_scaled, verbose=0)
    Y_pred_int = np.argmax(Y_pred_prob, axis=1)
    Y_test_int = np.argmax(Y_test, axis=1)

    accuracy = accuracy_score(Y_test_int, Y_pred_int)
    print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")

    print("\n📋 Classification Report:")
    print(classification_report(Y_test_int, Y_pred_int,
                                target_names=class_names, digits=4))

    # AUC-ROC
    try:
        auc_scores = roc_auc_score(Y_test, Y_pred_prob,
                                   average=None, multi_class='ovr')
        print("\n🎯 AUC-ROC Scores:")
        for cls, auc in zip(class_names, auc_scores):
            print(f"   {cls}: {auc:.4f}")
        macro_auc = np.mean(auc_scores)
        print(f"   Macro Average: {macro_auc:.4f}")
    except Exception as e:
        print(f"⚠️ Could not compute AUC: {e}")
        macro_auc = 0.0

    # MCC and F1
    mcc = matthews_corrcoef(Y_test_int, Y_pred_int)
    f1_macro = f1_score(Y_test_int, Y_pred_int, average='macro')
    f1_weighted = f1_score(Y_test_int, Y_pred_int, average='weighted')

    print(f"\n🧮 Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"📊 F1 Scores:")
    print(f"   Macro: {f1_macro:.4f}")
    print(f"   Weighted: {f1_weighted:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(Y_test_int, Y_pred_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Practical Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(Config.MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\n✅ Confusion matrix saved: {cm_path}")
    plt.close()

    # Save artifacts
    print("\n💾 Saving model artifacts...")

    # Save in the modern .keras format
    try:
        model.save(Config.MODEL_PATH)
        print(f"✅ Model saved: {Config.MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save model to {Config.MODEL_PATH}: {e}")

    with open(Config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    with open(Config.ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    np.save(Config.FEATURE_NAMES_PATH, np.array(feature_names))
    np.save(Config.CLASS_NAMES_PATH, class_names)
    
    # --- NEW: Save test data for SHAP and conversion ---
    np.save(Config.X_TEST_DATA_PATH, X_test_scaled)
    np.save(Config.Y_TEST_DATA_PATH, Y_test)
    print(f"   Saved test data for SHAP/Conversion.")

    # Save metadata
    metadata = {
        'version': Config.MODEL_VERSION,
        'trained_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'classification_type': '3-class practical (Benign, Suspicious, Malicious)',
        'num_samples_train': int(X_train.shape[0] if 'X_train' in locals() else 0),
        'num_samples_val': int(X_val.shape[0] if 'X_val' in locals() else 0),
        'num_samples_test': int(X_test.shape[0]),
        'num_features': int(X_test.shape[1]),
        'num_classes': len(class_names),
        'classes': class_names.tolist(),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'mcc': float(mcc),
        'auc_macro': float(macro_auc)
    }

    with open(Config.METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save malware pattern rules
    save_pattern_rules()

    print(f"✅ All artifacts saved to: {Config.MODELS_DIR}/")

    # Final cleanup of test data
    del X_test_scaled, Y_test, Y_int_test
    gc.collect()
    
    return model, scaler, label_encoder


# ============================================================================
# 4. MALWARE PATTERN DETECTION (DETAILED ANALYSIS)
# ============================================================================

# (This section is unchanged)

def save_pattern_rules():
    """Save detailed malware detection patterns"""
    patterns = {
        'version': Config.MODEL_VERSION,
        'description': 'Evidence-based malware detection patterns',

        'risk_indicators': {
            'critical': {
                'description': 'Indicators of clear malicious intent',
                'permissions': Config.MALICIOUS_PERMISSIONS,
                'api_patterns': Config.MALICIOUS_API_PATTERNS,
                'weight': 10
            },
            'high': {
                'description': 'Suspicious behavior requiring investigation',
                'permissions': Config.SUSPICIOUS_PERMISSIONS,
                'behaviors': [
                    'Hidden from launcher', 'Obfuscated code',
                    'Anti-debugging techniques', 'Root access attempts',
                    'Background service always running'
                ], 'weight': 5
            },
            'medium': {
                'description': 'Potentially risky but could be legitimate',
                'behaviors': [
                    'Requests many permissions', 'Network activity without user interaction',
                    'Accesses location frequently', 'Takes screenshots', 'Records audio/video'
                ], 'weight': 2
            }
        },

        'malware_patterns': {
            'SMS_Fraud': {
                'description': 'Sends premium SMS without consent',
                'indicators': ['SEND_SMS', 'sendTextMessage', 'getSubscriberId'],
                'confidence_threshold': 0.7
            },
            'Spyware': {
                'description': 'Collects personal data covertly',
                'indicators': ['READ_CONTACTS', 'READ_SMS', 'READ_CALL_LOG',
                               'ACCESS_FINE_LOCATION', 'RECORD_AUDIO', 'CAMERA'],
                'confidence_threshold': 0.65
            },
            'Banking_Trojan': {
                'description': 'Targets banking/financial apps',
                'indicators': ['SYSTEM_ALERT_WINDOW', 'getRunningTasks',
                               'KeyEvent', 'AccessibilityService'],
                'confidence_threshold': 0.75
            },
            'Ransomware': {
                'description': 'Encrypts files and demands payment',
                'indicators': ['encrypt', 'decrypt', 'WRITE_EXTERNAL_STORAGE',
                               'DeviceAdminReceiver', 'SYSTEM_ALERT_WINDOW'],
                'confidence_threshold': 0.8
            },
            'Backdoor': {
                'description': 'Provides remote access',
                'indicators': ['Runtime.exec', 'ProcessBuilder', 'Socket',
                               'DexClassLoader', 'RECEIVE_BOOT_COMPLETED'],
                'confidence_threshold': 0.75
            },
            'Adware': {
                'description': 'Displays intrusive advertisements',
                'indicators': ['AdView', 'InterstitialAd', 'SYSTEM_ALERT_WINDOW',
                               'URLConnection'],
                'confidence_threshold': 0.6
            }
        },

        'benign_indicators': {
            'description': 'Indicators of legitimate applications',
            'signs': [
                'Published by known developer', 'High number of reviews',
                'Permissions match functionality', 'No obfuscation',
                'Signed with valid certificate', 'No hidden activities/services'
            ]
        }
    }

    with open(Config.PATTERN_RULES_PATH, 'w') as f:
        json.dump(patterns, f, indent=4)

    print(f"✅ Malware pattern rules saved: {Config.PATTERN_RULES_PATH}")


def analyze_malware_patterns(apk_features, permissions):
    """
    Detailed pattern-based analysis to identify malware behavior
    Returns: List of detected patterns with confidence scores
    """
    detected_patterns = []
    risk_score = 0

    # Load pattern rules
    if os.path.exists(Config.PATTERN_RULES_PATH):
        with open(Config.PATTERN_RULES_PATH, 'r') as f:
            patterns = json.load(f)
    else:
        return [], 0

    # Check for critical risk indicators
    critical_perms = set(patterns['risk_indicators']['critical']['permissions'])
    detected_critical = [p for p in permissions if any(cp in p for cp in critical_perms)]

    if detected_critical:
        risk_score += 10 * len(detected_critical)
        detected_patterns.append({
            'type': 'Critical Risk',
            'description': 'Dangerous permissions detected',
            'evidence': detected_critical,
            'severity': 'CRITICAL'
        })

    # Check for suspicious permissions
    suspicious_perms = set(patterns['risk_indicators']['high']['permissions'])
    detected_suspicious = [p for p in permissions if any(sp in p for sp in suspicious_perms)]

    if len(detected_suspicious) >= 3:  # Multiple suspicious permissions
        risk_score += 5 * len(detected_suspicious)
        detected_patterns.append({
            'type': 'High Risk Behavior',
            'description': 'Multiple suspicious permissions',
            'evidence': detected_suspicious,
            'severity': 'HIGH'
        })

    # Check for specific malware patterns
    for malware_type, pattern_info in patterns['malware_patterns'].items():
        matched_indicators = [ind for ind in pattern_info['indicators']
                                if any(ind in p for p in permissions)]

        if len(matched_indicators) >= 2:  # At least 2 indicators match
            confidence = len(matched_indicators) / len(pattern_info['indicators'])

            if confidence >= pattern_info['confidence_threshold']:
                risk_score += 10
                detected_patterns.append({
                    'type': malware_type,
                    'description': pattern_info['description'],
                    'evidence': matched_indicators,
                    'confidence': f"{confidence*100:.1f}%",
                    'severity': 'HIGH' if confidence > 0.7 else 'MEDIUM'
                })

    # Normalize risk score to 0-100
    risk_score = min(100, risk_score)

    return detected_patterns, risk_score


# ============================================================================
# 5. APK FEATURE EXTRACTION
# ============================================================================

# (This section is unchanged)

def extract_comprehensive_features(apk_path, trained_feature_names):
    """Extract comprehensive static features from APK"""
    if not ANDROGUARD_AVAILABLE:
        raise APKAnalysisError("Androguard is not installed")

    print(f"\n🔍 Analyzing APK: {os.path.basename(apk_path)}")

    # Validate APK
    if not os.path.exists(apk_path):
        raise APKAnalysisError(f"APK file not found: {apk_path}")

    file_size_mb = os.path.getsize(apk_path) / (1024 * 1024)
    if file_size_mb > Config.MAX_APK_SIZE_MB:
        raise APKAnalysisError(f"APK too large: {file_size_mb:.1f}MB")

    features = {}
    start_time = time.time()

    try:
        print("   Performing deep static analysis...")
        a, d, dx = AnalyzeAPK(apk_path)

        # Extract permissions
        permissions = a.get_permissions()
        print(f"   Extracted {len(permissions)} permissions")

        # Store all permissions for pattern analysis
        all_perms = permissions

        # Map known permissions to features
        for perm in Config.SUSPICIOUS_PERMISSIONS + Config.MALICIOUS_PERMISSIONS:
            full_perm = f"android.permission.{perm}"
            features[f'perm_{perm}'] = 1 if full_perm in permissions else 0

        # API calls (Partial extraction for memory efficiency in feature dictionary)
        api_count = 0
        for method in dx.get_methods():
            api_name = f"{method.class_name}.{method.name}"
            # Only track the API patterns that might be in the trained features
            if any(p in api_name for p in Config.MALICIOUS_API_PATTERNS) or api_name in trained_feature_names:
                features[f'api_{api_name}'] = features.get(f'api_{api_name}', 0) + 1
            api_count += 1

        # Component counts
        features['num_activities'] = len(a.get_activities())
        features['num_services'] = len(a.get_services())
        features['num_receivers'] = len(a.get_receivers())
        features['num_providers'] = len(a.get_providers())

        # Metadata
        features['apk_size_mb'] = file_size_mb
        features['min_sdk'] = a.get_min_sdk_version() if a.get_min_sdk_version() else 0
        features['target_sdk'] = a.get_target_sdk_version() if a.get_target_sdk_version() else 0

        analysis_time = time.time() - start_time
        print(f"   ✅ Analysis completed in {analysis_time:.2f}s (Features: {len(features)})")
        
        # Explicitly delete large objects from memory
        del a, d, dx 
        gc.collect()

    except Exception as e:
        print(f"   ⚠️ Deep analysis failed: {e}, using basic analysis")
        all_perms = []
        try:
            a = APK(apk_path)
            permissions = a.get_permissions()
            all_perms = permissions

            for perm in Config.SUSPICIOUS_PERMISSIONS + Config.MALICIOUS_PERMISSIONS:
                full_perm = f"android.permission.{perm}"
                features[f'perm_{perm}'] = 1 if full_perm in permissions else 0

            features['num_activities'] = len(a.get_activities())
            features['num_services'] = len(a.get_services())
            features['num_receivers'] = len(a.get_receivers())
            features['apk_size_mb'] = file_size_mb

            print(f"   ✅ Basic analysis completed")
            del a
            gc.collect()

        except Exception as e_basic:
            raise APKAnalysisError(f"Failed to parse APK: {e_basic}")

    # Align features with training set
    feature_vector = np.zeros(len(trained_feature_names), dtype=Config.DATA_DTYPE)

    for i, fname in enumerate(trained_feature_names):
        if fname in features:
            feature_vector[i] = features[fname]

    matched_features = np.sum(feature_vector > 0)
    print(f"   Matched {matched_features}/{len(trained_feature_names)} training features")

    return feature_vector.reshape(1, -1), all_perms


# ============================================================================
# 6. PREDICTION WITH PATTERN ANALYSIS
# ============================================================================

# (This section is unchanged)

def load_model_artifacts():
    """Load all saved model artifacts (Optimized for prediction)"""
    model_exists = os.path.exists(Config.MODEL_PATH) or os.path.exists(Config.MODEL_PATH_LEGACY)

    if not model_exists:
        raise ModelNotTrainedError(
            "Model has not been trained yet. Please run the training cell."
        )

    try:
        # Optimization: Load model in read-only mode, without compilation
        if os.path.exists(Config.MODEL_PATH):
            print(f"Loading model from: {Config.MODEL_PATH}")
            model = load_model(Config.MODEL_PATH, compile=False)
        elif os.path.exists(Config.MODEL_PATH_LEGACY):
            print(f"Loading model from: {Config.MODEL_PATH_LEGACY}")
            model = load_model(Config.MODEL_PATH_LEGACY, compile=False)
        else:
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH} or {Config.MODEL_PATH_LEGACY}")

        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        with open(Config.ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)

        feature_names = np.load(Config.FEATURE_NAMES_PATH, allow_pickle=True).tolist()
        class_names = np.load(Config.CLASS_NAMES_PATH, allow_pickle=True)

        with open(Config.METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        return model, scaler, label_encoder, feature_names, class_names, metadata

    except Exception as e:
        raise Exception(
            f"Failed to load model artifacts: {e}\n"
            "Solution: Re-run the training cell to generate the model artifacts."
        )


def predict_apk(apk_path, model, scaler, label_encoder, feature_names, class_names):
    """Predict malware class with detailed pattern analysis"""
    # Extract features
    feature_vector, permissions = extract_comprehensive_features(apk_path, feature_names)

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    pred_prob = model.predict(feature_vector_scaled, verbose=0)[0]
    pred_index = np.argmax(pred_prob)
    pred_label = class_names[pred_index]
    confidence = pred_prob[pred_index]
    
    # Aggressive Memory Cleanup: Delete intermediate arrays
    del feature_vector, feature_vector_scaled 
    gc.collect()

    # Analyze patterns
    patterns, risk_score = analyze_malware_patterns(pred_prob, permissions)

    # Determine risk level
    if pred_label == 'Malicious' or risk_score >= 70:
        risk_level = 'HIGH'
        risk_color = 'red'
    elif pred_label == 'Suspicious' or risk_score >= 40:
        risk_level = 'MEDIUM'
        risk_color = 'orange'
    else:
        risk_level = 'LOW'
        risk_color = 'green'

    result = {
        'prediction': pred_label,
        'confidence': float(confidence),
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_color': risk_color,
        'all_probabilities': {cls: float(prob) for cls, prob in zip(class_names, pred_prob)},
        'permissions': permissions,
        'detected_patterns': patterns,
        'num_features_matched': int(np.sum(pred_prob > 0.01)) # A proxy for matched features
    }

    return result


# ============================================================================
# 7. STREAMLIT UI WITH PATTERN ANALYSIS
# ============================================================================

# (This section is unchanged)

def run_streamlit_ui():
    """Production Streamlit interface with pattern analysis"""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(
        page_title="Practical Malware Classifier",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ Practical Android Malware Classifier")
    st.markdown("**Simplified 3-Class Detection: Benign, Suspicious, Malicious**")
    st.markdown("---")

    try:
        # Load model artifacts once at the start
        model, scaler, encoder, feature_names, class_names, metadata = load_model_artifacts()

        # Sidebar
        with st.sidebar:
            st.success("✅ Model Loaded")
            st.subheader("Model Information")
            st.markdown(f"**Version:** {metadata['version']}")
            st.markdown(f"**Classification:** {metadata['classification_type']}")
            st.markdown(f"**Accuracy:** {metadata['accuracy']*100:.2f}%")
            st.markdown(f"**F1 (Macro):** {metadata['f1_macro']:.4f}")

            st.markdown("---")
            st.subheader("Detection Classes")
            for cls in class_names:
                emoji = "✅" if cls == "Benign" else ("⚠️" if cls == "Suspicious" else "🚨")
                st.markdown(f"{emoji} **{cls}**")

        # Main content
        st.subheader("📱 Upload APK for Analysis")

        uploaded_file = st.file_uploader(
            "Choose an APK file",
            type=['apk'],
            help=f"Maximum file size: {Config.MAX_APK_SIZE_MB}MB"
        )

        tmp_path = None
        if uploaded_file:
            try:
                # Use tempfile for secure and automatic cleanup
                with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner("🔍 Analyzing APK with pattern detection..."):
                    result = predict_apk(
                        tmp_path, model, scaler, encoder,
                        feature_names, class_names
                    )

                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Classification")
                    if result['risk_level'] == 'HIGH':
                        st.error(f"🚨 **{result['prediction'].upper()}**")
                    elif result['risk_level'] == 'MEDIUM':
                        st.warning(f"⚠️ **{result['prediction'].upper()}**")
                    else:
                        st.success(f"✅ **{result['prediction'].upper()}**")

                    st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                    st.metric("Risk Score", f"{result['risk_score']}/100",
                              delta=result['risk_level'])

                    st.markdown("---")
                    st.subheader("All Probabilities")
                    prob_df = pd.DataFrame(
                        list(result['all_probabilities'].items()),
                        columns=['Class', 'Probability']
                    ).sort_values('Probability', ascending=False)
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)

                with col2:
                    st.subheader(f"📋 Permissions ({len(result['permissions'])})")

                    with st.expander("View all permissions", expanded=False):
                        st.code('\n'.join(result['permissions']))

                    # Detected patterns
                    if result['detected_patterns']:
                        st.subheader("🔍 Detected Malware Patterns")
                        for pattern in result['detected_patterns']:
                            severity = pattern['severity']
                            emoji = "🚨" if severity == 'CRITICAL' else ("⚠️" if severity == 'HIGH' else "ℹ️")

                            with st.expander(f"{emoji} {pattern['type']} ({severity})", expanded=True):
                                st.markdown(f"**Description:** {pattern['description']}")
                                st.markdown(f"**Evidence:**")
                                for evidence in pattern['evidence']:
                                    st.markdown(f"- `{evidence}`")
                                if 'confidence' in pattern:
                                    st.markdown(f"**Match Confidence:** {pattern['confidence']}")
                    else:
                        st.info("✅ No suspicious patterns detected")

                    st.markdown("---")
                    st.info(f"**Features matched (Approx.):** {result['num_features_matched']}/{len(feature_names)}")

            except APKAnalysisError as e:
                st.error(f"❌ APK Analysis Error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.exception(e)
            finally:
                # Memory Optimization: Guarantee cleanup of the temporary file
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    # print(f"Cleaned up temp file: {tmp_path}") # Optional: for debugging

    except ModelNotTrainedError as e:
        st.error("❌ Model Not Trained")
        st.warning(str(e))

    except Exception as e:
        st.error("❌ Failed to Load Model")
        st.exception(e)


# ============================================================================
# 8. NEW: SHAP EXPLAINABILITY
# ============================================================================

def run_shap_explanation(num_samples=5):
    """Loads the model and explains predictions on test data using SHAP."""
    if not SHAP_AVAILABLE:
        print("❌ SHAP not installed. Cannot run explanation.")
        print("   Please run: pip install shap")
        return

    print("\n" + "="*70)
    print("🧠 RUNNING SHAP EXPLAINABILITY")
    print("="*70)

    try:
        # 1. Load artifacts
        print("Loading model and data...")
        model = load_model(Config.MODEL_PATH)
        feature_names = np.load(Config.FEATURE_NAMES_PATH, allow_pickle=True)
        class_names = np.load(Config.CLASS_NAMES_PATH, allow_pickle=True)
        X_train_sample = np.load(Config.X_TRAIN_SAMPLE_PATH, allow_pickle=True)
        X_test_scaled = np.load(Config.X_TEST_DATA_PATH, allow_pickle=True)
        Y_test = np.load(Config.Y_TEST_DATA_PATH, allow_pickle=True)
        Y_test_int = np.argmax(Y_test, axis=1)

        print(f"✅ Loaded model, {len(feature_names)} features, {len(class_names)} classes.")
        print(f"   Using {len(X_train_sample)} samples for SHAP background.")
        print(f"   Explaining {num_samples} samples from test set.")

        # 2. Create Explainer
        # For Keras/TF2, GradientExplainer is a good choice.
        # We use the small sample of training data as the "background"
        explainer = shap.GradientExplainer(model, X_train_sample)

        # 3. Select samples to explain
        explain_samples = X_test_scaled[:num_samples]
        
        # 4. Get SHAP values
        print("\nCalculating SHAP values (this may take a moment)...")
        shap_values = explainer.shap_values(explain_samples)

        print("✅ SHAP values calculated. Displaying results:")

        # 5. Display explanations
        for i in range(num_samples):
            true_label = class_names[Y_test_int[i]]
            pred_prob = model.predict(explain_samples[i:i+1], verbose=0)[0]
            pred_label = class_names[np.argmax(pred_prob)]
            
            print("\n" + "-"*50)
            print(f"Sample {i+1}:")
            print(f"   TRUE Label:     {true_label}")
            print(f"   PREDICTED Label: {pred_label} (Confidence: {np.max(pred_prob):.2%})")
            print("-" * 50)

            # Get the SHAP values for the *predicted* class
            shap_values_for_pred_class = shap_values[np.argmax(pred_prob)][i]
            
            # Get indices of top 10 features
            abs_shap = np.abs(shap_values_for_pred_class)
            top_indices = np.argsort(abs_shap)[-10:][::-1] # Top 10 descending

            print(f"   Top 10 features contributing to '{pred_label}':")
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_val = shap_values_for_pred_class[idx]
                original_val = explain_samples[i][idx]
                
                direction = " (PUSHES TOWARDS)" if shap_val > 0 else " (pulls away from)"
                if pred_label == "Benign" and shap_val < 0:
                     direction = " (PUSHES TOWARDS)" # For Benign, negative SHAP is good
                
                print(f"     - {feature_name: <40} | SHAP: {shap_val: .4f} {direction}")

        print("\n" + "="*70)
        print("✅ SHAP analysis complete.")
        print("   To visualize plots, run this in a Jupyter Notebook and use:")
        print("   shap.summary_plot(shap_values[0], X_test_scaled[:num_samples], feature_names=feature_names)")
        print("="*70)

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Please run the training first ('python your_script.py train')")
    except Exception as e:
        print(f"❌ An error occurred during SHAP analysis: {e}")


# ============================================================================
# 9. NEW: TFLITE & ONNX CONVERSION
# ============================================================================

def convert_model_formats():
    """Loads the trained .keras model and converts it to TFLite and ONNX."""
    print("\n" + "="*70)
    print("🔄 CONVERTING MODEL TO TFLITE & ONNX")
    print("="*70)

    try:
        # 1. Load the trained Keras model
        print(f"Loading model from: {Config.MODEL_PATH}")
        model = load_model(Config.MODEL_PATH)
        print(f"✅ Model loaded successfully.")

        # --- TFLite Conversion ---
        print("\nConverting to TensorFlow Lite...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Add optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Define representative dataset for quantization (optional but recommended)
            try:
                X_train_sample = np.load(Config.X_TRAIN_SAMPLE_PATH)
                def representative_dataset_gen():
                    for i in range(len(X_train_sample)):
                        yield [X_train_sample[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                # Force float16 fallback quantization
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
                    tf.lite.OpsSet.SELECT_TF_OPS # Enable TensorFlow ops.
                ]
                converter.target_spec.supported_types = [tf.float16]
                print("   Using representative dataset for quantization.")
            except FileNotFoundError:
                print("   No representative dataset found. Proceeding without quantization.")

            tflite_model = converter.convert()

            with open(Config.TFLITE_MODEL_PATH, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ TFLite model saved: {Config.TFLITE_MODEL_PATH}")
            print(f"   Size: {os.path.getsize(Config.TFLITE_MODEL_PATH) / 1024:.2f} KB")

        except Exception as e:
            print(f"❌ TFLite conversion failed: {e}")

        # --- ONNX Conversion ---
        if not ONNX_AVAILABLE:
            print("\n⚠️ Skipping ONNX conversion (tf2onnx not installed).")
            return

        print("\nConverting to ONNX...")
        try:
            # Get the input signature from the model
            input_signature = [
                tf.TensorSpec(model.input_shape, model.inputs[0].dtype, name="input_1")
            ]
            
            # Convert
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13 # A common opset version
            )
            
            with open(Config.ONNX_MODEL_PATH, "wb") as f:
                f.write(model_proto.SerializeToString())
                
            print(f"✅ ONNX model saved: {Config.ONNX_MODEL_PATH}")
            print(f"   Size: {os.path.getsize(Config.ONNX_MODEL_PATH) / 1024:.2f} KB")

            # Optional: Verify ONNX model
            print("   Verifying ONNX model...")
            ort_session = onnxruntime.InferenceSession(Config.ONNX_MODEL_PATH)
            X_test_sample = np.load(Config.X_TEST_DATA_PATH)[0:1].astype(np.float32)
            
            ort_inputs = {ort_session.get_inputs()[0].name: X_test_sample}
            ort_outs = ort_session.run(None, ort_inputs)
            
            tf_out = model.predict(X_test_sample, verbose=0)
            
            np.testing.assert_allclose(tf_out, ort_outs[0], rtol=1e-3, atol=1e-5)
            print("✅ ONNX model verification successful!")

        except Exception as e:
            print(f"❌ ONNX conversion failed: {e}")

    except FileNotFoundError:
        print(f"❌ Model file not found: {Config.MODEL_PATH}")
        print("   Please run the training first ('python your_script.py train')")
    except Exception as e:
        print(f"❌ An error occurred during conversion: {e}")


# ============================================================================
# 10. SCRIPT EXECUTION (MAIN) - *** UPDATED ***
# ============================================================================

def run_training_pipeline():
    """Encapsulates the entire training process."""
    print("\n" + "="*70)
    print("🚀 STARTING MODEL TRAINING...")
    print("="*70)
    try:
        # 1. Load Data
        df = load_real_datasets()
        
        if df.empty:
            print("No data was loaded. Exiting.")
            return

        # 2. Preprocess Data
        X, Y_cat, Y_int, features, classes, encoder = preprocess_dataset(df)
        
        # 3. Train Model
        train_practical_model(X, Y_cat, Y_int, features, classes, encoder)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE.")
        print("="*70)

    except DataLoadError as e:
        print(f"\n❌ DATA LOADING FAILED: {e}")
    except Exception as e:
        print(f"\n❌ AN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

# Entry point for the script
if __name__ == "__main__":
    
    # Check if running in an interactive environment (like Colab/Jupyter)
    if 'ipykernel' in sys.modules:
        print("Script loaded in interactive environment (e.g., Colab).")
        print("Call functions directly:")
        print(" - run_training_pipeline()")
        print(" - run_streamlit_ui()")
        print(" - run_shap_explanation()")
        print(" - convert_model_formats()")
    
    # Check for command-line arguments
    elif len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            run_training_pipeline()
            
        elif command == 'shap':
            run_shap_explanation(num_samples=5)
            
        elif command == 'convert':
            convert_model_formats()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage:")
            print("   - To run the UI: streamlit run your_script.py")
            print("   - To train:      python your_script.py train")
            print("   - To explain:    python your_script.py shap")
            print("   - To convert:    python your_script.py convert")
    
    # Default to running the Streamlit UI
    # This is triggered by: streamlit run your_script_name.py
    else:
        if STREAMLIT_AVAILABLE:
            print("\n" + "="*70)
            print("🚀 STARTING STREAMLIT UI...")
            print("   To train, run:    python your_script.py train")
            print("   To explain, run:  python your_script.py shap")
            print("   To convert, run:  python your_script.py convert")
            print("="*70)
            run_streamlit_ui()
        else:
            print("\n" + "="*70)
            print("❌ Streamlit is not installed. UI cannot run.")
            print("   Install it with: pip install streamlit")
            print("   ---")
            print("   Usage:")
            print("   - To train:      python your_script.py train")
            print("   - To explain:    python your_script.py shap")
            print("   - To convert:    python your_script.py convert")
            print("="*70)