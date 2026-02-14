"""
CNN Model Architecture Module for Telugu Poem Classification.

Provides four model architectures:
1. Single-task CNN — predicts chandas (meter type)
2. Multi-task CNN — predicts chandas + source simultaneously
3. BiLSTM baseline — for CNN vs LSTM comparison
4. Attention CNN — CNN + Self-Attention mechanism

All models use the Functional API for Keras 3 compatibility.
GPU is disabled via CUDA_VISIBLE_DEVICES env var (set in main.py)
because CUDA kernel launches fail on the target VM.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import config


def configure_gpu():
    """
    Print device info. GPU is controlled via CUDA_VISIBLE_DEVICES env var.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("[Device] Running on CPU (GPU disabled via CUDA_VISIBLE_DEVICES)")


def build_cnn_model(n_classes: int, name: str = "chandas_cnn") -> Model:
    """
    Build a single-task CNN model for poem classification.

    Architecture:
        Embedding(30000, 200)
        Conv1D(256, kernel=5, relu) → BatchNorm → MaxPool(3)
        Conv1D(128, kernel=3, relu) → BatchNorm → MaxPool(3)
        Conv1D(64, kernel=3, relu) → BatchNorm → GlobalMaxPool
        Dropout(0.4) → Dense(256, relu) → Dropout(0.3) → Dense(128, relu)
        Dense(n_classes, softmax)

    This three-layer convolution stack mirrors how humans process
    poetic rhythm: first detecting syllable-level patterns (Conv1),
    then phrase-level meter (Conv2), then overall structure (Conv3).
    """
    input_layer = Input(shape=(config.MAX_SEQ_LEN,), name='text_input')

    # Embedding: learn dense representations for Telugu tokens
    x = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        name='embedding'
    )(input_layer)

    # Conv Block 1: detect local syllable patterns (5-gram)
    x = layers.Conv1D(config.CONV1_FILTERS, config.CONV1_KERNEL,
                      activation='relu', padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(pool_size=config.POOL_SIZE, name='pool1')(x)

    # Conv Block 2: detect phrase-level meter patterns (3-gram)
    x = layers.Conv1D(config.CONV2_FILTERS, config.CONV2_KERNEL,
                      activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(pool_size=config.POOL_SIZE, name='pool2')(x)

    # Conv Block 3: detect overall rhythmic structure
    x = layers.Conv1D(config.CONV3_FILTERS, config.CONV3_KERNEL,
                      activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.GlobalMaxPooling1D(name='global_pool')(x)

    # Classification head
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout1')(x)
    x = layers.Dense(config.DENSE1_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.Dense(config.DENSE2_UNITS, activation='relu', name='dense2')(x)

    # Output
    output = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=input_layer, outputs=output, name=name)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_multitask_cnn(n_chandas: int, n_source: int,
                        name: str = "multitask_cnn") -> Model:
    """
    Build a multi-task CNN model with shared feature extraction
    and two separate output heads for chandas and source prediction.
    """
    input_layer = Input(shape=(config.MAX_SEQ_LEN,), name='text_input')

    x = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        name='shared_embedding'
    )(input_layer)

    # Shared convolution backbone
    x = layers.Conv1D(config.CONV1_FILTERS, config.CONV1_KERNEL,
                      activation='relu', padding='same', name='shared_conv1')(x)
    x = layers.BatchNormalization(name='shared_bn1')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='shared_pool1')(x)

    x = layers.Conv1D(config.CONV2_FILTERS, config.CONV2_KERNEL,
                      activation='relu', padding='same', name='shared_conv2')(x)
    x = layers.BatchNormalization(name='shared_bn2')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='shared_pool2')(x)

    x = layers.Conv1D(config.CONV3_FILTERS, config.CONV3_KERNEL,
                      activation='relu', padding='same', name='shared_conv3')(x)
    x = layers.BatchNormalization(name='shared_bn3')(x)
    shared_features = layers.GlobalMaxPooling1D(name='shared_global_pool')(x)
    shared_features = layers.Dropout(config.DROPOUT_RATE, name='shared_dropout')(shared_features)

    # Chandas branch
    chandas_x = layers.Dense(config.DENSE2_UNITS, activation='relu',
                             name='chandas_dense1')(shared_features)
    chandas_x = layers.Dropout(0.3, name='chandas_dropout')(chandas_x)
    chandas_output = layers.Dense(n_chandas, activation='softmax',
                                  name='chandas_output')(chandas_x)

    # Source branch
    source_x = layers.Dense(config.DENSE2_UNITS, activation='relu',
                            name='source_dense1')(shared_features)
    source_x = layers.Dropout(0.3, name='source_dropout')(source_x)
    source_output = layers.Dense(n_source, activation='softmax',
                                 name='source_output')(source_x)

    model = Model(
        inputs=input_layer,
        outputs=[chandas_output, source_output],
        name=name
    )
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss={
            'chandas_output': 'categorical_crossentropy',
            'source_output': 'categorical_crossentropy'
        },
        loss_weights={
            'chandas_output': config.CHANDAS_LOSS_WEIGHT,
            'source_output': config.SOURCE_LOSS_WEIGHT
        },
        metrics={
            'chandas_output': ['accuracy'],
            'source_output': ['accuracy']
        }
    )
    return model


# ============================================================
# IMPROVEMENT 1: BiLSTM Baseline Model
# ============================================================

def build_bilstm_model(n_classes: int, name: str = "bilstm_chandas") -> Model:
    """
    Build a Bidirectional LSTM baseline model for comparison with CNN.

    While CNN captures LOCAL spatial patterns (syllable groups, gaṇas),
    BiLSTM captures SEQUENTIAL dependencies (long-range rhythm flow).
    """
    input_layer = Input(shape=(config.MAX_SEQ_LEN,), name='text_input')

    x = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        name='embedding'
    )(input_layer)

    x = layers.Bidirectional(
        layers.LSTM(config.LSTM_UNITS, return_sequences=True,
                    dropout=config.LSTM_DROPOUT, recurrent_dropout=0.1),
        name='bilstm_1'
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(config.LSTM_UNITS // 2,
                    dropout=config.LSTM_DROPOUT, recurrent_dropout=0.1),
        name='bilstm_2'
    )(x)

    x = layers.Dropout(config.DROPOUT_RATE, name='dropout1')(x)
    x = layers.Dense(config.DENSE2_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    output = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=input_layer, outputs=output, name=name)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# IMPROVEMENT 2: Self-Attention Layer + Attention CNN
# ============================================================

class SelfAttention(layers.Layer):
    """
    Self-Attention layer that learns which positions in the sequence
    are most important for classification.

    In Telugu poetry, this mimics how humans focus on specific syllable
    positions where yati (caesura) and prasa (rhyme) occur.
    """

    def __init__(self, attention_units: int = 64, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_units = attention_units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        cfg = super(SelfAttention, self).get_config()
        cfg.update({'attention_units': self.attention_units})
        return cfg


def build_attention_cnn_model(n_classes: int,
                              name: str = "attention_cnn") -> Model:
    """
    Build a CNN model enhanced with Self-Attention mechanism.

    Combines CNN's local pattern detection with self-attention that
    learns to focus on metrically important positions (yati/prasa).
    """
    input_layer = Input(shape=(config.MAX_SEQ_LEN,), name='text_input')

    x = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        name='embedding'
    )(input_layer)

    x = layers.Conv1D(config.CONV1_FILTERS, config.CONV1_KERNEL,
                      activation='relu', padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='pool1')(x)

    x = layers.Conv1D(config.CONV2_FILTERS, config.CONV2_KERNEL,
                      activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='pool2')(x)

    x = layers.Conv1D(config.CONV3_FILTERS, config.CONV3_KERNEL,
                      activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)

    # Self-Attention instead of GlobalMaxPooling
    x = SelfAttention(
        attention_units=config.ATTENTION_UNITS,
        name='self_attention'
    )(x)

    x = layers.Dropout(config.DROPOUT_RATE, name='dropout1')(x)
    x = layers.Dense(config.DENSE1_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.Dense(config.DENSE2_UNITS, activation='relu', name='dense2')(x)
    output = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=input_layer, outputs=output, name=name)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    configure_gpu()

    print("\n" + "=" * 60)
    print("1. SINGLE-TASK CNN (Chandas Prediction)")
    print("=" * 60)
    m1 = build_cnn_model(n_classes=8)
    m1.summary()

    print("\n" + "=" * 60)
    print("2. MULTI-TASK CNN (Chandas + Source)")
    print("=" * 60)
    m2 = build_multitask_cnn(n_chandas=8, n_source=28)
    m2.summary()

    print("\n" + "=" * 60)
    print("3. BiLSTM BASELINE (Chandas Prediction)")
    print("=" * 60)
    m3 = build_bilstm_model(n_classes=8)
    m3.summary()

    print("\n" + "=" * 60)
    print("4. ATTENTION CNN (Chandas Prediction)")
    print("=" * 60)
    m4 = build_attention_cnn_model(n_classes=8)
    m4.summary()
