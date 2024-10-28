from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, Layer, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# Define a custom layer for learnable query
class LearnableQuery(Layer):
    def __init__(self, query_dim):
        super(LearnableQuery, self).__init__()
        self.query_dim = query_dim

    def build(self, input_shape):
        # Create a trainable weight for the query
        self.query = self.add_weight(name='query',
                                     shape=(1, 1, self.query_dim),  # Add an additional dimension to make it 3D
                                     initializer='random_normal',
                                     trainable=True)

    def call(self, inputs):
        # Return the learnable query for attention
        return self.query

# LSTM model definition with task type and options for learnable query and multi-head attention
def create_lstm_model(input_shape, task_type='classification', use_learnable_query=True, use_multihead_attention=True):
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with more complexity (128 units) and return sequences
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with return sequences for stacking
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Conditional addition of learnable query
    if use_learnable_query:
        learnable_query = LearnableQuery(query_dim=128, kernel_regularizer=l2(1e-4))(x)
        # Reshape the query to match the batch size and sequence length of the input (e.g., shape (batch_size, 1, 128))
        query_reshaped = Reshape((1, 128))(learnable_query)  # Add batch dimension

        # Conditional addition of multi-head attention
        if use_multihead_attention:
            multi_head_attention_output = MultiHeadAttention(num_heads=8, key_dim=128)(
                query=query_reshaped, value=x, key=x)
            # Concatenate the LSTM output with the attention output
            combined_output = Concatenate(axis=1)([x, multi_head_attention_output])
        else:
            combined_output = x  # If no multi-head attention, just use LSTM output
    else:
        combined_output = x  # If no learnable query, just use LSTM output

    # Third LSTM layer with return_sequences=False
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4))(combined_output)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Determine activation function and loss based on task type
    if task_type == 'regression':
        outputs = Dense(1, activation='linear')(x)  # Linear activation for regression
        loss_function = 'mean_squared_error'  # MSE for regression
    elif task_type == 'classification':
        outputs = Dense(1, activation='sigmoid')(x)  # Sigmoid activation for binary classification
        loss_function = 'binary_crossentropy'  # Binary crossentropy for binary classification
    else:
        raise ValueError("Invalid task type. Use 'regression' or 'classification'.")
    
    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_function)
    
    return model
