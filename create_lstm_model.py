from tensorflow.keras.models import Sequential
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
def create_lstm_model(input_shape, task_type='classification', use_learnable_query=False, use_multihead_attention=False):
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(units=1,activation = "sigmoid"))
    model.summary()
    

    # Conditional addition of learnable query
    if use_learnable_query:
        learnable_query = LearnableQuery(query_dim=128, kernel_regularizer=l2(1e-4))(model)
        # Reshape the query to match the batch size and sequence length of the input (e.g., shape (batch_size, 1, 128))
        query_reshaped = Reshape((1, 128))(learnable_query)  # Add batch dimension

        # Conditional addition of multi-head attention
        if use_multihead_attention:
            multi_head_attention_output = MultiHeadAttention(num_heads=8, key_dim=128)(
                query=query_reshaped, value=model, key=model)
            # Concatenate the LSTM output with the attention output
            combined_output = Concatenate(axis=1)([model, multi_head_attention_output])
        else:
            combined_output = model  # If no multi-head attention, just use LSTM output
    else:
        combined_output = model  # If no learnable query, just use LSTM output

    
    # Determine activation function and loss based on task type
    if task_type == 'regression':
        optimizer = "adam"
        loss_function = 'mean_squared_error'  # MSE for regression
    elif task_type == 'classification':
        optimizer = "adam"
        loss_function = 'binary_crossentropy'  # Binary crossentropy for binary classification
    else:
        raise ValueError("Invalid task type. Use 'regression' or 'classification'.")
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function)
    
    return model
