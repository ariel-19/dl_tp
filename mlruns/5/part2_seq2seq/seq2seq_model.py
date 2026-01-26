# tp5/part2_seq2seq/seq2seq_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from .simple_attention import SimpleAttention

def build_seq2seq_attention(
    input_vocab_size,
    output_vocab_size,
    latent_dim=256,
    max_encoder_seq_length=100,
    max_decoder_seq_length=20
):
    # Encoder
    encoder_inputs = Input(shape=(max_encoder_seq_length, input_vocab_size))
    encoder = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    
    # Concatenate forward and backward states
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # Attention layer
    context_vector, attention_weights = SimpleAttention()(encoder_outputs)
    
    # Decoder
    decoder_inputs = Input(shape=(max_decoder_seq_length, output_vocab_size))
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, 
        initial_state=[state_h, state_c]
    )
    
    # Concatenate context vector with decoder outputs
    context_vector = tf.expand_dims(context_vector, 1)
    context_vector = tf.tile(context_vector, [1, max_decoder_seq_length, 1])
    decoder_combined_context = Concatenate()([decoder_outputs, context_vector])
    
    # Dense layer for output
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_combined_context)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model