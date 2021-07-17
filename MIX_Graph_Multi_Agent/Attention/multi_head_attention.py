import  tensorflow as tf
def scaled_dot_product_attention(q, k, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    return attention_weights


# ## Multi-head Attention

# In[ ]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (-1, x.shape[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = scaled_dot_product_attention(q, k, mask)
        #hard_weight.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (-1, scaled_attention.shape[1], self.d_model))  # (batch_size, seq_len_q, d_model)

        return concat_attention, attention_weights

def main():
    temp_mha = MultiHeadAttention(d_model=256, num_heads=4)
    q = tf.random.uniform((64, 2, 60))  # (batch_size, encoder_sequence, d_model)
    k = tf.random.uniform((64, 10, 60))  # (batch_size, encoder_sequence, d_model)

    out, attn = temp_mha(k, k=k, q=q, mask=None)
    print(out.shape, attn.shape)



if __name__ == '__main__':
    main()