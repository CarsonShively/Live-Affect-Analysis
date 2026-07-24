import tensorflow as tf

class AttentionPool(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.attention_score = tf.keras.layers.Conv2D(1, kernel_size=1)
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        channels = tf.shape(x)[-1]
        
        attention_scores = self.attention_score(x)
    
        attention_scores = tf.reshape(attention_scores, [batch_size, -1])
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        
        x = tf.reshape(x, [batch_size, -1, channels])
        
        attention_scores = attention_scores[..., tf.newaxis]
        
        attention_vector = tf.reduce_sum(x * attention_scores, axis=1)
        
        return attention_vector