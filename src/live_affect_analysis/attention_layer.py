import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        self.hidden_size = max(channels // reduction, 1)
        
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        self.layer1 = tf.keras.layers.Dense(self.hidden_size, activation="gelu")
        self.layer2 = tf.keras.layers.Dense(channels, activation="sigmoid")
        
    def call(self, x):
        weights = self.pool(x)
        weights = self.layer1(weights)
        weights = self.layer2(weights)
        
        weights = weights[:, tf.newaxis, tf.newaxis, :]
        
        return x * weights