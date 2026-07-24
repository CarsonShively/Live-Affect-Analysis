import tensorflow as tf

class LowLatencyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.backbone = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            pooling="avg",
            include_top=False,
            include_preprocessing=True
        )
        self.backbone.trainable = False
        
        self.dropout = tf.keras.layers.Dropout(0.15)
        
        self.category_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.valence_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.arousal_hidden = tf.keras.layers.Dense(128, activation="gelu")
        
        self.category_head = tf.keras.layers.Dense(26)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        
    def call(self, x, training=False):
        
        feature_vector = self.backbone(x, training=False)
        feature_vector = self.dropout(feature_vector)
        
        category = self.category_hidden(feature_vector)
        valence = self.valence_hidden(feature_vector)
        arousal = self.arousal_hidden(feature_vector)
        
        category = self.dropout(category, training=training)
        valence = self.dropout(valence, training=training)
        arousal = self.dropout(arousal, training=training)
        
        category = self.category_head(category)
        valence = self.valence_head(valence)
        arousal = self.arousal_head(arousal)
        
        out = {
            "category": category,
            "valence": valence,
            "arousal": arousal
        }
        
        return out