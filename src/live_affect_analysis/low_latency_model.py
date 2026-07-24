import tensorflow as tf
from live_affect_analysis.attention_pool import AttentionPool

class LowLatencyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.backbone = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            pooling=None,
            include_top=False,
            include_preprocessing=True
        )
        self.backbone.trainable = False
        
        self.happiness_attention = AttentionPool()
        self.calmness_attention = AttentionPool()
        self.sadness_attention = AttentionPool()
        self.fear_attention = AttentionPool()
        self.anger_attention = AttentionPool()
        self.valence_attention = AttentionPool()
        self.arousal_attention = AttentionPool()       
        
        self.happiness_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.calmness_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.sadness_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.fear_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.anger_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.valence_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.arousal_hidden = tf.keras.layers.Dense(128, activation="gelu")
        
        self.happiness_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.calmness_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.sadness_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.fear_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.anger_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.valence_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        self.arousal_hidden2 = tf.keras.layers.Dense(64, activation="gelu")
        
        
        self.happiness_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.sadness_head = tf.keras.layers.Dense(1)
        self.fear_head = tf.keras.layers.Dense(1)
        self.anger_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
            tf.keras.layers.RandomContrast(0.15)
        ])
        
        
        
    def call(self, x, training=False):
        
        x = self.augmentation(x, training=training)
        
        feature_map = self.backbone(x, training=False)
        
        happiness_feature_attention = self.happiness_attention(feature_map)
        calmness_feature_attention = self.calmness_attention(feature_map)
        sadness_feature_attention = self.sadness_attention(feature_map)
        fear_feature_attention = self.fear_attention(feature_map)
        anger_feature_attention = self.anger_attention(feature_map)
        valence_feature_attention = self.valence_attention(feature_map)
        arousal_feature_attention = self.arousal_attention(feature_map)
    
    
        happiness = self.happiness_hidden(happiness_feature_attention)
        calmness = self.calmness_hidden(calmness_feature_attention)
        sadness = self.sadness_hidden(sadness_feature_attention)
        fear = self.fear_hidden(fear_feature_attention)
        anger = self.anger_hidden(anger_feature_attention)
        valence = self.valence_hidden(valence_feature_attention)
        arousal = self.arousal_hidden(arousal_feature_attention)
        
        happiness = self.happiness_hidden2(happiness)
        calmness = self.calmness_hidden2(calmness)
        sadness = self.sadness_hidden2(sadness)
        fear = self.fear_hidden2(fear)
        anger = self.anger_hidden2(anger)
        valence = self.valence_hidden2(valence)
        arousal = self.arousal_hidden2(arousal)

        happiness = self.happiness_head(happiness)
        calmness = self.calmness_head(calmness)
        sadness = self.sadness_head(sadness)
        fear = self.fear_head(fear)
        anger = self.anger_head(anger)
        valence = self.valence_head(valence)
        arousal = self.arousal_head(arousal)
        
        out = {
            "happiness": happiness,
            "calmness": calmness,
            "sadness": sadness,
            "fear": fear,
            "anger": anger,
            "valence": valence,
            "arousal": arousal
        }
        
        return out