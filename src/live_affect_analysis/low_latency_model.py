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
        
        self.confusion_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.happiness_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.confidence_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.calmness_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.distress_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.fear_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.anger_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.valence_hidden = tf.keras.layers.Dense(128, activation="gelu")
        self.arousal_hidden = tf.keras.layers.Dense(128, activation="gelu")
        
        
        self.confusion_head = tf.keras.layers.Dense(1)
        self.happiness_head = tf.keras.layers.Dense(1)
        self.confidence_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.distress_head = tf.keras.layers.Dense(1)
        self.fear_head = tf.keras.layers.Dense(1)
        self.anger_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.1)
        ])
        
    def call(self, x, training=False):
        
        x = self.augmentation(x, training=training)
        
        feature_vector = self.backbone(x, training=False)
        
        confusion = self.confusion_hidden(feature_vector)
        happiness = self.happiness_hidden(feature_vector)
        confidence = self.confidence_hidden(feature_vector)
        calmness = self.calmness_hidden(feature_vector)
        distress = self.distress_hidden(feature_vector)
        fear = self.fear_hidden(feature_vector)
        anger = self.anger_hidden(feature_vector)
        valence = self.valence_hidden(feature_vector)
        arousal = self.arousal_hidden(feature_vector)
        
        confusion = self.dropout(confusion, training=training)
        happiness = self.dropout(happiness, training=training)
        confidence = self.dropout(confidence, training=training)
        calmness = self.dropout(calmness, training=training)
        distress = self.dropout(distress, training=training)
        fear = self.dropout(fear, training=training)
        anger = self.dropout(anger, training=training)
        valence = self.dropout(valence, training=training)
        arousal = self.dropout(arousal, training=training)
        
        confusion = self.confusion_head(confusion)
        happiness = self.happiness_head(happiness)
        confidence = self.confidence_head(confidence)
        calmness = self.calmness_head(calmness)
        distress = self.distress_head(distress)
        fear = self.fear_head(fear)
        anger = self.anger_head(anger)
        valence = self.valence_head(valence)
        arousal = self.arousal_head(arousal)
        
        out = {
            
            "confusion": confusion,
            "happiness": happiness,
            "confidence": confidence,
            "calmness": calmness,
            "distress": distress,
            "fear": fear,
            "anger": anger,
            "valence": valence,
            "arousal": arousal
        }
        
        return out