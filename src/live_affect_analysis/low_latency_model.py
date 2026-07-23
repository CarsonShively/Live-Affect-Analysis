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
        
        self.dropout = tf.keras.layers.Dropout(0.20)
        
        self.category_hidden = tf.keras.layers.Dense(256)
        self.valence_hidden = tf.keras.layers.Dense(256)
        self.arousal_hidden = tf.keras.layers.Dense(256)
        self.dominance_hidden = tf.keras.layers.Dense(256)
        self.gender_hidden = tf.keras.layers.Dense(256)
        self.age_hidden = tf.keras.layers.Dense(256)
        
        self.category_norm = tf.keras.layers.LayerNormalization()
        self.valence_norm = tf.keras.layers.LayerNormalization()
        self.arousal_norm = tf.keras.layers.LayerNormalization()
        self.dominance_norm = tf.keras.layers.LayerNormalization()
        self.gender_norm = tf.keras.layers.LayerNormalization()
        self.age_norm = tf.keras.layers.LayerNormalization()
        
        
        self.category_hidden2 = tf.keras.layers.Dense(128)
        self.valence_hidden2 = tf.keras.layers.Dense(128)
        self.arousal_hidden2 = tf.keras.layers.Dense(128)
        self.dominance_hidden2 = tf.keras.layers.Dense(128)
        self.gender_hidden2 = tf.keras.layers.Dense(128)
        self.age_hidden2 = tf.keras.layers.Dense(128)
        
        self.category_norm2 = tf.keras.layers.LayerNormalization()
        self.valence_norm2 = tf.keras.layers.LayerNormalization()
        self.arousal_norm2 = tf.keras.layers.LayerNormalization()
        self.dominance_norm2 = tf.keras.layers.LayerNormalization()
        self.gender_norm2 = tf.keras.layers.LayerNormalization()
        self.age_norm2 = tf.keras.layers.LayerNormalization()
        
        self.category_hidden3 = tf.keras.layers.Dense(64)
        self.valence_hidden3 = tf.keras.layers.Dense(64)
        self.arousal_hidden3 = tf.keras.layers.Dense(64)
        self.dominance_hidden3 = tf.keras.layers.Dense(64)
        self.gender_hidden3 = tf.keras.layers.Dense(64)
        self.age_hidden3 = tf.keras.layers.Dense(64)
        
        self.category_norm3 = tf.keras.layers.LayerNormalization()
        self.valence_norm3 = tf.keras.layers.LayerNormalization()
        self.arousal_norm3 = tf.keras.layers.LayerNormalization()
        self.dominance_norm3 = tf.keras.layers.LayerNormalization()
        self.gender_norm3 = tf.keras.layers.LayerNormalization()
        self.age_norm3 = tf.keras.layers.LayerNormalization()
        
        self.category_head = tf.keras.layers.Dense(26)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)
        self.gender_head = tf.keras.layers.Dense(1)
        self.age_head = tf.keras.layers.Dense(3)
        
    def call(self, x, training=False):
        
        feature_vector = self.backbone(x, training=False)
        
        
        category = self.category_hidden(feature_vector)
        valence = self.valence_hidden(feature_vector)
        arousal = self.arousal_hidden(feature_vector)
        dominance = self.dominance_hidden(feature_vector)
        gender = self.gender_hidden(feature_vector)
        age = self.age_hidden(feature_vector)
                
        category = self.category_norm(category)
        valence = self.valence_norm(valence)
        arousal = self.arousal_norm(arousal)
        dominance = self.dominance_norm(dominance)
        gender = self.gender_norm(gender)
        age = self.age_norm(age)
                
        category = tf.keras.activations.gelu(category)
        valence = tf.keras.activations.gelu(valence)
        arousal = tf.keras.activations.gelu(arousal)
        dominance = tf.keras.activations.gelu(dominance)
        gender = tf.keras.activations.gelu(gender)
        age = tf.keras.activations.gelu(age)
        
        category = self.dropout(category, training=training)
        valence = self.dropout(valence, training=training)
        arousal = self.dropout(arousal, training=training)
        dominance = self.dropout(dominance, training=training)
        gender = self.dropout(gender, training=training)
        age = self.dropout(age, training=training)
        
        category = self.category_hidden2(category)
        valence = self.valence_hidden2(valence)
        arousal = self.arousal_hidden2(arousal)
        dominance = self.dominance_hidden2(dominance)
        gender = self.gender_hidden2(gender)
        age = self.age_hidden2(age)
                
        category = self.category_norm2(category)
        valence = self.valence_norm2(valence)
        arousal = self.arousal_norm2(arousal)
        dominance = self.dominance_norm2(dominance)
        gender = self.gender_norm2(gender)
        age = self.age_norm2(age)
                
        category = tf.keras.activations.gelu(category)
        valence = tf.keras.activations.gelu(valence)
        arousal = tf.keras.activations.gelu(arousal)
        dominance = tf.keras.activations.gelu(dominance)
        gender = tf.keras.activations.gelu(gender)
        age = tf.keras.activations.gelu(age)
        
        category = self.dropout(category, training=training)
        valence = self.dropout(valence, training=training)
        arousal = self.dropout(arousal, training=training)
        dominance = self.dropout(dominance, training=training)
        gender = self.dropout(gender, training=training)
        age = self.dropout(age, training=training)

        category = self.category_hidden3(category)
        valence = self.valence_hidden3(valence)
        arousal = self.arousal_hidden3(arousal)
        dominance = self.dominance_hidden3(dominance)
        gender = self.gender_hidden3(gender)
        age = self.age_hidden3(age)
                
        category = self.category_norm3(category)
        valence = self.valence_norm3(valence)
        arousal = self.arousal_norm3(arousal)
        dominance = self.dominance_norm3(dominance)
        gender = self.gender_norm3(gender)
        age = self.age_norm3(age)
                
        category = tf.keras.activations.gelu(category)
        valence = tf.keras.activations.gelu(valence)
        arousal = tf.keras.activations.gelu(arousal)
        dominance = tf.keras.activations.gelu(dominance)
        gender = tf.keras.activations.gelu(gender)
        age = tf.keras.activations.gelu(age)
        
        category = self.dropout(category, training=training)
        valence = self.dropout(valence, training=training)
        arousal = self.dropout(arousal, training=training)
        dominance = self.dropout(dominance, training=training)
        gender = self.dropout(gender, training=training)
        age = self.dropout(age, training=training)
        
        
        category = self.category_head(category)
        valence = self.valence_head(valence)
        arousal = self.arousal_head(arousal)
        dominance = self.dominance_head(dominance)
        gender = self.gender_head(gender)
        age = self.age_head(age)
        
        out = {
            "category": category,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "gender": gender,
            "age": age
        }
        
        return out