import tensorflow as tf

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
        
        self.dropout = tf.keras.layers.Dropout(0.20)
        
        self.category_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.category_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.category_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.category_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)

        self.valence_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.valence_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.valence_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.valence_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)
        
        self.arousal_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.arousal_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.arousal_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.arousal_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)
        
        self.dominance_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.dominance_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.dominance_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.dominance_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)
        
        self.gender_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.gender_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.gender_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.gender_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)
        
        self.age_projection = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", use_bias=False)
        self.age_projection_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.age_spatial = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)
        self.age_spatial_norm = tf.keras.layers.GroupNormalization(groups=8)
        
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        self.category_hidden = tf.keras.layers.Dense(128)
        self.valence_hidden = tf.keras.layers.Dense(128)
        self.arousal_hidden = tf.keras.layers.Dense(128)
        self.dominance_hidden = tf.keras.layers.Dense(128)
        self.gender_hidden = tf.keras.layers.Dense(128)
        self.age_hidden = tf.keras.layers.Dense(128)
        
        self.category_norm = tf.keras.layers.LayerNormalization()
        self.valence_norm = tf.keras.layers.LayerNormalization()
        self.arousal_norm = tf.keras.layers.LayerNormalization()
        self.dominance_norm = tf.keras.layers.LayerNormalization()
        self.gender_norm = tf.keras.layers.LayerNormalization()
        self.age_norm = tf.keras.layers.LayerNormalization()
        
        
        self.category_hidden2 = tf.keras.layers.Dense(64)
        self.valence_hidden2 = tf.keras.layers.Dense(64)
        self.arousal_hidden2 = tf.keras.layers.Dense(64)
        self.dominance_hidden2 = tf.keras.layers.Dense(64)
        self.gender_hidden2 = tf.keras.layers.Dense(64)
        self.age_hidden2 = tf.keras.layers.Dense(64)
        
        self.category_norm2 = tf.keras.layers.LayerNormalization()
        self.valence_norm2 = tf.keras.layers.LayerNormalization()
        self.arousal_norm2 = tf.keras.layers.LayerNormalization()
        self.dominance_norm2 = tf.keras.layers.LayerNormalization()
        self.gender_norm2 = tf.keras.layers.LayerNormalization()
        self.age_norm2 = tf.keras.layers.LayerNormalization()
        
        self.category_head = tf.keras.layers.Dense(26)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)
        self.gender_head = tf.keras.layers.Dense(1)
        self.age_head = tf.keras.layers.Dense(3)
        
    def call(self, x, training=False):
        
        feature_map = self.backbone(x, training=False)
        
        category = self.category_projection(feature_map)
        category = self.category_projection_norm(category)
        category = tf.keras.activations.gelu(category)
        category = self.category_spatial(category)
        category = self.category_spatial_norm(category)
        category = tf.keras.activations.gelu(category)
        
        valence = self.valence_projection(feature_map)
        valence = self.valence_projection_norm(valence)
        valence = tf.keras.activations.gelu(valence)
        valence = self.valence_spatial(valence)
        valence = self.valence_spatial_norm(valence)
        valence = tf.keras.activations.gelu(valence)
        
        arousal = self.arousal_projection(feature_map)
        arousal = self.arousal_projection_norm(arousal)
        arousal = tf.keras.activations.gelu(arousal)
        arousal = self.arousal_spatial(arousal)
        arousal = self.arousal_spatial_norm(arousal)
        arousal = tf.keras.activations.gelu(arousal)

        dominance = self.dominance_projection(feature_map)
        dominance = self.dominance_projection_norm(dominance)
        dominance = tf.keras.activations.gelu(dominance)
        dominance = self.dominance_spatial(dominance)
        dominance = self.dominance_spatial_norm(dominance)
        dominance = tf.keras.activations.gelu(dominance)
        
        gender = self.gender_projection(feature_map)
        gender = self.gender_projection_norm(gender)
        gender = tf.keras.activations.gelu(gender)
        gender = self.gender_spatial(gender)
        gender = self.gender_spatial_norm(gender)
        gender = tf.keras.activations.gelu(gender)
        
        age = self.age_projection(feature_map)
        age = self.age_projection_norm(age)
        age = tf.keras.activations.gelu(age)
        age = self.age_spatial(age)
        age = self.age_spatial_norm(age)
        age = tf.keras.activations.gelu(age)
        
        category = self.pool(category)
        valence = self.pool(valence)
        arousal = self.pool(arousal)
        dominance = self.pool(dominance)
        gender = self.pool(gender)
        age = self.pool(age)
        
        category = self.category_hidden(category)
        valence = self.valence_hidden(valence)
        arousal = self.arousal_hidden(arousal)
        dominance = self.dominance_hidden(dominance)
        gender = self.gender_hidden(gender)
        age = self.age_hidden(age)
                
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