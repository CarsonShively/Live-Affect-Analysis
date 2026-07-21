import tensorflow as tf

class GatedFeatureFusion(tf.keras.Model):
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
        
        self.conv_start = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same", use_bias=False)
        self.norm_start = tf.keras.layers.BatchNormalization()
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False)
        self.norm_1 = tf.keras.layers.BatchNormalization()
        
        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False)
        self.norm_2 = tf.keras.layers.BatchNormalization()
        
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False)
        self.norm_3 = tf.keras.layers.BatchNormalization()
        
        self.conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False)
        self.norm_4 = tf.keras.layers.BatchNormalization()
        
        self.f_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.c_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.c_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        
        self.f_projection = tf.keras.layers.Dense(128, use_bias=False)
        self.c_projection = tf.keras.layers.Dense(128, use_bias=False)
        
        self.f_norm = tf.keras.layers.LayerNormalization()
        self.c_norm = tf.keras.layers.LayerNormalization()
        
        self.category_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        self.valence_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        self.arousal_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        self.dominance_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        self.gender_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        self.age_gate = tf.keras.layers.Dense(128, activation="sigmoid")
        
        self.category_hidden = tf.keras.layers.Dense(128)
        self.valence_hidden = tf.keras.layers.Dense(128)
        self.arousal_hidden = tf.keras.layers.Dense(128)
        self.dominance_hidden = tf.keras.layers.Dense(128)
        self.gender_hidden = tf.keras.layers.Dense(128)
        self.age_hidden = tf.keras.layers.Dense(128)
        
        self.category_norm = tf.keras.layers.BatchNormalization()
        self.valence_norm = tf.keras.layers.BatchNormalization()
        self.arousal_norm = tf.keras.layers.BatchNormalization()
        self.dominance_norm = tf.keras.layers.BatchNormalization()
        self.gender_norm = tf.keras.layers.BatchNormalization()
        self.age_norm = tf.keras.layers.BatchNormalization()
        
        self.category_head = tf.keras.layers.Dense(26)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)
        self.gender_head = tf.keras.layers.Dense(1)
        self.age_head = tf.keras.layers.Dense(3)
        
    def call(self, x, training=False):
        
        feature_map = self.backbone(x, training=False)
        
        x = self.conv_start(feature_map)
        x = self.norm_start(x, training=training)
        x = tf.keras.activations.gelu(x)
        residual = x
        
        x = self.conv_1(x)
        x = self.norm_1(x, training=training)
        x = tf.keras.activations.gelu(x)
        x = self.conv_2(x)
        x = self.norm_2(x, training=training)
        x = residual + x
        x = tf.keras.activations.gelu(x)
        residual = x
        
        x = self.conv_3(x)
        x = self.norm_3(x, training=training)
        x = tf.keras.activations.gelu(x)
        x = self.conv_4(x)
        x = self.norm_4(x, training=training)
        x = residual + x
        x = tf.keras.activations.gelu(x)
        
        f_prime = self.f_avg_pool(feature_map)
        c_prime_avg = self.c_avg_pool(x)
        c_prime_max = self.c_max_pool(x)
        
        c_prime = tf.concat([c_prime_avg, c_prime_max], axis=-1)
        
        f_prime = self.f_projection(f_prime)
        c_prime = self.c_projection(c_prime)
        
        f_prime = self.f_norm(f_prime)
        c_prime = self.c_norm(c_prime)
        
        f_prime = tf.keras.activations.gelu(f_prime)
        c_prime = tf.keras.activations.gelu(c_prime)
        
        difference = tf.abs(f_prime - c_prime)
        agreement = f_prime * c_prime
        
        gate_input = tf.concat([f_prime, c_prime, difference, agreement], axis=-1)
        
        category = self.category_gate(gate_input)
        valence = self.valence_gate(gate_input)
        arousal = self.arousal_gate(gate_input)
        dominance = self.dominance_gate(gate_input)
        gender = self.gender_gate(gate_input)
        age = self.age_gate(gate_input)
        
        category = category * f_prime + (1 - category) * c_prime
        valence = valence * f_prime + (1 - valence) * c_prime
        arousal = arousal * f_prime + (1 - arousal) * c_prime
        dominance = dominance * f_prime + (1 - dominance) * c_prime
        gender = gender * f_prime + (1 - gender) * c_prime
        age = age * f_prime + (1 - age) * c_prime
        
        category = self.category_hidden(category)
        valence = self.valence_hidden(valence)
        arousal = self.arousal_hidden(arousal)
        dominance = self.dominance_hidden(dominance)
        gender = self.gender_hidden(gender)
        age = self.age_hidden(age)
                
        category = self.category_norm(category, training=training)
        valence = self.valence_norm(valence, training=training)
        arousal = self.arousal_norm(arousal, training=training)
        dominance = self.dominance_norm(dominance, training=training)
        gender = self.gender_norm(gender, training=training)
        age = self.age_norm(age, training=training)
                
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