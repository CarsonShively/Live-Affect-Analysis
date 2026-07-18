import tensorflow as tf
from live_affect_analysis.attention_layer import AttentionLayer

class AffectAnalysisCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.filters_count_1 = 32
        self.filters_count_2 = 64
        self.filters_count_3 = 128
        self.dropout_value = 0.1
        
        self.gobal_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.gobal_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        
        self.rescale = tf.keras.layers.Rescaling(1 / 255.0)
        self.dropout = tf.keras.layers.Dropout(self.dropout_value)

        self.conv_start = tf.keras.layers.Conv2D(filters=self.filters_count_1, kernel_size=(3, 3), padding="same", use_bias=True)
        self.shortcut1 = tf.keras.layers.Conv2D(filters=self.filters_count_2, kernel_size=(1, 1), strides=2, padding="same", use_bias=False)
        self.shortcut2 = tf.keras.layers.Conv2D(filters=self.filters_count_3, kernel_size=(1, 1), strides=2, padding="same", use_bias=False)

        self.attention1 = AttentionLayer(self.filters_count_1)
        self.attention2 = AttentionLayer(self.filters_count_2)
        self.attention3 = AttentionLayer(self.filters_count_3)

        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters_count_1, kernel_size=(3, 3), padding="same", use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters_count_1, kernel_size=(3, 3), padding="same", use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters_count_2, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)
        self.conv4 = tf.keras.layers.Conv2D(filters=self.filters_count_2, kernel_size=(3, 3), padding="same", use_bias=False)
        self.conv5 = tf.keras.layers.Conv2D(filters=self.filters_count_3, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)
        self.conv6 = tf.keras.layers.Conv2D(filters=self.filters_count_3, kernel_size=(3, 3), padding="same", use_bias=False)
        
        self.group_norm_start = tf.keras.layers.GroupNormalization(groups=8)
        self.shortcut1_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.shortcut2_norm = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_1 = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_2 = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_3 = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_4 = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_5 = tf.keras.layers.GroupNormalization(groups=8)
        self.group_norm_6 = tf.keras.layers.GroupNormalization(groups=8)


        self.linear1 = tf.keras.layers.Dense(256, activation="gelu")
        self.linear2 = tf.keras.layers.Dense(128, activation="gelu")
        
        self.category_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.satisfaction_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.calmness_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.valence_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.arousal_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.dominance_hidden = tf.keras.layers.Dense(64, activation="gelu")
        
        self.category_head = tf.keras.layers.Dense(8)
        self.satisfaction_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.rescale(x)
        
        x = self.conv_start(x)
        x = self.group_norm_start(x)
        x = tf.keras.activations.gelu(x)
        
        residual = x
        
        x = self.conv1(x)
        x = self.group_norm_1(x)
        x = tf.keras.activations.gelu(x)
        
        x = self.conv2(x)
        x = self.group_norm_2(x)
        
        x = self.attention1(x)
        
        residual = residual + x
        residual = tf.keras.activations.gelu(residual)
        
        shortcut = self.shortcut1(residual)
        shortcut = self.shortcut1_norm(shortcut)
        
        x = self.conv3(residual)
        x = self.group_norm_3(x)
        x = tf.keras.activations.gelu(x)
        
        x = self.conv4(x)
        x = self.group_norm_4(x)
        
        x = self.attention2(x)
        
        residual = shortcut + x
        residual = tf.keras.activations.gelu(residual)
        
        shortcut = self.shortcut2(residual)
        shortcut = self.shortcut2_norm(shortcut)
        
        x = self.conv5(residual)
        x = self.group_norm_5(x)
        x = tf.keras.activations.gelu(x)
        
        x = self.conv6(x)
        x = self.group_norm_6(x)
        
        x = self.attention3(x)
        
        residual = shortcut + x
        residual = tf.keras.activations.gelu(residual)

        avg_pool = self.gobal_avg_pool(residual)        
        max_pool = self.gobal_max_pool(residual)
        vector = tf.concat([avg_pool, max_pool], axis=-1)
        
        vector = self.linear1(vector)
        vector = self.linear2(vector)
        
        vector = self.dropout(vector, training=training)
        
        cat_hidden = self.category_hidden(vector)
        sat_hidden = self.satisfaction_hidden(vector)
        cal_hidden = self.calmness_hidden(vector)
        val_hidden = self.valence_hidden(vector)
        arl_hidden = self.arousal_hidden(vector)
        dom_hidden = self.dominance_hidden(vector)
        
        cat_head = self.category_head(cat_hidden)
        sat_head = self.satisfaction_head(sat_hidden)
        cal_head = self.calmness_head(cal_hidden)
        val_head = self.valence_head(val_hidden)
        arl_head = self.arousal_head(arl_hidden)
        dom_head = self.dominance_head(dom_hidden)
        
        out = {
            "category": cat_head,
            "satisfaction": sat_head,
            "calmness": cal_head,
            "valence": val_head,
            "arousal": arl_head,
            "dominance": dom_head
        }
        
        return out