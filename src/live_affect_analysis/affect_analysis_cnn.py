import tensorflow as tf

class AffectAnalysisCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.filters_count_1 = 32
        self.filters_count_2 = 64
        self.filters_count_3 = 128
        self.filters_count_4 = 256
        self.dropout_value = 0.2

        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.gobal_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.dropout = tf.keras.layers.Dropout(self.dropout_value)

        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters_count_1, kernel_size=(3, 3), padding="same", activation="gelu")
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters_count_2, kernel_size=(3, 3), padding="same", activation="gelu")
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters_count_3, kernel_size=(3, 3), padding="same", activation="gelu")
        self.conv4 = tf.keras.layers.Conv2D(filters=self.filters_count_4, kernel_size=(3, 3), padding="same", activation="gelu")

        self.linear1 = tf.keras.layers.Dense(128, activation="gelu")
        self.linear2 = tf.keras.layers.Dense(64, activation="gelu")
        self.linear3 = tf.keras.layers.Dense(32, activation="gelu")
        
        self.category_hidden = tf.keras.layers.Dense(16, activation="gelu")
        self.satisfaction_hidden = tf.keras.layers.Dense(16, activation="gelu")
        self.calmness_hidden = tf.keras.layers.Dense(16, activation="gelu")
        self.valence_hidden = tf.keras.layers.Dense(16, activation="gelu")
        self.arousal_hidden = tf.keras.layers.Dense(16, activation="gelu")
        self.dominance_hidden = tf.keras.layers.Dense(16, activation="gelu")
        
        self.category_head = tf.keras.layers.Dense(8)
        self.satisfaction_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_2_pooled = self.pool(x_2)
        x_3 = self.conv3(x_2_pooled)
        x_4 = self.conv4(x_3)
        x_vector = self.gobal_pool(x_4)
        x_linear_1 = self.linear1(x_vector)
        x_linear_1 = self.dropout(x_linear_1, training=training)
        x_linear_2 = self.linear2(x_linear_1)
        x_linear_2 = self.dropout(x_linear_2, training=training)
        x_linear_3 = self.linear3(x_linear_2)
        x_linear_3 = self.dropout(x_linear_3, training=training)
        
        x_cat_hidden = self.category_hidden(x_linear_3)
        x_cat_hidden = self.dropout(x_cat_hidden, training=training)
        x_sat_hidden = self.satisfaction_hidden(x_linear_3)
        x_sat_hidden = self.dropout(x_sat_hidden, training=training)
        x_cal_hidden = self.calmness_hidden(x_linear_3)
        x_cal_hidden = self.dropout(x_cal_hidden, training=training)
        x_val_hidden = self.valence_hidden(x_linear_3)
        x_val_hidden = self.dropout(x_val_hidden, training=training)
        x_arl_hidden = self.arousal_hidden(x_linear_3)
        x_arl_hidden = self.dropout(x_arl_hidden, training=training)
        x_dom_hidden = self.dominance_hidden(x_linear_3)
        x_dom_hidden = self.dropout(x_dom_hidden, training=training)
        
        x_cat_head = self.category_head(x_cat_hidden)
        x_sat_head = self.satisfaction_head(x_sat_hidden)
        x_cal_head = self.calmness_head(x_cal_hidden)
        x_val_head = self.valence_head(x_val_hidden)
        x_arl_head = self.arousal_head(x_arl_hidden)
        x_dom_head = self.dominance_head(x_dom_hidden)
        
        out = {
            "category": x_cat_head,
            "satisfaction": x_sat_head,
            "calmness": x_cal_head,
            "valence": x_val_head,
            "arousal": x_arl_head,
            "dominance": x_dom_head
        }
        
        return out