import tensorflow as tf

class BaselineCNN(tf.keras.Model):
    def __init__(self):
        
        self.fliters1 = 32
        self.fliters1 = 64
        self.fliters1 = 128
        self.fliters1 = 256
        
        self.rescale = tf.keras.layers.Rescaling(1 / 255.0)
        
        self.conv1 = tf.keras.layers.Conv2D(filters=self.fliters1, kernel_size=(5, 5), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=self.fliters2, kernel_size=(5, 5), activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=self.fliters3, kernel_size=(5, 5), activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(filters=self.fliters4, kernel_size=(5, 5), activation="relu")
        
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        self.hidden1 = tf.keras.layers.Dense(256, activation="relu")
        self.hidden1 = tf.keras.layers.Dense(128, activation="relu")
        
        self.category_hidden = tf.keras.layers.Dense(64, activation="relu")
        self.satisfaction_hidden = tf.keras.layers.Dense(64, activation="relu")
        self.calmness_hidden = tf.keras.layers.Dense(64, activation="relu")
        self.valence_hidden = tf.keras.layers.Dense(64, activation="relu")
        self.arousal_hidden = tf.keras.layers.Dense(64, activation="relu")
        self.dominance_hidden = tf.keras.layers.Dense(64, activation="relu")
        
        self.category_head = tf.keras.layers.Dense(8)
        self.satisfaction_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)
        
    def call(self, x):
        x = self.rescale(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.pool(x)
        
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        
        cat_hidden = self.category_hidden(x)
        sat_hidden = self.satisfaction_hidden(x)
        cal_hidden = self.calmness_hidden(x)
        val_hidden = self.valence_hidden(x)
        arl_hidden = self.arousal_hidden(x)
        dom_hidden = self.dominance_hidden(x)
        
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