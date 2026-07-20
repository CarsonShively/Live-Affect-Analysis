import tensorflow as tf

class FrozenTest(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            pooling=None,
            include_preprocessing=True
        )
        
        self.backbone.trainable = False
        
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.hidden = tf.keras.layers.Dense(64)
        
        self.category_head = tf.keras.layers.Dense(8)
        self.satisfaction_head = tf.keras.layers.Dense(1)
        self.calmness_head = tf.keras.layers.Dense(1)
        self.valence_head = tf.keras.layers.Dense(1)
        self.arousal_head = tf.keras.layers.Dense(1)
        self.dominance_head = tf.keras.layers.Dense(1)
        
    def call(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.hidden(x)   
        
        cat_head = self.category_head(x)
        sat_head = self.satisfaction_head(x)
        cal_head = self.calmness_head(x)
        val_head = self.valence_head(x)
        arl_head = self.arousal_head(x)
        dom_head = self.dominance_head(x)
        
        out = {
            "category": cat_head,
            "satisfaction": sat_head,
            "calmness": cal_head,
            "valence": val_head,
            "arousal": arl_head,
            "dominance": dom_head
        }
        
        return out