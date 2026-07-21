import threading
import av

class ProcessFrame():
    def __init__(self):
        self.lock = threading.Lock()
        self.inferencing = False
        self.prediction = None
    
    def inference(self, image):
        try:
            
        except:
            
        finally:
            with self.lock():
                self.inferencing = False
        
        
    def build_overlay(self, image):
        # use stored prediction
    
    def recv(self, frame):
        image = frame.to_ndarray(frame="bgr24")
        
        with self.lock():
            if not self.inferencing:
                should_score = True
                self.inferencing = True
            else:
                should_score = False                
            
        if should_score:
            self.inference(image)
            
        image = self.build_overlay(image)
            
        return av.VideoFrame.from_ndarray(image, format="bgr24")