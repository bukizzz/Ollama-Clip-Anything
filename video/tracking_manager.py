from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker

class TrackingManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingManager, cls).__new__(cls)
            cls._instance.face_tracker = FaceTracker()
            cls._instance.object_tracker = ObjectTracker()
        return cls._instance

    def get_face_tracker(self):
        return self.face_tracker

    def get_object_tracker(self):
        return self.object_tracker
