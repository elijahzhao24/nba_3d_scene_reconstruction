from player_detector import RoboflowPlayerDetector


# Coordinate the whole player tracking pipeline
class PlayerTrackingPipeline: any {

    def __init__(self)-> any :
        self.roboflowPlayerDetector = RoboflowPlayerDetector()
}