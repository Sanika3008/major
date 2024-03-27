from Detector import *
import os

def main():
    videoPath = 0
    #"test_videos/street.mp4"

    configPath = os.path.join("main", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("main", "frozen_inference_graph.pb")
    classesPath = os.path.join("main", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    result = detector.onVideo()
    pp = 'person'
    if pp in result:
        print("person")
    

if __name__ == '__main__':
    main()