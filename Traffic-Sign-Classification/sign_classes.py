"""
sign_classes.py
===============
Human-readable names for the 43 classes of the
German Traffic Sign Recognition Benchmark (GTSRB).

The index of each name matches the numeric ``ClassId`` used by the dataset,
so ``CLASS_NAMES[14]`` is always the "Stop" sign, etc.
"""

CLASS_NAMES = [
    "Speed limit (20 km/h)",                                   # 0
    "Speed limit (30 km/h)",                                   # 1
    "Speed limit (50 km/h)",                                   # 2
    "Speed limit (60 km/h)",                                   # 3
    "Speed limit (70 km/h)",                                   # 4
    "Speed limit (80 km/h)",                                   # 5
    "End of speed limit (80 km/h)",                            # 6
    "Speed limit (100 km/h)",                                  # 7
    "Speed limit (120 km/h)",                                  # 8
    "No passing",                                              # 9
    "No passing for vehicles over 3.5 tonnes",                 # 10
    "Right-of-way at the next intersection",                   # 11
    "Priority road",                                           # 12
    "Yield",                                                   # 13
    "Stop",                                                    # 14
    "No vehicles",                                             # 15
    "Vehicles over 3.5 tonnes prohibited",                     # 16
    "No entry",                                                # 17
    "General caution",                                         # 18
    "Dangerous curve to the left",                             # 19
    "Dangerous curve to the right",                            # 20
    "Double curve",                                            # 21
    "Bumpy road",                                              # 22
    "Slippery road",                                           # 23
    "Road narrows to the right",                               # 24
    "Road work",                                               # 25
    "Traffic signals",                                         # 26
    "Pedestrians",                                             # 27
    "Children crossing",                                       # 28
    "Bicycles crossing",                                       # 29
    "Beware of ice/snow",                                      # 30
    "Wild animals crossing",                                   # 31
    "End of all speed and passing limits",                     # 32
    "Turn right ahead",                                        # 33
    "Turn left ahead",                                         # 34
    "Ahead only",                                              # 35
    "Go straight or right",                                    # 36
    "Go straight or left",                                     # 37
    "Keep right",                                              # 38
    "Keep left",                                               # 39
    "Roundabout mandatory",                                    # 40
    "End of no passing",                                       # 41
    "End of no passing for vehicles over 3.5 tonnes",          # 42
]

NUM_CLASSES = len(CLASS_NAMES)  # 43
