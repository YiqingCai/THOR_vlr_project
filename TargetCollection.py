#!/usr/bin/env python

import robosims.server
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

env = robosims.server.Controller(
                player_screen_width=400,
                player_screen_height=400,
                darwin_build='/Users/Cece/Desktop/THOR/RoboSims/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64',
                # linux_build='/home/xiaoyan1/data/thor-cmu-201703101558-Linux64/thor-cmu-201703101558-Linux64',
                x_display="0.0")

env.start()
env.reset('FloorPlan224') # FloorPlan223 and FloorPlan224 are also available
#actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
actions = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight']
for i in range(2000):
    event = env.step(dict(action=random.choice(actions)))
    print random.choice(actions)
    print event.metadata['lastActionSuccess']
    if event.metadata['lastActionSuccess'] == True and np.std(event.frame) > 25:
        img = event.frame
        # plt.figure()
        # plt.imshow(event.frame)
        # print event.frame
        # cv2.imshow('img', event.frame)
        cv2.imwrite('/Users/Cece/Desktop/THOR/data/input/target_'+str(i)+'.jpg', img)
        # raw_input("Press Enter to continue...")
#        plt.savefig('/Users/Cece/Desktop/THOR/data/input/observation_'+str(i)+'.jpg')
    #observation, reward, done, info = env.step(dict(action=random.choice(actions)))
    #print(event.metadata['lastActionSuccess'])
env.stop()
