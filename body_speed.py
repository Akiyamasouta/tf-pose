import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from convertFormat2Vector import CalculatAngle
from tf_pose import common

parts = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',  5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8:'RHip', 9: 'RKnee',
         10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'}

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0*0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
            
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    cnt = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #####################################################
        height, width = image.shape[0], image.shape[1]
        for human in humans:
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                x = int(body_part.x * width + 0.5)
                y = int(body_part.y * height + 0.5)
                debug_info = 'x:' + str(x) + 'y:' + str(y)

        print(f'###############################\n{round(cap.get(cv2.CAP_PROP_FPS))}')
        if cnt==0:
            prev_x = x
            prev_y = y
        if cnt > round(cap.get(cv2.CAP_PROP_FPS)):
            #前の座標と今の座標から速度計算出力
            dis = np.sqrt((x-prev_x)**2+(y-prev_y)**2)
            spped = dis / cap.get(cv2.CAP_PROP_FPS)
            print("#####################################\n", spped)

            #前の座標を今の座標に変える
            prev_x = x
            prev_y = y
            cnt = 0
        
        cnt += 1

            
                #print(parts[i])
                #print(body_part)
                #print(debug_info)
        #cangle = CalculatAngle()
        #angle = cangle.convertFormat2Vector(humans)
        #print('肘、肩、膝')
        #print(angle)
        #####################################################
        if not args.showBG:
            image = np.zeros(image.shape)
        
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')