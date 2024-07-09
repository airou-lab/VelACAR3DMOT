#-----------------------------------------------
# Author: Mathis Morales                       
# Email : mathis-morales@outlook.fr             
# git   : https://github.com/MathisMM            
#-----------------------------------------------


# nuScenes detection score threshold for each category:
def get_score_thresh(args,cat):

    if args.detection_method == 'CRN':
        if cat == 'car':
            return 0.312
        elif cat =='pedestrian':
            return 0.639
        elif cat =='truck':
            return 0.452
        elif cat =='bus':
            return 0.172
        elif cat =='bicycle':
            return 0.210
        elif cat =='motorcycle':
            return 0.217
        elif cat =='trailer':
            return 0.033
        else : 
            return 0.999  # high filter for false positives, wrong names, and untracked objects 

    elif args.detection_method == 'Radiant':
        if cat == 'car':
            return 0.056
        elif cat =='pedestrian':
            return 0.096
        elif cat =='truck':
            return 0.070
        elif cat =='bus':
            return 0.119
        elif cat =='bicycle':
            return 0.058
        elif cat =='motorcycle':
            return 0.031
        elif cat =='trailer':
            return 0.100
        else : 
            return 0.999  # high filter for false positives, wrong names, and untracked objects 