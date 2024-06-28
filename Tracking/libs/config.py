#-----------------------------------------------
# Author: Mathis Morales                       
# Email : mathis-morales@outlook.fr             
# git   : https://github.com/MathisMM            
#-----------------------------------------------


# nuScenes detection score threshold for each category:
def score_thresh(cat):
    if cat == 'car':
        return 0.4
    elif cat =='pedestrian':
        return 0.6
    elif cat =='truck':
        return 0.4
    elif cat =='bus':
        return 0.2
    elif cat =='bicycle':
        return 0.3
    elif cat =='motorcycle':
        return 0.3
    elif cat =='trailer':
        return 0.1
    else : 
        return 0.8  # high filter for false negative, wrong names, and untracked objects 