import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from my_libs.covariance import Covariance

class Filter(object):
    def __init__(self, bbox3D, velocity, info, ID, cat, use_vel, use_R):

        self.initial_pos = bbox3D
        self.initial_vel = velocity
        self.time_since_update = 0
        self.id = ID
        self.hits = 1                   # number of total hits including the first detection
        self.info = info                # other information associated  
        self.tracking_name = cat

        self.use_vel = use_vel
        self.use_R = use_R

class KF(Filter):
    def __init__(self, bbox3D, velocity, info, ID, cat, use_vel, use_R):
        super().__init__(bbox3D, velocity, info, ID, cat, use_vel, use_R)

        # print(self.initial_pos)
        # print(self.initial_vel)
        # print(self.info)
        # print(self.id)
        # print(self.tracking_name)
        # exit()

        if use_vel:
            # using radar-measured velocity as part of the measured values

            self.kf = KalmanFilter(dim_x=10, dim_z=10)       
            # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

            # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
            # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz between 2 samples (dx,dy,dz = vx,vy,vz)
            # while all others (theta, l, w, h) remain the same
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
                                  [0,1,0,0,0,0,0,0,1,0],
                                  [0,0,1,0,0,0,0,0,0,1],
                                  [0,0,0,1,0,0,0,0,0,0],  
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,1]])     

        
            # The whole state is measured at each sampled and we assume constant velocity for the sweeps frames (during 1/6 seconds) 
            # measurement function, dim_z * dim_x, the first 9 (x,y,z,theta,w,l,h,vx,vy) dimensions of the measurement correspond to the state
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
                                  [0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,1]])

            ###################################################
            # Covariance matrices from Hsu-kuang Chiu : Probabilistic 3D Multi-Object Tracking for Autonomous Driving
            covariance = Covariance(covariance_id=3)    # nuScenes ID
            self.kf.P = covariance.P[self.tracking_name]
            self.kf.Q = covariance.Q[self.tracking_name]

            self.kf.P = self.kf.P[:-1,:-1] # not using angular velocity
            self.kf.Q = self.kf.Q[:-1,:-1] # not using angular velocity
            
            if use_R : self.kf.R = covariance.R[self.tracking_name] # Measurement noise (optional)
            ###################################################

            # initialize data
            self.kf.x[:7] = self.initial_pos.reshape((7, 1))
            self.kf.x[7:] = self.initial_vel.reshape((3, 1))

        else :
            # Not using velocity in the measurement matrix

            self.kf = KalmanFilter(dim_x=10, dim_z=7)       
            # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

            # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
            # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
            # while all others (theta, l, w, h) remain the same
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
                                  [0,1,0,0,0,0,0,0,1,0],
                                  [0,0,1,0,0,0,0,0,0,1],
                                  [0,0,0,1,0,0,0,0,0,0],  
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,1]])     

            # measurement function, dim_z * dim_x, the first 7 (x,y,z,theta,w,l,h) dimensions of the measurement correspond to the state
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
                                  [0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0]])

            ###################################################
            # Covariance matrices from Hsu-kuang Chiu : Probabilistic 3D Multi-Object Tracking for Autonomous Driving
            covariance = Covariance(covariance_id=2)    # nuScenes ID
            self.kf.P = covariance.P[self.tracking_name]
            self.kf.Q = covariance.Q[self.tracking_name]

            self.kf.P = self.kf.P[:-1,:-1] # not using angular velocity
            self.kf.Q = self.kf.Q[:-1,:-1] # not using angular velocity

            if use_R : self.kf.R = covariance.R[self.tracking_name] # Measurement noise (optional)
            ###################################################

            # initialize data
            self.kf.x[:7] = self.initial_pos.reshape((7, 1))




        # # measurement uncertainty, uncomment if measurement data cannot be trusted due to detection noise
        # # self.kf.R[0:,0:] *= 10.   

        # # initial state uncertainty at time 0
        # # Given a single data, the initial velocity is very uncertain, so give a high uncertainty to start
        # self.kf.P[7:, 7:] *= 1000.  
        # self.kf.P *= 10.

        # # process uncertainty, make the constant velocity part more certain
        # velocity_accuracy = 0.1/3.6 # Nuscenes datasheet specifies a velocity accuracy of 0.1 km/h
        # self.kf.Q = np.diag([0,0,0,0,0,0,0,velocity_accuracy**2,velocity_accuracy**2,velocity_accuracy**2])
        # # self.kf.Q[7:, 7:] *= 0.01
        

    def compute_innovation_matrix(self):
        ''' 
        compute the innovation matrix for association with mahalanobis distance
        '''
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    def get_velocity(self):
        # return the object velocity in the state

        return self.kf.x[7:]