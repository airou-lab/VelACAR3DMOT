# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

#-----------------------------------------------
# Modified by : Mathis Morales                       
# Email       : mathis-morales@outlook.fr             
# git         : https://github.com/MathisMM            
#-----------------------------------------------

import numpy as np, os, copy, math
from libs.box import Box3D
from pyquaternion import Quaternion

from libs.matching import data_association
from libs.kalman_filter import KF
# from libs.vis import vis_obj

# np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
	def __init__(self, args, cat, ID_init=0):                    

		# counter
		self.trackers = []
		self.frame_count = 0
		self.ID_count = [ID_init]
		self.id_now_output = []

		# config
		self.cat = cat
		self.affi_process = args.affi_pro	# post-processing affinity
		
		# argument-extracted values
		self.det_name = args.detection_method
		self.use_vel = args.use_vel
		self.args = args

		self.get_param(cat)
		self.print_param()

		# debug
		self.debug_id = None
		# self.debug_id = 1

	def get_param(self, cat):

		if self.args.run_hyper_exp:
			if self.det_name == 'CRN':
				if cat == 'car': 			algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'truck': 		algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'trailer': 		algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 3, 2 
				elif cat == 'bus': 			algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'motorcycle':	algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 3, 2 
				elif cat == 'bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', self.args.metric, self.args.thresh, 3, 2  
				else: assert False, 'cat name error: %s'%(cat)
			
			elif self.det_name == 'Radiant':
				if cat == 'car': 			algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'truck': 		algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'trailer': 		algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 3, 2 
				elif cat == 'bus': 			algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 1, 2 
				elif cat == 'motorcycle':	algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 3, 2 
				elif cat == 'bicycle': 		algm, metric, thres, min_hits, max_age = 'hungar', self.args.metric, self.args.thresh, 3, 2 
				else: assert False, 'cat name error: %s'%(cat)

			else: assert False, 'Error : Unknown detector %s'%(self.det_name)

		else :
			if self.det_name == 'CRN':	#Fine tuning done
				if cat == 'car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_2d', -0.3, 1, 2 
				elif cat == 'pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.3, 1, 2 
				elif cat == 'truck': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_2d',    4, 1, 2 
				elif cat == 'trailer': 		algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.8, 3, 2 
				elif cat == 'bus': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_2d', -0.7, 1, 2 
				elif cat == 'motorcycle':	algm, metric, thres, min_hits, max_age = 'hungar', 'giou_2d', -0.8, 3, 2 
				elif cat == 'bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'dist_2d',    4, 3, 2 
				else: assert False, 'cat name error: %s'%(cat)
			
			elif self.det_name == 'Radiant':
				if cat == 'car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.5, 1, 2
				elif cat == 'truck': 		algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'trailer': 		algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.3, 3, 2
				elif cat == 'bus': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'motorcycle':	algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.7, 3, 2
				elif cat == 'bicycle': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d',    6, 3, 2
				else: assert False, 'cat name error: %s'%(cat)
			
			elif self.det_name == 'GT':
				if cat == 'car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.5, 1, 2
				elif cat == 'truck': 		algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'trailer': 		algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.3, 3, 2
				elif cat == 'bus': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.4, 1, 2
				elif cat == 'motorcycle':	algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.7, 3, 2
				elif cat == 'bicycle': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d',    6, 3, 2
				else: assert False, 'cat name error: %s'%(cat)

			else: assert False, 'Error : Unknown detector %s'%(self.det_name)

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1	
		self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
			algm, metric, thres, max_age, min_hits

		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

	def print_param(self):
		print('\n\n***************** Parameters for %s *********************' % self.cat)
		print('matching algorithm is %s' % self.algm)
		print('distance metric is %s' % self.metric)
		print('distance threshold is %f' % self.thres)
		print('min hits is %f' % self.min_hits)
		print('max age is %f' % self.max_age)
		# print('ego motion compensation is %d' % self.ego_com)

	def format_dets_df (self,dets_df):
		'''
		Takes a dataframe in the format : [t,x,y,z,w,l,h,r1,r2,r3,r4,vx,vy,vz,score,token]
		and return formatted_df in the format : [x,y,z,w,l,h,theta,score] for associations
		and a info_df in the format : [vx,vy,vz,r1,r2,r3,r4,score,token] containing the remaining informations
		'''
		# print (dets_df)
		# print (100*'#')
		formatted_df = copy.deepcopy(dets_df)
		info_df = copy.deepcopy(dets_df)

		formatted_df = formatted_df.drop(['t','vx','vy','vz','score','token'],axis=1)		#[x,y,z,w,l,h,r1,r2,r3,r4]
		info_df = info_df.drop(['x','y','z','w','l','h',],axis=1)							#[t,r1,r2,r3,r4,vx,vy,vz,score,token]
		info_df = info_df.loc[:,['vx','vy','vz','r1','r2','r3','r4','score','token','t']]	#[vx,vy,vz,r1,r2,r3,r4,score,token,t]

		theta_list = []
		for i in formatted_df.index:
			# r1-4 are quaternions in the form : qw,qx,qy,qz
			
			qw = formatted_df.iloc[i]['r1']
			qx = formatted_df.iloc[i]['r2']
			qy = formatted_df.iloc[i]['r3']
			qz = formatted_df.iloc[i]['r4']
			# quaternion = [qw,qx,qy,qz]
			quaternion = Quaternion(qw,qx,qy,qz)
			yaw_angle = quaternion.radians

			# Project into xy plane.
			# rot_mat = np.dot(quaternion.rotation_matrix, np.array([1, 0, 0]))

		    # Measure yaw using arctan.
			# yaw_angle = np.arctan2(rot_mat[1], rot_mat[0])

			# numerator = 2 * (qx*qy + qw*qz)
			# denominator = 1 - 2*(qy*qy + qz*qz)

			# yaw_angle = np.arctan2(numerator, denominator)

			theta_list.append(yaw_angle) 


		formatted_df.insert(6,'theta',theta_list)
		formatted_df=formatted_df.drop(['r1','r2','r3','r4'],axis=1)		#[x,y,z,w,l,h,theta]

		return formatted_df, info_df.to_numpy()

	def process_dets(self, dets_df):
		# convert each detection into the class Box3D 
		# inputs: 
		# 	dets_df - a pandas dataframe of detections in the format [x,y,z,w,l,h,theta]

		dets = dets_df.to_numpy()	# convert df to numpy
		dets_new = []
		for det in dets:
			det_tmp = Box3D.array2bbox_raw(det)
			dets_new.append(det_tmp)

		return dets_new

	def within_range(self, theta):
		# make sure the orientation is within a proper range

		if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
		if theta < -np.pi: theta += np.pi * 2

		return theta

	def orientation_correction(self, theta_pre, theta_obs):
		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		
		# make the theta still in the range
		theta_pre = self.within_range(theta_pre)
		theta_obs = self.within_range(theta_obs)

		# if the angle of two theta is not acute angle, then make it acute
		if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
			theta_pre += np.pi       
			theta_pre = self.within_range(theta_pre)

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
			if theta_obs > 0: theta_pre += np.pi * 2
			else: theta_pre -= np.pi * 2

		return theta_pre, theta_obs

	def prediction(self):
		# get predicted locations from existing tracks

		trks = []
		for t in range(len(self.trackers)):
			
			# propagate locations
			kf_tmp = self.trackers[t]
			if kf_tmp.id == self.debug_id:
				print('\n before prediction')
				print(kf_tmp.kf.x.reshape((-1)))
				print('\n current velocity')
				print(kf_tmp.get_velocity())
			kf_tmp.kf.predict()
			if kf_tmp.id == self.debug_id:
				print('After prediction')
				print(kf_tmp.kf.x.reshape((-1)))
			kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

			# update statistics	
			kf_tmp.time_since_update += 1 		
			trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
			trks.append(Box3D.array2bbox(trk_tmp))

		return trks

	def update(self, matched, unmatched_trks, dets, info):
		# update matched trackers with assigned detections
		
		dets = copy.copy(dets)
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
				assert len(d) == 1, 'error : more than one association'

				# update statistics
				trk.time_since_update = 0		# reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				bbox3d = Box3D.bbox2array(dets[d[0]])
				vel = info[d, :3]
				trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

				if trk.id == self.debug_id:
					print('After ego-compensation')
					print(trk.kf.x.reshape((-1)))
					print('matched measurement')
					print(bbox3d.reshape((-1)))
					print('uncertainty')
					print(trk.kf.P)
					print('measurement noise')
					print(trk.kf.R)
					print('\n previous velocity')
					print(trk.get_velocity())

				# kalman filter update with observation
				if self.use_vel:
					# augmenting box with velocity information
					vel = vel.reshape(3,)
					bbox3d = np.concatenate([bbox3d, vel], axis=0)

				trk.kf.update(bbox3d)
				# trk.kf.x[7:,0]=vel

				if trk.id == self.debug_id:
					print('after matching')
					print(trk.kf.x.reshape((-1)))
					print('\n current velocity')
					print(trk.get_velocity())

				trk.kf.x[3] = self.within_range(trk.kf.x[3])
				trk.info = info[d, 3:][0]	#r1:r4, score, token, t

			# debug use only
			# else:
				# print('track ID %d is not matched' % trk.id)

	def birth(self, dets, info, unmatched_dets):
		# create and initialise new trackers for unmatched detections

		dets = copy.deepcopy(dets)
		new_id_list = list()					# new ID generated for unmatched detections

		for i in unmatched_dets:        			# a scalar of index
			trk = KF(Box3D.bbox2array(dets[i]), info[i, :3], info[i, 3:], self.ID_count[0], self.cat, self.use_vel, self.args.use_R)	# x,y,z,theta,l,w,h | vx,vy,vz | r1,r2,r3,r4,score,token,t
			self.trackers.append(trk)
			new_id_list.append(trk.id)
			# print('track ID %s has been initialized due to new detection' % trk.id)
			if self.args.verbose >= 1: print('birth of tracklet',trk.id)
			self.ID_count[0] += 1

		return new_id_list

	def output(self):
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

		num_trks = len(self.trackers)
		results = []
		for trk in reversed(self.trackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)
			vx,vy = trk.kf.x[7:9]
			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				results.append(np.concatenate((d, vx, vy, [trk.id], trk.info)).reshape(1, -1)) 	#[h,w,l,x,y,z,theta,vx,vy,ID,quaternions,score,token,t]
			num_trks -= 1

			# death, remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(num_trks)
				if self.args.verbose >= 1: print('Death of tracklet',trk.id)

		return results

	def process_affi(self, affi, matched, unmatched_dets, new_id_list):

		# post-processing affinity matrix, convert from affinity between raw detection and past total tracklets
		# to affinity between past "active" tracklets and current active output tracklets, so that we can know 
		# how certain the results of matching is. The approach is to find the correspondes of ID for each row and
		# each column, map to the actual ID in the output trks, then purmute/expand the original affinity matrix
		
		###### determine the ID for each past track
		trk_id = self.id_past 			# ID in the trks for matching

		###### determine the ID for each current detection
		det_id = [-1 for _ in range(affi.shape[0])]		# initialization

		# assign ID to each detection if it is matched to a track
		for match_tmp in matched:		
			det_id[match_tmp[0]] = trk_id[match_tmp[1]]

		# assign the new birth ID to each unmatched detection
		count = 0
		assert len(unmatched_dets) == len(new_id_list), 'error'
		for unmatch_tmp in unmatched_dets:
			det_id[unmatch_tmp] = new_id_list[count] 	# new_id_list is in the same order as unmatched_dets
			count += 1
		assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

		############################ update the affinity matrix based on the ID matching
		
		# transpose so that now row is past trks, col is current dets	
		affi = affi.transpose() 			

		###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
		permute_row = list()
		for output_id_tmp in self.id_past_output:
			index = trk_id.index(output_id_tmp)
			permute_row.append(index)
		affi = affi[permute_row, :]	
		assert affi.shape[0] == len(self.id_past_output), 'error'

		###### compute the permutation for columns (current tracklets), possible to delete and add new rows
		# addition can be because some tracklets propagated from previous frames with no detection matched
		# so they are not contained in the original detection columns of affinity matrix, deletion can happen
		# because some detections are not matched

		max_index = affi.shape[1]
		permute_col = list()
		to_fill_col, to_fill_id = list(), list() 		# append new columns at the end, also remember the ID for the added ones
		for output_id_tmp in self.id_now_output:
			try:
				index = det_id.index(output_id_tmp)
			except:		# some output ID does not exist in the detections but rather predicted by KF
				index = max_index
				max_index += 1
				to_fill_col.append(index); to_fill_id.append(output_id_tmp)
			permute_col.append(index)

		# expand the affinity matrix with newly added columns
		append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
		append.fill(self.min_sim)
		affi = np.concatenate([affi, append], axis=1)

		# find out the correct permutation for the newly added columns of ID
		for count in range(len(to_fill_col)):
			fill_col = to_fill_col[count]
			fill_id = to_fill_id[count]
			row_index = self.id_past_output.index(fill_id)

			# construct one hot vector because it is proapgated from previous tracks, so 100% matching
			affi[row_index, fill_col] = self.max_sim		
		affi = affi[:, permute_col]

		return affi

	def compute_confidance_score(self, results, affi, matched):
		'''
		Tracking confidance score is the affinity score between the prediction and the detection
		i.e : how accurate the prediction was for this object.
		In case the detection is a new object, it makes sense to keep the detection score as the tracking score (default value)
		In case it's an existing tracked object, we replace the detection score by the affinity score.
		'''

		if len(matched)>0:
			for d,t in matched:
				for res in results[0]:
					if res[9]-1==t : 
						res[14]=affi[d][t]
			
						if self.metric in ['dist_3d', 'dist_2d', 'm_dis']:
							res[14]=-res[14]


		return results

	def track(self, dets_all, frame_number, scene_name, verbose):
		"""
		Params:
		  	dets_all: dataframe
			frame_number:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = self.format_dets_df(dets_all)
		# dets - a dataframe of detections in the format ['x','y','z','w','l','h','theta']
		# 		info: a array of other info for each det [vx,vy,vz,r1,r2,r3,r4,score,token,t]
	
		if self.debug_id: print('\nframe is %s' % frame_number)
	
		# logging
		if verbose>=1: print('\n\n*****************************************\n\nprocessing %s/frame %d' % (scene_name, frame_number))

		self.frame_count += 1

		# recall the last frames of outputs for computing ID correspondences during affinity processing
		self.id_past_output = copy.copy(self.id_now_output)
		self.id_past = [trk.id for trk in self.trackers]

		# process detection format
		dets = self.process_dets(dets)

		# tracks propagation based on velocity
		trks = self.prediction()

		# matching
		trk_innovation_matrix = None
		if self.metric == 'm_dis':
			trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 
		
		matched, unmatched_dets, unmatched_trks, cost, affi = \
			data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

		if verbose>=2:
			print('\ndetections are')
			for i, det in enumerate(dets):
				print('index:', i, det)
			print('\ntracklets are')
			for i, trk in enumerate(trks):
				print('index:', i, trk)

		if verbose>=3:
			print('matched indexes are')
			print(matched)
			print('unmatched indexes are')
			print(unmatched_dets)
			print('unmatched tracklets are')
			print(unmatched_trks)
			print('raw affinity matrix is')
			print(affi)

		# update trks with matched detection measurement
		self.update(matched, unmatched_trks, dets, info)
		
		# create and initialise new trackers for unmatched detections
		new_id_list = self.birth(dets, info, unmatched_dets)

		if verbose>=3:
			print('new ID list is')
			print(new_id_list)

		# output existing valid tracks
		results = self.output()
		if len(results) > 0: results = [np.concatenate(results)]		# h,w,l, x,y,z, theta, vx,vy, ID, r1:r4, confidence, token, t
		else:            	 results = [np.empty((0, 15))]

		results = self.compute_confidance_score(results, affi, matched)	# replacing confidance score for active tracks

		self.id_now_output = results[0][:, 9].tolist()					# only the active tracks that are outputed

		# post-processing affinity to convert to the affinity between resulting tracklets
		if self.affi_process:
			affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
			if verbose>=4:
				print('processed affinity matrix is')
				print(affi)

		# logging
		if verbose>=4:
			print('\ntop-1 cost selected')
			print(cost)
			for result_index in range(len(results)):
				print(results[result_index][:, :8])
				print('')

		return results[0], affi
