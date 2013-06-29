#/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv, cv2, video
import math
from common import anorm2, draw_str
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
QPI = math.pi / 4.0
COLS = 8 * 8 * 8


class App:
	def __init__(self, video_src):
		self.track_len = 10
		self.detect_interval = 1
		self.tracks = []
	
		self.capture = cv2.VideoCapture(video_src)
	
		self.frame_width = int(self.capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
		self.frame_height = int(self.capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
		self.frame_size = (self.frame_width, self.frame_height)
		self.grid_width = int(self.frame_width/8)
		self.grid_height = int(self.frame_height/8)
		self.grid_size = (self.grid_width, self.grid_height)
		self.total_frame_count = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))	
		print self.frame_size
		print self.grid_size
		print self.total_frame_count
		
		self.data_path = str(video_src) + ".oflw"
		self.fp = np.memmap(self.data_path, dtype='float32', mode='w+', shape=(self.total_frame_count,512))

		self.cam = video.create_capture(video_src)
		self.frame_idx = 0
	

	def run(self):
		while True:
			ret, frame = self.cam.read()
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			vis = frame.copy()

			if len(self.tracks) > 0:
				img0, img1 = self.prev_gray, frame_gray
				p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				good = d < 1
				new_tracks = []
				for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					tr.append((x, y))
					if len(tr) > self.track_len:
						del tr[0]
					new_tracks.append(tr)
					cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
				self.tracks = new_tracks
				cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
				draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

			if self.frame_idx % self.detect_interval == 0:
				mask = np.zeros_like(frame_gray)
				mask[:] = 255
				for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
					cv2.circle(mask, (x, y), 5, 0, -1)
				p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						self.tracks.append([(x, y)])
				tracks_now = [np.float32(tr) for tr in self.tracks]
				try:
					seq = tracks_now[0]
					tdepth = self.track_len
					track_vectors = np.append( tracks_now[0][ min(len(seq)-1, tdepth) ], seq[0])

					for n, tr in enumerate(tracks_now[1:]):
						track_vectors = np.append(track_vectors, np.append(tr[ min(len(tr)-1, tdepth) ], (tr[0])))

					# track_vectors is variable-length, up to max-number-of-corners
					track_vectors = np.reshape(np.array(track_vectors, dtype='float32'), (-1,4))
					
					xdelta = track_vectors[:,2] - track_vectors[:,0]
					ydelta = track_vectors[:,3] - track_vectors[:,1]
					thetas = np.arctan2(ydelta, xdelta)
					mags = np.sqrt(np.add(np.power(xdelta, 2.0), np.power(ydelta, 2.0)))
					
					xbin = np.floor(track_vectors[:,0] / self.grid_width)
					ybin = np.floor(track_vectors[:,1] / self.grid_height)
					
					# print [np.min(xbin), np.max(xbin), np.min(ybin), np.max(ybin)]
					
					# filter out vectors less than 1 pixel!
				 	weighted = np.where(mags > 1, mags, 0.0)
					theta_bins = np.floor_divide(np.add(thetas, math.pi), QPI)
# 					ctheta_bins = np.floor_divide(np.add(cthetas, math.pi), QPI)
									
					combo_bins = ((np.add(np.multiply(ybin, 4), xbin) * 8) + theta_bins)
					
					if combo_bins.shape[0] > 0:
				 
# 				 		print combo_bins.shape
# 						print np.max(combo_bins)
						bins_histo, bin_edges = np.histogram(combo_bins, COLS, weights=weighted)
# 						print "-----------------------------------------------"
# 						print bins_histo
# 						print self.frame_idx
# 						print "-----------------------------------------------"
	

						self.fp[self.frame_idx] = bins_histo
					else:

						if verbose: print 'Zero! frame: ', self.frame_idx
						# fp[self.frame_idx] = np.zeros(512, dtype='float32') # 16*8=128

				except IndexError:
					print 'Index Error! frame: ', self.frame_idx
				# 
				# 					fd6 = (self.frame_idx/6) - offset_strides
				# 					if verbose: print 'Index Error! frame: ', fd6
				# 					fp[fd6] = np.zeros(128, dtype='float32') # 16*8=128
			
			self.frame_idx += 1
			self.prev_gray = frame_gray
			cv2.imshow('lk_track', vis)

			ch = 0xFF & cv2.waitKey(1)
			if ch == 27:
				break

def main():
	import sys
	try: video_src = sys.argv[1]
	except: video_src = 0
	
	print __doc__
	App(video_src).run()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
