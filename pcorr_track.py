import sys, time, os, math
from common import clock, draw_str
# import the necessary things for OpenCV
import cv2
import cv2.cv as cv
import numpy as np
#import action.segment as aseg

HAVE_CV = True
DSF = 2
STRIDE = 6
MAXDIST = 100.0

# fps, grid_divs_x, grid_divs_y, stride, mode, display

def _process_phase_corr(**kwargs):

	if not HAVE_CV:
		print "WARNING: You must install OpenCV in order to analyze or view!"
		return
	
	ap = kwargs
	ap = {'fps':24, 'grid_divs_x':8, 'grid_divs_y':8, 'stride':STRIDE, 'mode':'analysis', 'display': True}
	
#	fstring = '~/dev/git/synaesthesia/films/Synae_test_v2.mov'
 	fstring = '~/Movies/action/JoggersRaining/JoggersRaining.mov'
# 	fstring = '~/Pictures/iPhoto Library/Masters/2012/05/31/20120531-210318/IMG_0140.MOV'
	
	movie_path = ap['movie_path'] = os.path.expanduser(fstring)
	
	data_path = ap['data_path'] = os.path.expanduser('~/dev/git/synaesthesia/films/Synae_test_v2.pkl')
	verbose = True
	
	if (ap['movie_path'] is None) or (ap['data_path'] is None):
		print "ERROR: Must supply both a movie and a data path!"
		return	
	
	capture = cv2.VideoCapture(ap['movie_path'])
	
	fps = ap['fps']
	grid_x_divs = ap['grid_divs_x']
	grid_y_divs = ap['grid_divs_y']
	frame_width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	frame_height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	frame_size = (frame_width, frame_height)
	grid_width = int((frame_width/grid_x_divs)/DSF)
	grid_height = int((frame_height/grid_y_divs)/DSF)
	grid_size = (grid_width, grid_height)
	
	centers_x = range((frame_width/16/DSF),(frame_width/DSF),(frame_width/8/DSF))
	centers_y = range((frame_height/16/DSF),(frame_height/DSF),(frame_height/8/DSF))
	
	if verbose:
		print fps, ' | ', frame_size, ' | ', grid_size
	
	# container for prev. frame's grayscale subframes
	prev_sub_grays, prev_grid_xvals, prev_grid_yvals = [], [0 for i in range(64)], [0 for i in range(64)]
	
	
	# last but not least, get total_frame_count and set up the memmapped file
	dur_total_secs = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT) / fps)
	stride_frames = ap['stride']
	# if ap['duration'] < 0:
	dur_secs = dur_total_secs
	# else:
	#	dur_secs = ap['duration']
	
	offset_secs = 0 # min(max(ap['offset'], 0), dur_total_secs)
	dur_secs = min(max(dur_secs, 0), (dur_total_secs - offset_secs))
	offset_strides = int(offset_secs * (fps / stride_frames))
	dur_strides = int(dur_secs * (fps / stride_frames))
	offset_frames = offset_strides * stride_frames
	dur_frames = dur_strides * stride_frames
			
	if verbose:
		print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
		print 'FRAMES: ', int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
		print 'DUR TOTAL: ', dur_total_secs
		print "OFFSET (SECONDS): ", offset_secs
		print "OFFSET (STRIDES): ", offset_strides
		print "OFFSET (FRAMES): ", offset_frames
		print "DUR (SECONDS): ", dur_secs
		print 'DUR (STRIDES): ', dur_strides
		print 'DUR (FRAMES): ', dur_frames
		print 'XDIVS: ', grid_x_divs
		print 'YDIVS: ', grid_y_divs
		print "FPS: ", fps
		print "stride_frames: ", stride_frames
	
	# set up memmap
	if ap['mode'] == 'playback' and ap['display'] == True:
		fp = np.memmap(data_path, dtype='float32', mode='r+', shape=((offset_strides + dur_strides),64,2))
	else:
		fp = np.memmap(data_path, dtype='float32', mode='w+', shape=(dur_strides,64,2))
	
	# set some drawing constants
	frame_idx = offset_frames
	end_frame = offset_frames + dur_frames
	
	if ap['display']:
		cv.NamedWindow('Image', cv.CV_WINDOW_AUTOSIZE)

	capture.set(cv.CV_CAP_PROP_POS_FRAMES, offset_frames)
		
	fhann = cv2.createHanningWindow((frame_width/DSF,frame_height/DSF), cv2.CV_32FC1)
	ghann = cv2.createHanningWindow((grid_width/DSF,grid_height/DSF), cv2.CV_32FC1)
	
	# load arrays with default vals
	for row in range(grid_y_divs):
		for col in range(grid_x_divs):
			#prev_sub_grays += [np.float32(frame_gray_pyr[(row*grid_height):((row+1)*grid_height), (col*grid_width):((col+1)*grid_width)])]
# 			sub_grays += [np.zeros_like(ghann, dtype=np.int8)]
			prev_sub_grays += [np.zeros_like(ghann, dtype=np.int8)]
			
			
			xval = int(0+centers_x[col])
			yval = int(0+centers_y[row])
			
			frame_gray_pyr = np.zeros_like(fhann, dtype=np.int8)
			prev_frame_gray_pyr = np.zeros_like(fhann, dtype=np.int8)
	
	ds_grid_height = grid_height/DSF
	ds_grid_width = grid_width/DSF

	while frame_idx < end_frame:
		
		if ((frame_idx % stride_frames) == 0):
			t = clock()
			print '1. fr. idx: ', frame_idx, ' (', frame_idx / float(end_frame), ' | ', end_frame, ')'

			# grab next frame
			ret, frame = capture.read()
			dt = clock() - t
			print 'time: %.1f ms' % (dt*1000)
			t = clock()
			if frame is None: 
				print 'Frame error! Exiting...'
				break # no image captured... end the processing
			prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			if DSF > 1:
				prev_frame_gray_pyr = cv2.pyrDown(prev_frame_gray)
			else:
				prev_frame_gray_pyr = prev_frame_gray #[:]
			
			# grid stage (prev frame)
			for row in range(grid_y_divs):
				for col in range(grid_x_divs):
					cell = ((row*grid_x_divs)+col)
					prev_sub_grays[(row*grid_x_divs)+col] = prev_frame_gray_pyr[(row*ds_grid_height):((row+1)*ds_grid_height), (col*ds_grid_width):((col+1)*ds_grid_width)]
			dt = clock() - t
			print '2. time: %.1f ms' % (dt*1000)
			
		if (frame_idx % stride_frames) == 1: 
			# grab next frame
			t = clock()
			ret, frame = capture.read()
			if frame is None: 
				print 'Frame error! Exiting...'
				break # no image captured... end the processing
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			if DSF > 1:
				frame_gray_pyr = cv2.pyrDown(frame_gray)
			else:
				frame_gray_pyr = frame_gray #[:]

			
			# grid stage (full)
			for row in range(grid_y_divs):
				for col in range(grid_x_divs):
					if ap['mode'] == 'playback':
						cell = ((row*grid_x_divs)+col)
						gres = fp[(frame_idx/stride_frames)][cell]
					else:
# 						print '--grid-----------'
# 						print (row*ds_grid_height)
# 						print (col*ds_grid_width)
# 						print (row,col)
						sub_gray = np.float32(frame_gray_pyr[(row*ds_grid_height):((row+1)*ds_grid_height), (col*ds_grid_width):((col+1)*ds_grid_width)])
# 						print sub_gray.shape
# 						print np.float32(prev_sub_grays[(row*grid_x_divs)+col]).shape
# 						print ghann.shape
						gres = cv2.phaseCorrelateRes(np.float32(prev_sub_grays[(row*grid_x_divs)+col]), sub_gray, ghann)
						
						dflag = 0
						if abs(gres[1]) > 0.9:
							fp[(frame_idx/stride_frames)][(row*grid_x_divs)+col] = [(gres[0][0]/grid_width),(gres[0][1]/grid_height)]
							dflag = 1
						else:
							fp[(frame_idx/stride_frames)][(row*grid_x_divs)+col] = [0,0]

					if ap['display'] == True and dflag > 0:
					
						xvec = gres[0][0]*1000.0
						yvec = gres[0][1]*1000.0
						
						 #sqrt preserving signs:
						if xvec < 0:
							xval = int((-1*((-1*xvec) ** 0.5))+centers_x[col])
						else:
							xval = int((xvec ** 0.5)+centers_x[col])
						if yvec < 0:
							yval = int((-1*((-1*yvec) ** 0.5))+centers_y[row])
						else:
							yval = int((yvec ** 0.5)+centers_y[row])

# 						xval = int(xvec)+centers_x[col]
# 						yval = int(yvec)+centers_y[row]
						#delta = ((xval - prev_grid_xvals[(row*grid_x_divs)+col]) ** 2.0) + ((yval - prev_grid_yvals[(row*grid_x_divs)+col]) ** 2.0)
					
						#print delta
						#if delta < MAXDIST:
						cv2.line(frame_gray_pyr, (centers_x[col], centers_y[row]), (xval, yval), (255,255,255))

						#prev_grid_xvals[(row*grid_x_divs)+col] = xval
						#prev_grid_yvals[(row*grid_x_divs)+col] = yval
			print 
			dt = clock() - t
			print 'time: %.1f ms' % (dt*1000)
			
			#### SHOW
			if ap['display']:
				print "show"
				cv.ShowImage('Image', cv.fromarray(frame_gray_pyr))
			fp.flush()
		
		if ((frame_idx % stride_frames) == 0):
			frame_idx += 1
		elif (frame_idx % stride_frames) == 1:
			frame_idx += 5
		capture.set(cv.CV_CAP_PROP_POS_FRAMES, frame_idx)
		
# 		print "fr. idx.: ", frame_idx
		# prev_frame_gray_pyr = np.float32(frame_gray_pyr[:])
		
		# handle events for abort
		k = cv2.waitKey (1)
		if k % 0x100 == 27:
			# user has press the ESC key, so exit
				break
	
	del fp
	if ap['display']:
		cv2.destroyAllWindows()

