import numpy as np
import cv2

# Data augmentation methods for helpng the model generalize to other tracks
# Reference: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

def flip_img(img, steer_angle):
	"""
	Generate image flipped along vertical axis, and corresponding steering angle (inverted)
	"""
	img = np.fliplr(img)
	steer_angle = - steer_angle

	return img, steer_angle

def brightness_aug(img):
	"""
	Randomly adjusts brightness of image to emulate different daylight conditions
	"""
	# HSV (Hue, Saturation, Value) where 'Value' corresponds to 'Brightness'
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	bright_ratio = np.random.uniform(0.25,0.50)
	img_hsv[:,:,2] = img_hsv[:,:,2] * bright_ratio
	img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

	return img

def transl_img(img,steer_angle,x_lim, y_lim):
	"""
	Randomly translate image in x and y directions with limits specified by the range_x and range_y parameters
	Vertical translation emulates sloped surface, ho
	Steering angle adjusted by 0.004 per pixel translation in x-direction
	"""
	tx = x_lim * (np.random.uniform(-1,1))
	ty = y_lim * (np.random.uniform(-1,1))
	M = np.float32([ [1, 0, tx], [0, 1, ty] ])
	
	img = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
	steer_angle += tx*0.004

	return img, steer_angle

def shadow_aug(img):
	"""
	Generate and cast random shadow across the entire width of the image
	"""

	# Randomly choosing coordinates for a point each on the left and right edges of the image
	(x_left, x_right) = (0, img.shape[1])
	(y_left, y_right) = (img.shape[0] * np.random.uniform(0,0.8), img.shape[0] * np.random.uniform(0,0.8))

	# Generating indices for all points in the image
	y_im, x_im = np.mgrid[0:img.shape[0],0:img.shape[1]]

	# Logic for setting points lying on and above the line (formed by the 2 randomly selected points) to 1 and the rest to 0
	# (y_im-y_left)/(x_im-x_left) - (y_right-y_left)/(x_right-x_left) >= 0 for points lying on and above the line
	# Expressed as products to avoid zero-division error
	shadow_mask = np.zeros_like(img[:, :, 1])
	shadow_mask[( (y_im-y_left)*(x_right-x_left) - (y_right-y_left)*(x_im-x_left) >= 0 )] = 1

	# Choosing region below the line to shade and choose lightness ratio for this region for emulating a shadow
	shade_inds = (shadow_mask == 1)
	bright_ratio = np.random.uniform(0.25, 0.50)

	# Casting shadow by adjusting the brightness of the indices lying in the shaded region in HSV(Hue, Saturation, Value) image
	img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	img_hsv[:,:,2][shade_inds] = img_hsv[:,:,2][shade_inds] * bright_ratio
	img = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)

	return img