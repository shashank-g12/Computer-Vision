import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os

patternSize = (7,5)

class OBJ:
	def __init__(self, filePath):
		self.objName = None
		self.vertices = []
		self.textcoords = []
		self.normals = []
		self.faces = []
		self.usemtl = False
		self.materials = {}
		self.faceToMaterial = {}
		self.mtlFileName = False
		self.readFile(filePath)
		self.mtlFilePath = "/".join(filePath.split('/')[:-1])+'/'+self.mtlFileName
		
		if self.usemtl and os.path.exists(self.mtlFilePath):
			self.readMtl(self.mtlFilePath)
	
	def readFile(self, filePath):
		curr = ''
		for line in open(filePath, 'r'):
			if line.startswith(('#', 's')): continue
			values = line.split()
			if not values: continue
			elif values[0] == 'o':
				self.objName = values[1]
			elif values[0] == 'v':
				self.vertices.append(list(map(float, values[1:])))
			elif values[0] == 'vt':
				self.textcoords.append(list(map(float, values[1:])))
			elif values[0] == 'vn':
				self.normals.append(list(map(float, values[1:])))
			elif values[0] == 'f':
				face_element= []
				for value in values[1:]:
					nums = list(value.split('/'))
					face_element.append(nums)
				self.faces.append(face_element)
				self.faceToMaterial[len(self.faces)-1] = curr
			elif values[0] == 'mtllib':
				self.usemtl=True
				self.mtlFileName=values[1]
			elif values[0]=='usemtl' and self.usemtl==True:
				self.materials[values[1]]={}
				curr=values[1]
		
	def readMtl(self, file):
		curr=''
		for line in open(file, 'r'):
			if line.startswith('#'):continue
			values = line.split()
			if not values: continue
			elif values[0] == 'newmtl':
				curr = values[1]
				continue
			self.materials[curr][values[0]] = list(map(float, values[1:]))

def findCorners(image):
	grayImage  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found, corners  = cv2.findChessboardCornersSB(grayImage, patternSize,  flags = cv2.CALIB_CB_NORMALIZE_IMAGE | \
											   cv2.CALIB_CB_EXHAUSTIVE |cv2.CALIB_CB_ACCURACY);
	return found, corners

def getPoints(image):
	num_points = patternSize[0]*patternSize[1]
	src_points = np.empty((num_points,2))
	dst_points = np.empty((num_points,2))
	found, corners = findCorners(image)
	if found==False:
		print('checkerboard pattern not recognized from the image')
		return
	for i in range(num_points):
		src_points[i] = np.array([float(i%7), float(i//7)])
		dst_points[i] = corners[i]
	return src_points, dst_points

def zhangsAlgorithm(img_list):
	def find_v(col1, col2):
		return np.array([col1[0]*col2[0], (col1[0]*col2[1]+col1[1]*col2[0]), \
						 col1[1]*col2[1], (col1[0]*col2[2]+col1[2]*col2[0]), \
						 (col1[2]*col2[1]+col1[1]*col2[2]), col1[2]*col2[2] ])
	
	def intrinsicParameters(b):
		K = np.zeros((3,3))
		K[2,2] = 1
		K[1][2] = (b[1]*b[3]-b[0]*b[4])/(b[0]*b[2]-b[1]**2)
		scale = b[5] - (b[3]**2 + K[1][2]*(b[1]*b[3]-b[0]*b[4]))/b[0]
		K[0][0] = math.sqrt(scale/b[0])
		K[1][1] = math.sqrt(scale*b[0]/(b[0]*b[2]-b[1]**2))
		K[0][1] = -b[1]*(K[0][0]**2)*K[1][1]/scale
		K[0][2] = (K[0][1]*K[1][2]/K[1][1])-(b[3]*(K[0][0]**2)/scale)
		return K
	
	homography = []
	for idx, img in enumerate(img_list):
		#get world points and img points
		src_points, dst_points = getPoints(img)
		h , _ = cv2.findHomography(src_points, dst_points, method = cv2.RANSAC)
		homography.append(h)
		
	#solve for omega (image of absolute conic)
	V = np.zeros((len(img_list)*2, 6))
	for i,h in enumerate(homography):
		V[i*2] = find_v(h[:,0], h[:, 1])
		V[i*2+1] = find_v(h[:,0], h[:,0]) - find_v(h[:,1], h[:,1])
	U,S,V_T = np.linalg.svd(np.dot(V.T, V))
	B = V_T[-1, :]
	return intrinsicParameters(B)

def getRotationAndTranslation(K, image):    
	src_points, dst_points = getPoints(image)
	h , _ = cv2.findHomography(src_points, dst_points, method = cv2.RANSAC)
	
	K_inv = np.linalg.inv(K)
	r1 = np.dot(K_inv, h[:, 0]).reshape(-1,1)
	scale = 1.0/np.linalg.norm(r1, 2)
	r1 *= scale
	r2 = scale*np.dot(K_inv, h[:, 1]).reshape(-1,1)
	r3 = np.cross(r1.squeeze(), r2.squeeze()).reshape(-1,1)
	t = scale*np.dot(K_inv, h[:, 2])
	R = np.concatenate((r1, r2, r3) , axis=1)
	U,S,V_T = np.linalg.svd(R)
	R = np.dot(U, V_T)
	return R, t

def render(image, obj, K, R, t , scale=1,  translate=np.array([2.0, 2.0, -1.0])):
	result = image.copy()
	for face in obj.faces:
		obj_points = []
		for vertices in face:
			vertex_idx = int(vertices[0])
			obj_points.append(obj.vertices[vertex_idx-1])
			
		obj_points = np.expand_dims(np.array(obj_points, dtype=np.float32)*scale + translate, axis=-1)
		img_points = cv2.projectPoints(obj_points, R, t, K, None)[0].squeeze()
		
		result = cv2.fillConvexPoly(result, np.int32(img_points), (0,159,189))
		result = cv2.polylines(result, [np.int32(img_points)], True , (221,255,255), 10)
	return result

def augmentedReality(idx, image, obj_file, K, scale=1, translate=np.array([2.0, 2.0, -1.0])):
	obj = OBJ(obj_file)
	R, t = getRotationAndTranslation(K, image)
	result = render(image, obj, K, R, t, scale=scale, translate=translate)
	plt.imshow(result)
	plt.savefig('./results/image_'+str(idx+1)+'.jpg')

def main():
	image_list = []
	for name in glob.glob('*.jpg'):
		image = cv2.imread(name)
		if image.shape[0]>2560 or image.shape[1]>2560:
			image = cv2.resize(image, (2560, 2560))
		image_list.append(image)
	K = zhangsAlgorithm(image_list)
	print('Intrinsic Parameters: \n', K)
	obj_file = 'cube.obj'
	for idx, image in enumerate(image_list):
		augmentedReality(idx, image, obj_file, K)



if __name__ == '__main__':
	main()
