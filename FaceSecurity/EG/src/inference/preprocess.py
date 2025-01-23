import numpy as np
import cv2
from PIL import Image
import sys
from tqdm import tqdm

def extract_frames(filename,num_frames,model,image_size=(380,380)):
	cap_org = cv2.VideoCapture(filename)
	
	if not cap_org.isOpened():
		print(f'Cannot open: {filename}')
		# sys.exit()
		return []
	
	croppedfaces=[]
	idx_list=[]
	frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
	
	frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
	for cnt_frame in range(frame_count_org): 
		ret_org, frame_org = cap_org.read()
		height,width=frame_org.shape[:-1]
		if not ret_org:
			tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(filename)))
			break
		
		if cnt_frame not in frame_idxs:
			continue

		frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
		
		faces = model.predict_jsons(frame)
		try:
			if len(faces)==0:
				tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(filename)))
				continue

			size_list=[]
			croppedfaces_temp=[]
			idx_list_temp=[]
			
			for face_idx in range(len(faces)):
				x0,y0,x1,y1=faces[face_idx]['bbox']
				bbox=np.array([[x0,y0],[x1,y1]])
				croppedfaces_temp.append(cv2.resize(crop_face(frame,None,bbox,False,crop_by_bbox=True,only_img=True,phase='test'),dsize=image_size).transpose((2,0,1)))
				idx_list_temp.append(cnt_frame)
				size_list.append((x1-x0)*(y1-y0))
			
			max_size=max(size_list)
			croppedfaces_temp=[f for face_idx,f in enumerate(croppedfaces_temp) if size_list[face_idx]>=max_size/2]
			idx_list_temp=[f for face_idx,f in enumerate(idx_list_temp) if size_list[face_idx]>=max_size/2]
			croppedfaces+=croppedfaces_temp
			idx_list+=idx_list_temp	
		except Exception as e:
			print(f'error in {cnt_frame}:{filename}')
			print(e)
			continue
	cap_org.release()

	

	return croppedfaces,idx_list

def extract_face(frame,model,image_size=(380,380)):
	
	
	faces = model.predict_jsons(frame)

	if len(faces)==0:
		print('No face is detected' )
		return []

	croppedfaces=[]
	for face_idx in range(len(faces)):
		x0,y0,x1,y1=faces[face_idx]['bbox']
		bbox=np.array([[x0,y0],[x1,y1]])
		croppedfaces.append(cv2.resize(crop_face(frame,None,bbox,False,crop_by_bbox=True,only_img=True,phase='test'),dsize=image_size).transpose((2,0,1)))
	
	return croppedfaces


def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
	assert phase in ['train','val','test']

	#crop face------------------------------------------
	H,W=len(img),len(img[0])

	assert landmark is not None or bbox is not None

	H,W=len(img),len(img[0])
	
	if crop_by_bbox:
		x0,y0=bbox[0]
		x1,y1=bbox[1]
		w=x1-x0
		h=y1-y0
		w0_margin=w/4#0#np.random.rand()*(w/8)
		w1_margin=w/4
		h0_margin=h/4#0#np.random.rand()*(h/5)
		h1_margin=h/4
	else:
		x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
		x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
		w=x1-x0
		h=y1-y0
		w0_margin=w/8#0#np.random.rand()*(w/8)
		w1_margin=w/8
		h0_margin=h/2#0#np.random.rand()*(h/5)
		h1_margin=h/5

	

	if margin:
		w0_margin*=4
		w1_margin*=4
		h0_margin*=2
		h1_margin*=2
	elif phase=='train':
		w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
	else:
		w0_margin*=0.5
		w1_margin*=0.5
		h0_margin*=0.5
		h1_margin*=0.5
			
	y0_new=max(0,int(y0-h0_margin))
	y1_new=min(H,int(y1+h1_margin)+1)
	x0_new=max(0,int(x0-w0_margin))
	x1_new=min(W,int(x1+w1_margin)+1)
	
	img_cropped=img[y0_new:y1_new,x0_new:x1_new]
	if landmark is not None:
		landmark_cropped=np.zeros_like(landmark)
		for i,(p,q) in enumerate(landmark):
			landmark_cropped[i]=[p-x0_new,q-y0_new]
	else:
		landmark_cropped=None
	if bbox is not None:
		bbox_cropped=np.zeros_like(bbox)
		for i,(p,q) in enumerate(bbox):
			bbox_cropped[i]=[p-x0_new,q-y0_new]
	else:
		bbox_cropped=None

	if only_img:
		return img_cropped
	if abs_coord:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
	else:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)