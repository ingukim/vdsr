import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
from PSNR import psnr
import scipy.io
import pickle
from MODEL import model
#from MODEL_FACTORIZED import model_factorized
import time
DATA_PATH = "./data/test/"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

def get_img_list(data_path): #data_path로 ./data/test/Set14 들어옴
	l = glob.glob(os.path.join(data_path,"*"))   # (L)l 은 ./data/test//Set14 에 있는 전체 파일 가짐 glob.glob 를 통해 특정 디렉토리(폴더)로 부터 파일 목록 뽑아옴    os.path.join ->경로를 병합하여 새 경로 생성
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l: #f는 ./data/test//Set14에 0~하는 경로
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat", 2]) #f[-4:] -> .mat 까지만
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat", 3]) #f[:-4] -> ./data/test/Set14/* 까지만
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat", 4])    
	return train_list #train_list[*]에는 ['./data/test//Set14//0.mat', './data/test//Set14//0_2.mat', 2]  3개의 info 저장됨

def get_test_image(test_list, offset, batch_size): #test_list는 img_list정보(3개), offset은 몇번째 , batch_size는 한번에 1개의 사진씩 처리하는 의미
	target_list = test_list[offset:offset+batch_size] #offset=0 , batch_size=1 , target_list=test_list[0:1]->test_list의 3개 정보를 target_list에 1개씩 저장
	input_list = []
	gt_list = []
	scale_list = []
	for pair in target_list:
		print (pair[1]) #./data/test/Set14/*_2~4 (2,3,4)
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if "img_2" in mat_dict: input_img = mat_dict["img_2"]           
		elif "img_3" in mat_dict: input_img = mat_dict["img_3"]
		elif "img_4" in mat_dict: input_img = mat_dict["img_4"]
		else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw'] #gt_img= 기본 1배
		input_list.append(input_img) #input_list.append(input_img)= _2 or _3 or _4
		gt_list.append(gt_img) #gt_list.append(gt_img) = 기본 1배 
		scale_list.append(pair[2]) #scale_list.appen(pair[2]) = 2배면 2, 3배면 3, 4배면 4
	return input_list, gt_list, scale_list   #input_list[0] = gt_list[0] , size는 같다 mat형태로 nxn 구조에 pixel값들을 가짐 (이전에는 주소만 가졌음)
	#input_list= 2배,3배,4배 gt_list는 1배, scale는 정수
    
def test_VDSR_with_sess(epoch, ckpt_path, data_path,sess):
	folder_list = glob.glob(os.path.join(data_path, 'Set*')) #'Set*'
	print ('folder_list', folder_list) #folder_list [./data/test//Set14]
	if not os.path.exists('./output_img'):
		os.mkdir('./output_img')        
	saver.restore(sess, ckpt_path)
	
	psnr_dict = {}
	psnr_bicub_cnt_2=0
	psnr_bicub_sum_2=0
	psnr_bicub_total_2=0

	psnr_bicub_cnt_3=0
	psnr_bicub_sum_3=0
	psnr_bicub_total_3=0
#
	psnr_bicub_cnt_4=0
	psnr_bicub_sum_4=0
	psnr_bicub_total_4=0
	
	psnr_vdsr_cnt_2=0
	psnr_vdsr_sum_2=0
	psnr_vdsr_total_2=0
	
	psnr_vdsr_cnt_3=0
	psnr_vdsr_sum_3=0
	psnr_vdsr_total_3=0

	psnr_vdsr_cnt_4=0
	psnr_vdsr_sum_4=0
	psnr_vdsr_total_4=0
	input_cnt=0
	vdsr_cnt=0
	gt_cnt=0
	for folder_path in folder_list: #초기 folder_path = ./data/test/Set14
		psnr_list = []
		img_list = get_img_list(folder_path) #img_list에는 [./data/test//Set14//0.mat', './data/test//Set14//0_2.mat', 2] 와같은 3개의 정보를 가짐
		for i in range(len(img_list)): #len(img_list)=42
			input_list, gt_list, scale_list = get_test_image(img_list, i, 1) #1 은 한번에 하나의 사진만 하겠다는 batch_size
			input_y = input_list[0]
			gt_y = gt_list[0]
			start_t = time.time()
			img_vdsr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
			img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
			end_t = time.time()
            
			print ("end_t",end_t,"start_t",start_t)
			print ("time consumption",end_t-start_t)
			print ("image_size", input_y.shape)

			input_count=str(input_cnt)
			vdsr_count=str(vdsr_cnt)
			gt_count=str(gt_cnt)
			#save test image and results
			id = img_list[i][1].split('/')[-1].split('.')[0]
#			output_id = 'output'+id+'.png'
#			input_id = 'input' + id + '.png'
            
        
			filename=os.path.basename(model_ckpt.split('/')[-1].split('.')[0])
			#print('filename=', filename)
			#scipy.misc.imsave(os.path.join('./output_img', 'image.png'),img_vdsr_y) #-> ./output_img dir에 image 저장           
			#scipy.misc.imsave('image.png',img_vdsr_y) #-> 이거 동작함 ->현재 디렉토리에 생성됨 

#			scipy.misc.imsave(os.path.join('./output_img', filename+'_input'+input_count+'.png'),input_y)   #image 생성
#			scipy.misc.imsave(os.path.join('./output_img', 'gt'+'_'+gt_count+'.png'),gt_y) #image 생성

#			scipy.misc.imsave(os.path.join('./output_img',input_id),input_y)
#			scipy.misc.imsave(os.path.join('/output_img',output_id),img_vdsr_y)
                      
			if scale_list[0] == 2:
				psnr_bicub = psnr(input_y, gt_y, scale_list[0])                
				psnr_bicub_cnt_2+=1
				psnr_bicub_sum_2+=psnr_bicub
				psnr_bicub_total_2=psnr_bicub_sum_2/psnr_bicub_cnt_2
				psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
				psnr_vdsr_cnt_2+=1
				psnr_vdsr_sum_2+=psnr_vdsr
				psnr_vdsr_total_2=psnr_vdsr_sum_2/psnr_vdsr_cnt_2
				scipy.misc.imsave(os.path.join('./output_img', filename+'_ground_truth'+'_x2'+gt_count+'.png'),gt_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_outputx2'+'_'+vdsr_count+'.png'),img_vdsr_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_bicubicx2'+'_'+input_count+'.png'),input_y)                
#				print("%d 번째 Average PSNR x2 scale: bicubic %f\tVDSR %f" %(psnr_bicub_cnt_2,psnr_bicub_total_2, psnr_vdsr_total_2))
				print("PSNR x2 scale : bicubic %f\tVDSR %f" %(psnr_bicub,psnr_vdsr))

			if scale_list[0]==3:
				psnr_bicub = psnr(input_y, gt_y, scale_list[0])                
				psnr_bicub_cnt_3+=1
				psnr_bicub_sum_3+=psnr_bicub
				psnr_bicub_total_3=psnr_bicub_sum_3/psnr_bicub_cnt_3
				psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
				psnr_vdsr_cnt_3+=1
				psnr_vdsr_sum_3+=psnr_vdsr
				psnr_vdsr_total_3=psnr_vdsr_sum_3/psnr_vdsr_cnt_3
				scipy.misc.imsave(os.path.join('./output_img', 'ground_truth'+'_x3'+gt_count+'.png'),gt_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_outputx3'+'_'+vdsr_count+'.png'),img_vdsr_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_bicubicx3'+'_'+input_count+'.png'),input_y)
#				print("%d 번째 Average PSNR x3 scale: bicubic %f\tVDSR %f" %(psnr_bicub_cnt_3,psnr_bicub_total_3, psnr_vdsr_total_3))
				print("PSNR x3 scale : bicubic %f\tVDSR %f" %(psnr_bicub,psnr_vdsr))

			if scale_list[0] ==4:
				psnr_bicub = psnr(input_y, gt_y, scale_list[0])                
				psnr_bicub_cnt_4+=1
				psnr_bicub_sum_4+=psnr_bicub
				psnr_bicub_total_4=psnr_bicub_sum_4/psnr_bicub_cnt_4
				psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
				psnr_vdsr_cnt_4+=1
				psnr_vdsr_sum_4+=psnr_vdsr
				psnr_vdsr_total_4=psnr_vdsr_sum_4/psnr_vdsr_cnt_4
				scipy.misc.imsave(os.path.join('./output_img', 'ground_truth'+'_x4'+gt_count+'.png'),gt_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_outputx4'+'_'+vdsr_count+'.png'),img_vdsr_y)
				scipy.misc.imsave(os.path.join('./output_img', filename+'_bicubicx4'+'_'+input_count+'.png'),input_y)
#				print("%d 번째 Average PSNR x4 scale: bicubic %f\tVDSR %f" %(psnr_bicub_cnt_4, psnr_bicub_total_4, psnr_vdsr_total_4))
				print("PSNR x4 scale : bicubic %f\tVDSR %f" %(psnr_bicub,psnr_vdsr))

			input_cnt+=1
			vdsr_cnt+=1
			gt_cnt+=1

#			print ("PSNR: bicubic %f\tVDSR %f" % (psnr_bicub, psnr_vdsr)
			psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])            
		psnr_dict[os.path.basename(folder_path)] = psnr_list
	if not os.path.exists('./psnr'):
		os.mkdir('psnr')
	with open('psnr/%s' % os.path.basename(ckpt_path), 'wb') as f:
		pickle.dump(psnr_dict, f)
def test_VDSR(epoch, ckpt_path, data_path): #data_path="./data/test/"
	with tf.Session() as sess:
		test_VDSR_with_sess(epoch, ckpt_path, data_path, sess)
if __name__ == '__main__': #특정 main에서만 실행되도록함
	#model_list = sorted(glob.glob("./checkpoints/*"))
	#model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
	model_list=glob.glob("./checkpoints/VDSR_adam_epoch_*.data-00000-of-00001") #아래에 있는 경로에 있는 파일들 목록 가져옴
	model_list=sorted([fn[:-20] for fn in model_list]) #epoch_*.ckpt-* 까지 가짐
	with tf.Session() as sess:
		input_tensor  			= tf.placeholder(tf.float32, shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', model)
		output_tensor, weights 	= shared_model(input_tensor)
		#output_tensor, weights 	= model(input_tensor)
		saver = tf.train.Saver(weights)
		tf.initialize_all_variables().run()
		for model_ckpt in model_list: #model_ckpt = ./checkpoints/VDSR_adam_epoch_000.ckpt-3364 
			print ('model_ckpt=',model_ckpt)
			epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0]) #숫자 뽑아내기
			#if epoch<60:
			#	continue
			print ("Testing model",model_ckpt)
			test_VDSR_with_sess(120, model_ckpt, DATA_PATH,sess)
