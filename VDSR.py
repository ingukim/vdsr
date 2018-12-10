import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL import model
from PSNR import psnr
from TEST import test_VDSR
#from MODEL_FACTORIZED import model_factorized
DATA_PATH = "./data/train/"
IMG_SIZE = (41, 41)
BATCH_SIZE = 64
BASE_LR = 0.0005
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 80

USE_QUEUE_LOADING = True

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

TEST_DATA_PATH = "./data/test/"

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*")) #os.path.join (경로 통합) , "*" = 무슨 글자든, 몇 글자든 상관 없다
    #glob.glob - 파일들의 목록을 뽑을 때 사용
	print (len(l)) #287088    574176
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))] #os.path.basename(path) -입력받은 경로의 가장 하위 컴포넌트를 리턴
#basename('C:\\Python30\\tmp') -> 'tmp'   , re.search(pattern, string) - string에서 pattern과 매치하는 텍스트를 탐색한다(임의 지점 매치)
	print (len(l)) #71772   143544
	train_list = []
	for f in l: #여기서 I값은 뭘 가졌는가(check)?
		if os.path.exists(f): #os.path.exists - (파일 이나 폴더가 존재하는지 확인)
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])  #f[:-4]는 뭔가? 앞에서 4자리 , f[-4:] 뒤에서 4자리
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"]) 
	return train_list #train_list= *, *_2, *_3, *_4 ex) 0_2.mat , 0_3.mat, 0_4.mat

def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, gt_list, np.array(cbcr_list)

def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	for pair in target_list:
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if "img_2" in mat_dict: 	input_img = mat_dict["img_2"]
		elif "img_3" in mat_dict: input_img = mat_dict["img_3"]
		elif "img_4" in mat_dict: input_img = mat_dict["img_4"]
		else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img[:,:,0])
		gt_list.append(gt_img[:,:,0])
	return input_list, gt_list

if __name__ == '__main__':  #VDSR.py에서만 실행되도록
	train_list = get_train_list(DATA_PATH) #DATA_PATH="./data/train/"
	
	if not USE_QUEUE_LOADING: #True가 아니면 출력
		print ("not use queue loading, just sequential loading...") #큐 로딩 사용 x, 순차적 로딩 사용


		### WITHOUT ASYNCHRONOUS DATA LOADING ###

		train_input  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1)) #BATCH_SIZE=64
		train_gt  		= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

		### WITHOUT ASYNCHRONOUS DATA LOADING ###
    
	else:
		print ("use queue loading"	)


		### WITH ASYNCHRONOUS DATA LOADING ###
    
		train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
		train_gt_single  	= tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
		q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0], IMG_SIZE[1], 1]]) #10000= Queue length
		enqueue_op = q.enqueue([train_input_single, train_gt_single]) #enqueue로 train_input 과 train_gt
    
		train_input, train_gt	= q.dequeue_many(BATCH_SIZE) #BATCH_SIZE만큼 뽑아서 train_input 과 train_gt에 저장 train_gt는 dataset에서 blur처리안한 image. train_input은 줄여서 blur했다가 다시 resize로 키운거 interpolated는됨.
    
		### WITH ASYNCHRONOUS DATA LOADING ###


	shared_model = tf.make_template('shared_model', model)
	#train_output, weights 	= model(train_input)
	train_output, weights 	= shared_model(train_input)
#	loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_gt, train_output))) #l2 loss func   train_output- train_gt
	loss = tf.reduce_mean(tf.abs(train_gt-train_output))    
	#loss = tf.reduce_mean(tf.abs(train_gt-train_output))  #l1 loss self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,output))
#	for w in weights:
#		loss += tf.nn.l2_loss(w)*5e-4
#		loss += loss*5e-4
	tf.summary.scalar("loss", loss)

	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
    #학습 속도 조절 (BASE_LR =starter_learning_rate(최초 학습시 사용될 learning_rate), global_step=현재 학습 횟수*BATCH_SIZE만큼, len(train_list)*LR_STEP*SIZE= 총 학습 횟수, LR_RATE= 얼마나 rate가 감소될것인가? , staircase=True (이산적으로 학습 속도 감속 유무), 즉, true일때 decay_rate (4번째 파라미터에 (global_step/ decay_steps)의 승수가 적용)
	tf.summary.scalar("learning rate", learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)   #Adam optimizer 사용
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list) #train_list를 무작위로 섞는다
	config = tf.ConfigProto()
	#config.operation_timeout_in_ms=10000

	with tf.Session(config=config) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
		if not os.path.exists('logs'):
			os.mkdir('logs')
		merged = tf.summary.merge_all()
		file_writer = tf.summary.FileWriter('logs', sess.graph)

		tf.initialize_all_variables().run()

		if model_path:
			print ("restore model...")
			saver.restore(sess, model_path)
			print ("Done")

		### WITH ASYNCHRONOUS DATA LOADING ###
		def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
			count = 0;
			length = len(file_list)
			try:
				while not coord.should_stop():
					i = count % length;
					input_img	= scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					gt_img		= scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:gt_img})
					count+=1
			except Exception as e:
				print ("stopping...", idx, e)
		### WITH ASYNCHRONOUS DATA LOADING ###
		threads = []
		def signal_handler(signum,frame):
			sess.run(q.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)
			print ("Done")
			sys.exit(1)
		original_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, signal_handler)

		if USE_QUEUE_LOADING:
			# create threads
			num_thread=20 #20
			coord = tf.train.Coordinator()
			for i in range(num_thread):
				length = int(len(train_list)/num_thread)
				t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
				threads.append(t)
				t.start()
			print ("num thread:" , len(threads))

			for epoch in range(0, MAX_EPOCH):
				max_step=len(train_list)//BATCH_SIZE 
				for step in range(max_step):
					_,l,output,lr, g_step, summary = sess.run([opt, loss, train_output, learning_rate, global_step, merged])
					print ("[epoch %2.4f] loss %.5f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))
					file_writer.add_summary(summary, step+epoch*max_step)
					#print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
				saver.save(sess, "./checkpoints/VDSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step) #checkpoint file 저장
		else:
			for epoch in range(0, MAX_EPOCH):
				for step in range(len(train_list)//BATCH_SIZE):
					offset = step*BATCH_SIZE
					input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE)
					feed_dict = {train_input: input_data, train_gt: gt_data}
					_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
					print ("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))
					del input_data, gt_data, cbcr_data

				saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

