import robosims.server
import random
import matplotlib.pyplot as plt
import numpy as np
#import cPickle as pickle
import cv2
from PIL import Image
import caffe
import os
import sys


# try:
#     import setproctitle
#     setproctitle.setproctitle(os.path.basename(os.getcwd()))
# except:
#     pass

# init
#caffe.set_device(0)
caffe.set_mode_cpu()

# hyperparameters
#H = 200 # number of hidden layer neurons
batch_size = 1 # every how many episodes to do a param update
learning_rate = 2*1e-5
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
#resume = False # resume from previous checkpoint?
#render = False
Numtarget = 0

# model initialization
model = {}
model['W1'] = np.random.randn(2048,512) / np.sqrt(2048) # "Xavier" initialization
model['W2'] = np.random.randn(1024,512) / np.sqrt(1024)
model['W3'] = np.random.randn(512,6) / np.sqrt(512)
model['W4'] = np.random.randn(512,1) / np.sqrt(512)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_image():
#     idx_observe = str(np.random.randint(0,199))
#     # print idx_observe
    idx_target = str(np.random.randint(0,1377))
    im_target = cv2.imread('/Users/Cece/Desktop/THOR/data/input/target_'+idx_target+'.jpg')
#     cv2.imwrite('/Users/Cece/Desktop/Target.jpg', im_target)
    # im_target = Image.open('/Users/Cece/Desktop/THOR/data/input/target_'+idx_target+'.jpg')
#     # im_observe = Image.open('/Users/Cece/Desktop/THOR/data/input/target_368.jpg')
#     # im_target = Image.open('/Users/Cece/Desktop/THOR/data/input/target_369.jpg')
#     # observe_in = np.array(im_observe, dtype = np.float32)
#     # observe_in = observe_in[:,:,::-1]
#     #observe_in -= self.mean
#     # observe_in = observe_in.transpose((2,0,1))
#     #observe_in = observe_in.flatten()
#     # im_target = im_observe
    cv2.imwrite('/Users/Cece/Desktop/target_in.jpg', im_target)
    target_in = np.array(im_target, dtype=np.float32)
    target_in = target_in[:, :, ::-1]
    target_in = target_in.transpose((2,0,1))
#     #target_in = target_in.flatten()
    return im_target, target_in

def discount_allrewards(r):
    """ take 2D float array of rewards and compute discounted reward """
    n = r.shape[0]
    r = r.flatten()
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r = np.reshape(discounted_r,(n,4))
    return discounted_r

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = np.asscalar(running_add)
    return discounted_r

def network_forward(observe_out, target_out):
    h_observe = np.dot(model['W1'].T, observe_out.T)
    h_target = np.dot(model['W1'].T, target_out.T)
    fuse = np.concatenate((h_observe, h_target)) #(1024,1)
    fuse = fuse.flatten()
    h_scene = np.dot(model['W2'].T, fuse) #(512,1)
    h_prob = np.dot(model['W3'].T, h_scene) #(4,1)
    p = softmax(h_prob)
    value = np.dot(model['W4'].T, h_scene) #(1, 1)
    return p, fuse, h_scene, value # return probability and hidden state and value

def network_backward(epfuse, ephscene, epdlogp, ephprob, epvalue_out, epobserve_out, eptarget_out, discounted_epr, epind):
    """ backward pass.  """
    # print epvalue.shape
    # print ephscene.shape
    dW4 = np.dot(ephscene.T, -(discounted_epr - epvalue_out)) #(512, 1)
    dW4 = np.reshape(dW4, (ephscene.shape[1], 1))
    dfuse1 = np.dot(-(discounted_epr - epvalue_out), model['W4'].T) #(10,512)
    # print dfuse1.shape
    #########################################################
    #p = exp(epdlogp / discounted_epr)
    b_dsoftmax = -1 / ephprob * (discounted_epr - epvalue_out)  # (n, 6)
    a_dsoftmax = np.zeros((6, 6, b_dsoftmax.shape[0]))  # (6,6,n)
    dsoftmax = np.zeros((b_dsoftmax.shape[0], 6)) #(6, 4)
    for n in range(b_dsoftmax.shape[0]):
        for i in range(6):
            for j in range(6):
                if i == j:
                    a_dsoftmax[i, j, n] = b_dsoftmax[n, i] * (1 - b_dsoftmax[n, i])
                else:
                    a_dsoftmax[i, j, n] = - b_dsoftmax[n, i] * b_dsoftmax[n, j]
        dsoftmax[n, :] = a_dsoftmax[epind[n], :, n]

    # print dsoftmax.shape
    dW3 = np.dot(ephscene.T, dsoftmax)
    # print dW3.shape
    dfuse2 = np.dot(dsoftmax, model['W3'].T)
    ##########################################################
    dfuse = dfuse1+dfuse2 #(10, 512)
    # print epfuse.shape
    dW2 = np.dot(epfuse.T, dfuse) #(1024, 512)
    # print dW2.shape
    dh = np.dot(dfuse, model['W2'].T) #(10, 1024)
    hsplit = np.split(dh, 2, axis=1) # 2*(10, 512)
    dW1 = 0.5 * (np.dot(epobserve_out.T, hsplit[0]) + np.dot(eptarget_out.T, hsplit[1]))
    return {'W1':dW1, 'W2':dW2, 'W3':dW3, 'W4':dW4}


######################################################
# load environment
env = robosims.server.Controller(
                player_screen_width=400,
                player_screen_height=400,
                darwin_build='/Users/Cece/Desktop/THOR/RoboSims/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64',
                # linux_build='/home/xiaoyan1/data/thor-cmu-201703101558-Linux64/thor-cmu-201703101558-Linux64',
                x_display="0.0")

env.start()
actionpool = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'MoveLeft', 'MoveRight']

# initialize target and observation
# env.reset('FloorPlan224') # FloorPlan223 and FloorPlan225 are also available
# event = env.step(dict(action=random.choice(actionpool)))
# while not event.metadata['lastActionSuccess']:
#     event = env.step(dict(action=random.choice(actionpool)))
# im_target = event.frame
# target_in = np.array(im_target, dtype=np.float32)
# target_in = target_in[:, :, ::-1]
# # cv2.imwrite('/Users/Cece/Desktop/target_in_1.jpg', target_in)
# target_in = target_in.transpose((2,0,1))
# cv2.imwrite('/Users/Cece/Desktop/target_in.jpg', event.frame)
# cv2.imwrite('/Users/Cece/Desktop/target_in_1.jpg', target_in)

# cv2.imwrite('/Users/Cece/Desktop/THOR/data/input/observe_in.jpg', event.frame)
# im_observe = cv2.imread('/Users/Cece/Desktop/THOR/data/input/observe_in.jpg'
im_target, target_in = load_image() # initial target
hist2,bins2 = np.histogram(im_target.ravel(),256,[0,256])

env.reset('FloorPlan224') # FloorPlan223 and FloorPlan225 are also available
event = env.step(dict(action=random.choice(actionpool)))
while not event.metadata['lastActionSuccess']:
    event = env.step(dict(action=random.choice(actionpool)))
cv2.imwrite('/Users/Cece/Desktop/observe_in.jpg', event.frame)
im_observe = cv2.imread('/Users/Cece/Desktop/observe_in.jpg')
hist1,bins1 = np.histogram(im_observe.ravel(),256,[0,256])

observe_in = np.array(im_observe, dtype = np.float32)
observe_in = observe_in[:,:,::-1]
# cv2.imwrite('/Users/Cece/Desktop/observe_in_1.jpg', observe_in)
observe_in = observe_in.transpose((2,0,1))

image_buffer = im_observe
# cv2.imwrite('/Users/Cece/Desktop/observe_in_1.jpg', observe_in)
# image_buffer = event.frame
# plt.figure()
# plt.imshow(target_in)
# plt.savefig('/Users/Cece/Desktop/test1.jpg')

actions = {'0': 'MoveAhead',
           '1': 'MoveBack',
           '2': 'RotateLeft',
           '3': 'RotateRight',
           '4': 'MoveLeft',
           '5': 'MoveRight'
           }

xobserve,xtarget,hs1,hs2,value_out,hprob,dlogps,drs,p_num = [],[],[],[],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

# load net
net = caffe.Net('data/ResNet-50.prototxt', 'data/ResNet-50-model.caffemodel', caffe.TEST)
NumofStep = 0
while True:
    #observe_in,target_in = load_image()
    diff = np.where(observe_in != target_in)
    CorErr = np.sum((hist1 - hist2) ** 2) / 255
    NumofStep = NumofStep + 1
    if (diff[0].shape[0] == 0):
        done = True
        dist = 'Arrived!'
        # reward = 10
        break
    else:
        if CorErr < 2000000:
            dist = 'Close!'
        else:
            dist = 'Gogogo!'
        done = False
        
        # load net
        # net = caffe.Net('data/ResNet-50.prototxt', 'data/ResNet-50-model.caffemodel', caffe.TEST)
        # shape for input (data blob is N x C x H x W), set data
        # net.blobs['data'].reshape(1, *observe_in.shape)
        net.blobs['data'].data[...] = np.resize(observe_in, (1, 3, 224, 224))
        net.forward()
        observe_out = net.blobs['pool5'].data
        observe_out = np.reshape(observe_out, (1, observe_out.shape[1]))
        # print observe_out.shape
        
        # net.blobs['data'].reshape(1, *target_in.shape)
        net.blobs['data'].data[...] = np.resize(target_in, (1, 3, 224, 224))
        net.forward()
        target_out = net.blobs['pool5'].data
        # print target_out.shape
        target_out = np.reshape(target_out, (1, target_out.shape[1]))
        # print target_out.shape

        prob, fuse, h_scene, value = network_forward(observe_out, target_out)
        # print prob.shape
        # print fuse.shape
        # print h_scene.shape
        # print value.shape

        dictionary = {'0': prob[0],
                '1': prob[1],
                '2': prob[2],
                '3': prob[3],
                '4': prob[4],
                '5': prob[5]
                }
        randPos = np.random.random()
        
        prob_sum = prob[0]
        if randPos > prob_sum:
            prob_sum += prob[1]
            if randPos > prob_sum:
                prob_sum += prob[2]
                if randPos > prob_sum:
                    prob_sum += prob[3]
                    if randPos > prob_sum:
                        prob_sum += prob[4]
                        if randPos > prob_sum:
                            action = '5'
                        else: action = '4'
                    else: action = '3'
                else: action = '2'
            else: action = '1'
        else: action = '0'
        
        if Numtarget < 3:
            event = env.step(dict(action=random.choice(actionpool)))
            print 'action: ', random.choice(actionpool) 
            if random.choice(actionpool) == 'MoveAhead': action = '0'
            if random.choice(actionpool) == 'MoveBack': action = '1'
            if random.choice(actionpool) == 'RotateLeft': action = '2'
            if random.choice(actionpool) == 'RotateRight': action = '3' 
            if random.choice(actionpool) == 'MoveLeft': action = '4' 
            if random.choice(actionpool) == 'MoveRight': action = '5' 
        else:
            event = env.step(dict(action=actions[action]))
            print 'action: ', actions[action]
        
        if event.metadata['lastActionSuccess']:
            reward = -0.01
            image_buffer = event.frame
            CorErr = np.sum((hist1 - hist2) ** 2) / 255
            if CorErr < 2000000:
                reward = 1
            if CorErr < 500000:  
                reward = 2
            if CorErr < 100000:
                reward = 3
            if CorErr < 50000:
                reward = 5
        else:
            reward = -0.1

        cv2.imwrite('/Users/Cece/Desktop/observe_in.jpg', image_buffer)
        image_buffer = cv2.imread('/Users/Cece/Desktop/observe_in.jpg')
        # im_observe = cv2.imread('/Users/Cece/Desktop/THOR/data/input/observe_in.jpg')
        hist1,bins1 = np.histogram(image_buffer.ravel(),256,[0,256])

        observe_in = np.array(image_buffer, dtype = np.float32)
        # print observe_in.shape
        observe_in = observe_in[:,:,::-1]
        observe_in = observe_in.transpose((2,0,1))
        
        diff = np.where(observe_in != target_in)
        # print observe_in.shape
        # print target_in.shape
        if (diff[0].shape[0] == 0):
            done = True
            reward = 10
        else:
            done = False

        print diff[0].shape[0]
        print done
        print dist
        # event = env.step(dict(action=random.choice(actionpool)))

        #############################################################################
        #store all the reward info
        # tmp_reward = []
        # for i in range(4):
        #     all_action = str(i)
        #     all_event = env.step(dict(action=actions[action]))
        #     if all_event.metadata['lastActionSuccess'] == 'False':
        #         tmp_reward.append(-0.03)
        #     else:
        #         tmp_reward.append(-0.01)
        #############################################################################

        # print 'action: ', actions[action]
        # print idx_observe
        
        # print(event.metadata['lastActionSuccess'])
        
                # break
        # im_observe = event.frame
        

        # # plt.figure()
        # # plt.imshow(im_observe)
        # observe_in = np.array(im_observe, dtype = np.float32)
        # observe_in = observe_in[:,:,::-1]
        # observe_in = observe_in.transpose((2,0,1))
        # env.stop()

    xobserve.append(observe_out) # observation
    xtarget.append(target_out) # target
    hs1.append(fuse) # hidden state
    hs2.append(h_scene)
    value_out.append(value[0])
    dlogps.append(-np.log(dictionary[action]))
    p_num.append(int(action)) # chosen index
    prob = prob.tolist()
    hprob.append(prob) # all prob
    # print hprob
    # all_reward.append(tmp_reward) # all rewards
    reward_sum += reward
    print 'reward_sum: ', reward_sum, ' current reward: ', reward
    drs.append(reward) #single reward
  
    if done or NumofStep == 30: # an episode finished
        episode_number += 1

        if done:
            print 'Done Yeeeeeeeeeeah'
            Numtarget += 1
            cv2.imwrite('/Users/Cece/Desktop/observe_done.jpg', image_buffer)
            cv2.imwrite('/Users/Cece/Desktop/target_done.jpg', im_target)
            
            # env.reset('FloorPlan224') # FloorPlan223 and FloorPlan225 are also available
            # event = env.step(dict(action=random.choice(actionpool)))
            # while not event.metadata['lastActionSuccess']:
            #     event = env.step(dict(action=random.choice(actionpool)))
            # target_in = np.array(event.frame, dtype=np.float32)
            # target_in = target_in[:, :, ::-1]
            # target_in = target_in.transpose((2,0,1))
            # cv2.imwrite('/Users/Cece/Desktop/target_in.jpg', event.frame)
            
            im_target, target_in = load_image() # choose new target
            hist2,bins2 = np.histogram(im_target.ravel(),256,[0,256])

            env.reset('FloorPlan224') # FloorPlan223 and FloorPlan224 are also available
            event = env.step(dict(action=random.choice(actionpool)))
            while not event.metadata['lastActionSuccess']:
                event = env.step(dict(action=random.choice(actionpool)))
            cv2.imwrite('/Users/Cece/Desktop/observe_in.jpg', event.frame)
            im_observe = cv2.imread('/Users/Cece/Desktop/observe_in.jpg')
            hist1,bins1 = np.histogram(im_observe.ravel(),256,[0,256])

            observe_in = np.array(im_observe, dtype = np.float32)
            observe_in = observe_in[:,:,::-1]
            observe_in = observe_in.transpose((2,0,1))
            image_buffer = im_observe

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epobserve_out = np.vstack(xobserve)
        eptarget_out = np.vstack(xtarget)
        # print eptarget_out.shape
        epfuse = np.vstack(hs1)
        # print epfuse.shape
        ephscene = np.vstack(hs2)
        # print ephscene.shape
        epdlogp = np.vstack(dlogps)
        epind = np.vstack(p_num)
        ephprob = np.vstack(hprob)
        # print ephprob.shape
        epvalue_out = np.vstack(value_out)
        # epallr = np.vstack(all_reward)
        epr = np.vstack(drs)
        # print epr.shape

        

        # compute the discounted reward backwards through time
        # print '************************'
        # print epr
        discounted_epr = discount_rewards(epr) #n*1
        # discounted_epr -= np.mean(discounted_epr + 1e-5)
        # discounted_epr /= np.std(discounted_epr + 1e-5)
        # discounted_epallr = discounted_allrewards(epallr) #n*4
        # print discounted_epr.shape
        # print np.array(value_out).shape
        epvalue = (discounted_epr - epvalue_out)**2
        # print epvalue
        value_loss = np.sum((discounted_epr - epvalue_out)**2)
        print 'value loss: ', value_loss
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        # print discounted_epr
        # discounted_epr -= np.mean(discounted_epr + 1e-5)
        # discounted_epr /= np.std(discounted_epr + 1e-5)

        # print epdlogp
        # print '&&&&&&&&&&&&&&&&&&'
        # print discounted_epr
        # print epvalue_out
        epdlogp *= (discounted_epr - epvalue_out) # modulate the gradient with advantage (PG magic happens right here.)
        # print '************', epdlogp
        # epdlogp = -epdlogp
        policy_loss = np.sum(epdlogp)
        print 'policy loss: ', policy_loss
        total_loss = policy_loss + 0.5 * value_loss
        print 'total loss: ', total_loss
        grad = network_backward(epfuse, ephscene, epdlogp, ephprob, epvalue_out, epobserve_out, eptarget_out, discounted_epr, epind)
        for k in model:
            # print k
            # print 'yeeeeeeeeeeeeeeeeah'
            # print grad_buffer['W4'].shape
            # print grad['W4'].shape
            grad_buffer[k] += grad[k] # accumulate grad over batch
        
        xobserve,xtarget,hs1,hs2,value_out,hprob,dlogps,drs,p_num = [],[],[],[],[],[],[],[],[]
        NumofStep = 0
        # perform rmsprop parameter update every batch_size episodes
        print 'episode_number: ', episode_number
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            print 'Done updating parameters ^_^'
            print 'Number of reaching target: ', Numtarget 

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        #if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        # env.start()
        # env.reset('FloorPlan225') # reset env
        # pool = ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft']
        # event = env.step(dict(action=random.choice(pool)))
        # plt.savefig(event.frame, '/home/xiaoyan1/data/observe.jpg')
        if episode_number % 50 == 0:
            np.save('net', model)
        
