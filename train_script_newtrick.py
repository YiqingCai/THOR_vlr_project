import robosims.server
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import caffe
import os
import sys
from skimage.measure import compare_ssim as ssim


# init
#caffe.set_device(0)
caffe.set_mode_cpu()

# hyperparameters
batch_size = 1 # every how many episodes to do a param update
learning_rate = 2*1e-5
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
Numtarget = 0

# model initialization
model = {}
model['W1'] = np.random.randn(2048,512) / np.sqrt(2048) # "Xavier" initialization
model['W2'] = np.random.randn(1024,512) / np.sqrt(1024)
model['W3'] = np.random.randn(512,6) / np.sqrt(512)
model['W4'] = np.random.randn(512,1) / np.sqrt(512)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load_image():
    idx_target = str(np.random.randint(0,1377))
    im_target = cv2.imread('/Users/Cece/Desktop/THOR/data/input/target_'+idx_target+'.jpg')
    cv2.imwrite('/Users/Cece/Desktop/target_in.jpg', im_target)
    target_in = np.array(im_target, dtype=np.float32)
    target_in = target_in[:, :, ::-1]
    target_in = target_in.transpose((2,0,1))
    return im_target, target_in


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = np.asscalar(running_add)
    return discounted_r


def network_forward(observe_out, target_out):
    h_observe = np.dot(model['W1'].T, observe_out.T)
    h_target = np.dot(model['W1'].T, target_out.T)
    fuse = np.concatenate((h_observe, h_target)) #(1024,1)
    fuse = fuse.flatten()
    h_scene = np.dot(model['W2'].T, fuse) #(512,1)
    h_prob = np.dot(model['W3'].T, h_scene) #(6,1)
    p = softmax(h_prob)
    value = np.dot(model['W4'].T, h_scene) #(1, 1)
    return p, fuse, h_scene, value # return probability and hidden state and value


def network_backward(epfuse, ephscene, epdlogp, ephprob, epvalue_out, epobserve_out, eptarget_out, discounted_epr, epind):
    """ backward pass.  """
    dW4 = np.dot(ephscene.T, -(discounted_epr - epvalue_out)) #(512, 1)
    dW4 = np.reshape(dW4, (ephscene.shape[1], 1))
    dfuse1 = np.dot(-(discounted_epr - epvalue_out), model['W4'].T) #(10,512)
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
    dW3 = np.dot(ephscene.T, dsoftmax)
    dfuse2 = np.dot(dsoftmax, model['W3'].T)
    dfuse = dfuse1+dfuse2 #(10, 512)
    dW2 = np.dot(epfuse.T, dfuse) #(1024, 512)
    dh = np.dot(dfuse, model['W2'].T) #(10, 1024)
    hsplit = np.split(dh, 2, axis=1) # 2*(10, 512)
    dW1 = 0.5 * (np.dot(epobserve_out.T, hsplit[0]) + np.dot(eptarget_out.T, hsplit[1]))
    return {'W1':dW1, 'W2':dW2, 'W3':dW3, 'W4':dW4}


def compareImage(observe_im, im_target):  
    hist_observe, bins = np.histogram(observe_im.ravel(),256,[0,256])
    hist_target, bins = np.histogram(im_target.ravel(),256,[0,256])
    hist_diff = np.sum((hist_observe - hist_target)**2) / 255
    # im1 = cv2.cvtColor(observe_im, cv2.COLOR_BGR2GRAY)
    # im2 = cv2.cvtColor(im_target, cv2.COLOR_BGR2GRAY)
    #struct_simi = ssim(im1,im2)
    struct_simi = ssim(observe_im, im_target, multichannel = True)
    return struct_simi, hist_diff


######################################################
# load environment
env = robosims.server.Controller(
                player_screen_width = 400,
                player_screen_height = 400,
                darwin_build='/Users/Cece/Desktop/THOR/RoboSims/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64',
                #linux_build='/home/xiaoyang/Desktop/Thor/thor-cmu-201703101558-Linux64/thor-cmu-201703101558-Linux64',
                x_display="0.0")

print 'start'
env.start()
actionpool = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'MoveLeft', 'MoveRight']

# initialize target and observation
im_target, target_in = load_image() # initial target
hist2,bins2 = np.histogram(im_target.ravel(),256,[0,256])
print 'reset'
env.reset('FloorPlan224') # FloorPlan223 and FloorPlan225 are also available
event = env.step(dict(action=random.choice(actionpool)))
while not event.metadata['lastActionSuccess']:
    event = env.step(dict(action=random.choice(actionpool)))
print 'event_frame', event.frame.shape
cv2.imwrite('/Users/Cece/Desktop/observe_in.jpg', event.frame)
im_observe = cv2.imread('/Users/Cece/Desktop/observe_in.jpg')
hist1,bins1 = np.histogram(im_observe.ravel(),256,[0,256])
observe_in = np.array(im_observe, dtype = np.float32)
observe_in = observe_in[:,:,::-1]
observe_in = observe_in.transpose((2,0,1))
image_buffer = im_observe


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
print 'load net'
net = caffe.Net('data/ResNet-50.prototxt', 'data/ResNet-50-model.caffemodel', caffe.TEST)
NumofStep = 0
value_loss = 5
reward_sum = 0
reward = 0

while True:
    diff = np.where(observe_in != target_in)
    CorErr = np.sum((hist1 - hist2) ** 2) / 255
    NumofStep = NumofStep + 1
    if (diff[0].shape[0] == 0):
        done = True
        dist = 'Arrived!'
        break
    else:
        if CorErr < 2000000:
            dist = 'Close!'
        else:
            dist = 'Gogogo!'
        done = False
        
        
        net.blobs['data'].data[...] = np.resize(observe_in, (1, 3, 224, 224))
        net.forward()
        observe_out = net.blobs['pool5'].data
        observe_out = np.reshape(observe_out, (1, observe_out.shape[1]))
        
        
        net.blobs['data'].data[...] = np.resize(target_in, (1, 3, 224, 224))
        net.forward()
        target_out = net.blobs['pool5'].data
        target_out = np.reshape(target_out, (1, target_out.shape[1]))
    

        prob, fuse, h_scene, value = network_forward(observe_out, target_out)
       

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
        
        if reward > 0:
            print 'getting closer'
            
            event_ahead = env.step(dict(action = '0')) #move ahead
            struct_simi_a, hist_diff_a = compareImage(event_ahead.frame, im_target) #(observe_im, im_target)

            event_back = env.step(dict(action = '1')) #move back
            struct_simi_b, hist_diff_b = compareImage(event_back.frame, im_target) #(observe_im, im_target)

            event_mvleft = env.step(dict(action = '4')) #move left
            struct_simi_l, hist_diff_l = compareImage(event_mvleft.frame, im_target) #(observe_im, im_target)

            event_mvright = env.step(dict(action= '5')) #move right
            struct_simi_r, hist_diff_r = compareImage(event_mvright.frame, im_target) #(observe_im, im_target)

            # event_choice = {str(np.sum(hist1_0 - hist2)**2): event_ahead,
            #                 str(np.sum(hist1_1 - hist2)**2): event_back,
            #                 str(np.sum(hist1_2 - hist2)**2): event_mvleft,
            #                 str(np.sum(hist1_3 - hist2)**2): event_mvright,
            #                 }
            event_choice = {str(struct_simi_a): event_ahead,
                            str(struct_simi_b): event_back,
                            str(struct_simi_l): event_mvleft,
                            str(struct_simi_r): event_mvright,
                            }
            
            struct_max = max(struct_simi_a, struct_simi_b, struct_simi_l, struct_simi_r)
            

            event = event_choice[str(struct_max)]
            

        elif value_loss > 2 or reward_sum < 0.5:
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
            struct_simi, CorErr = compareImage(event.frame, im_target)
            print 'Struct similarity', struct_simi
            if CorErr < 2500000 or struct_simi > 0.70:
                reward = 1
            if CorErr < 800000:  
                reward = 2
            if CorErr < 450000:
                reward = 3
            if CorErr < 150000:
                reward = 5
        else:
            reward = - 1

        

        cv2.imwrite('/Users/Cece/Desktop/observe_in.jpg', image_buffer)
        image_buffer = cv2.imread('/Users/Cece/Desktop/observe_in.jpg')
        hist1,bins1 = np.histogram(image_buffer.ravel(),256,[0,256])

        observe_in = np.array(image_buffer, dtype = np.float32)
        observe_in = observe_in[:,:,::-1]
        observe_in = observe_in.transpose((2,0,1))
        
        diff = np.where(observe_in != target_in)
        if (diff[0].shape[0] == 0):
            done = True
            reward = 10
        else:
            done = False

        print diff[0].shape[0]
        print done
        print dist
        

    xobserve.append(observe_out) # observation
    xtarget.append(target_out) # target
    hs1.append(fuse) # hidden state
    hs2.append(h_scene)
    value_out.append(value[0])
    dlogps.append(-np.log(dictionary[action]))
    p_num.append(int(action)) # chosen index
    prob = prob.tolist()
    hprob.append(prob) # all prob
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
        epfuse = np.vstack(hs1)
        ephscene = np.vstack(hs2)
        epdlogp = np.vstack(dlogps)
        epind = np.vstack(p_num)
        ephprob = np.vstack(hprob)
        epvalue_out = np.vstack(value_out)
        epr = np.vstack(drs)


        discounted_epr = discount_rewards(epr) #n*1
        epvalue = (discounted_epr - epvalue_out)**2
        value_loss = np.sum((discounted_epr - epvalue_out)**2)
        print 'value loss: ', value_loss
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        # discounted_epr -= np.mean(discounted_epr + 1e-5)
        # discounted_epr /= np.std(discounted_epr + 1e-5)

        epdlogp *= (discounted_epr - epvalue_out) # modulate the gradient with advantage (PG magic happens right here.)
        policy_loss = np.sum(epdlogp)
        print 'policy loss: ', policy_loss
        total_loss = policy_loss + 0.5 * value_loss
        print 'total loss: ', total_loss
        grad = network_backward(epfuse, ephscene, epdlogp, ephprob, epvalue_out, epobserve_out, eptarget_out, discounted_epr, epind)
        for k in model:
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
        # if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        if episode_number % 50 == 0:
            np.save('net', model)
        