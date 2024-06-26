import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from NetworkMix import NetworkMix


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='CIFAR10', help='location of the data corpus')
parser.add_argument('--datapath', type=str, default='/home/sdki/Downloads/CVPR-data/AddNIST', help='location of the data corpus')

parser.add_argument('--valid_size', type=float, default=0, help='validation data size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs') 
parser.add_argument('--auto_augment', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--random_crop', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--random_flip', action='store_true', default=False, help='use autoaugment')

parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='AddNIST', help='experiment name')
parser.add_argument('--seed', type=int, default=20, help='random seed')

parser.add_argument('--add_epochs', type=int, default=1, help='num of training epochs to increase for depth') 
parser.add_argument('--add_epochs_w', type=int, default=3, help='num of training epochs to increase for width') 
parser.add_argument('--target_acc', type=float, default=100.00, help='desired target accuracy')
parser.add_argument('--target_acc_tolerance', type=float, default=0.10, help='tolerance for desired target accuracy')
parser.add_argument('--ch_drop_tolerance', type=float, default=0.05, help='tolerance when dropping channels')
parser.add_argument('--dp_break_tolerance', type=int, default=1, help='tolerance when terminating depth search')
parser.add_argument('--ch_break_tolerance', type=int, default=3, help='tolerance when terminating channel search')
parser.add_argument('--dp_add_tolerance', type=float, default=0.10, help='tolerance when increasing depth')
parser.add_argument('--min_width', type=int, default=16, help='minimum number of init channels in search')
parser.add_argument('--max_width', type=int, default=48, help='maximum number of init channels in search')
parser.add_argument('--width_resolution', type=int, default=8, help='resolution for number of channels search')
parser.add_argument('--min_depth', type=int, default=5, help='minimum number of init layers in search')
parser.add_argument('--max_depth', type=int, default=100, help='maximum number of init layers in search')

args = parser.parse_args()

args.save = 'Rubric-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'Searchlog.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('GPU not available.')
    sys.exit(1)
 
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('GPU Device = %d' % args.gpu)
  logging.info("Arguments = %s", args)

  train_queue, valid_queue, test_queue, classes, input_shape = utils.get_loaders(args)
  
  # Run search on the fetched sub dataset.
  curr_arch_ops, curr_arch_kernel, f_epochs, f_channels, f_layers, curr_arch_train_acc = search_depth_and_width(args,   
  	                                                                                                                      classes,
                                                                                                                          input_shape,        
                                                                                                                          train_queue,
                                                                                                                          valid_queue,                                                                                                                          
                                                                                                                          test_queue)
  
  d_w_model_info = {'curr_arch_ops': curr_arch_ops,
                    'curr_arch_kernel': curr_arch_kernel,
                    'curr_arch_train_acc': curr_arch_train_acc,
                    'f_channels': f_channels,
                    'f_layers': f_layers}

  '''
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations_and_kernels(args, 
                                                                                                    classes,
                                                                                                    input_shape,
                                                                                                    train_queue,
                                                                                                    valid_queue,
                                                                                                    test_queue, 
                                                                                                    d_w_model_info)

  '''
  logging.info('END OF SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')

  model = NetworkMix(f_channels, len(classes), f_layers, curr_arch_ops, curr_arch_kernel, input_shape)
  model = model.cuda()                                                                                                                           
  logging.info(model)
  logging.info('FINAL DISCOVERED ARCHITECTURE DETAILS:')
  logging.info("Model Depth %s Model Width %s", f_layers, f_channels)
  logging.info('Discovered Final Epochs %s', f_epochs)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info("Training Accuracy %f Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

def search_depth_and_width(args, classes, input_shape, train_queue, valid_queue, test_queue):

  logging.info('#############################################################################')
  logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

  CIFAR_CLASSES = len(classes) 
  target_acc=args.target_acc
  min_width=args.min_width
  max_width=args.max_width
  width_resolution=args.width_resolution
  min_depth=args.min_depth
  max_depth=args.max_depth
  ch_drop_tolerance = args.ch_drop_tolerance
  target_acc_tolerance = args.target_acc_tolerance
  # We start with max width but with min depth.
  #channels = max_width 
  channels = min_width 
  layers = min_depth
  f_epochs = 0
  # Initialize
  curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
  curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
  curr_arch_train_acc = next_arch_train_acc = 0.0

  logging.info('RUNNING MACRO SEARCH FIRST...')

  model = NetworkMix(channels, CIFAR_CLASSES, layers, curr_arch_ops, curr_arch_kernel, input_shape)
  
  model = model.cuda()
  logging.info(model)
  logging.info('MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", layers, channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info('Training epochs %s', args.epochs)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('Training Model...')
  curr_arch_train_acc = train_test(args, classes, model,
   train_queue, valid_queue, test_queue)
  logging.info("Baseline Train Acc %f", curr_arch_train_acc)

  # Search depth
  depth_fail_count = 0
  channels_up = False

  while ((curr_arch_train_acc < (target_acc - target_acc_tolerance)) and (layers != max_depth)):
  
    # prepare next candidate architecture.  
    layers += 1

    next_arch_ops = np.zeros((layers,), dtype=int)
    next_arch_kernel = 3*np.ones((layers,), dtype=int)
    model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, input_shape)
    model = model.cuda()
    logging.info(model)
    logging.info('#############################################################################')
    logging.info('Moving to Next Candidate Architecture...')
    logging.info('MODEL DETAILS')
    logging.info("Model Depth %s Model Width %s", layers, channels)
    logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
    logging.info('Total number of epochs %s', args.epochs)
    logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    logging.info("Depth Fail Count %s", depth_fail_count)
    logging.info('Training Model...')
    next_arch_train_acc = train_test(args, classes, model,
      train_queue, valid_queue, test_queue)
    logging.info("Candidate Train Acc %f", next_arch_train_acc)
    
    # As long as we get significant improvement by increasing depth.
    
    if (next_arch_train_acc >= curr_arch_train_acc + args.dp_add_tolerance):
      # update current architecture.
      depth_fail_count = 0
      curr_arch_ops = next_arch_ops
      curr_arch_kernel = next_arch_kernel
      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
      curr_arch_train_acc = next_arch_train_acc
      f_channels = channels
      f_epochs = args.epochs
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)
      
    elif((next_arch_train_acc < curr_arch_train_acc + args.dp_add_tolerance) and ((depth_fail_count != args.dp_break_tolerance))):
      depth_fail_count += 1
      layers -= 1
      args.epochs = args.epochs + args.add_epochs
      logging.info('Increasing Epoch in DEPTH block...')
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)
      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
      continue
      
    elif(channels != max_width):
      if not channels_up:
        layers -= 1
        channels += int(width_resolution/2)
        channels_up = True
        logging.info('Increasing CHANNELS in WIDTH block...')
      else: 
        logging.info('Increasing Epoch in WIDTH block...')
        args.epochs = args.epochs + args.add_epochs
        channels_up = False
        layers -= 1
              
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)
      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
    else:
      logging.info('INCREASING CHANNELS REPEAT...')
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)
      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
      break
  # Search width
  # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

  f_layers = len(curr_arch_ops) # discovered final number of layers
  #f_channels = max_width # discovered final number of channels
  logging.info('Discovered Final Depth %s', f_layers)
  logging.info('Epochs so far %s', f_epochs)
  logging.info('END OF DEPTH SEARCH...')
  best_arch_train_acc = curr_arch_train_acc
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')
  logging.info('RUNNING WIDTH SEARCH NOW...') 

  channels = f_channels
  width_fail_count = 0
  while (channels > min_width):
    # prepare next candidate architecture.
    channels = channels - int(width_resolution/4)
    # Although these do not change.
    model = NetworkMix(channels, CIFAR_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel, input_shape)
    model = model.cuda()
    logging.info(model)
    args.epochs = args.epochs + args.add_epochs_w

    logging.info('Moving to Next Candidate Architecture...')
    logging.info('MODEL DETAILS')
    logging.info("Model Depth %s Model Width %s", f_layers, channels)
    logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
    logging.info('Total number of epochs %f', args.epochs)
    logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    logging.info('Training Model...')
    logging.info("Width Fail Count %s", width_fail_count)
    # train and test candidate architecture.
    next_arch_train_acc = train_test(args, classes, model, 
    train_queue, valid_queue, test_queue)
    logging.info("Candidate Train Acc %f ", next_arch_train_acc)

    if (next_arch_train_acc >= (curr_arch_train_acc - 0.0)):

      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
      curr_arch_train_acc = next_arch_train_acc
    
      f_channels = channels 
      f_epochs = args.epochs
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)
      width_fail_count = 0
    elif (width_fail_count != args.ch_break_tolerance):
      width_fail_count += 1
      logging.info("Train Acc Diff %f", next_arch_train_acc-curr_arch_train_acc)
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)

      continue
    else:
      logging.info("Train Acc Diff %f ", next_arch_train_acc-curr_arch_train_acc)
      logging.info("Highest Train Acc %f ", curr_arch_train_acc)

      break; 

  logging.info('Discovered Final Width %s', f_channels)
  logging.info('Discovered Final Epochs %s', f_epochs)
  logging.info('END OF WIDTH SEARCH...')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')  

  return curr_arch_ops, curr_arch_kernel, f_epochs, f_channels, f_layers, curr_arch_train_acc

def search_operations(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING OPERATION SEARCH...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  # curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel    

  for i in range(layers):
  
    #if i < 2*layers//3:
    if i < layers+1:  
      args.epochs = args.epochs + args.add_epochs

      next_arch_ops[i] = 1

      model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, input_shape)
      model = model.cuda()
    
      logging.info('NEXT MODEL DETAILS')
      logging.info("Model Depth %s Model Width %s", layers, channels)
      logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
      logging.info('Total number of epochs %f', args.epochs)
      logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
      logging.info('Training Model...')
      next_arch_train_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
      logging.info("Best Training Accuracy %f ", next_arch_train_acc)

      #if next_arch_test_acc > curr_arch_test_acc + 0.10: ######### Add arg
      if next_arch_train_acc > curr_arch_train_acc: ######### Add arg      
        curr_arch_ops = next_arch_ops
        curr_arch_kernel = next_arch_kernel
        curr_arch_train_acc = next_arch_train_acc
        # curr_arch_test_acc = next_arch_test_acc
      else:
        next_arch_ops[i] = 0

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc  

def search_kernels(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING KERNEL SEARCH...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel

  #kernels = [5]
  kernels = [5, 7]

  for i in range(layers): 
  
    #if i < 2*layers//3:
    if i < layers+1:
      best_k = 3 
      for k in kernels:
        args.epochs = args.epochs + args.add_epochs

        next_arch_kernel[i] = k
   
        model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, input_shape)
        model = model.cuda()
    
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
        logging.info('Total number of epochs %f', args.epochs)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        next_arch_train_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
        logging.info("Best Training Accuracy %f ", next_arch_train_acc)


        # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
        #if (next_arch_test_acc > curr_arch_test_acc + 0.10): # Add args
        if next_arch_train_acc > curr_arch_train_acc: ######### Add arg
          best_k = k
          curr_arch_ops = next_arch_ops
          curr_arch_kernel[i] = k
          curr_arch_train_acc = next_arch_train_acc
          # curr_arch_test_acc = next_arch_test_acc
        else:
          next_arch_kernel[i] = best_k
        
  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc

def search_ops_and_ks_simultaneous(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING OPERATIONS AND KERNELS SEARCH SIMULTANEOUSLY...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  kernels = [3, 5, 7]
  operations = [0, 1]

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel
  # Can be navigated from the last layers instead of first ones.
  for i in range(layers):  
    for k in kernels:
      for o in operations:

        args.epochs = args.epochs + args.add_epochs

        next_arch_ops[i] = o
        next_arch_kernel[i] = k
 
        model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, input_shape)
        model = model.cuda()
  
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
        logging.info('Total number of epochs %f', args.epochs)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        next_arch_train_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
        logging.info("Best Training Accuracy %f ", next_arch_train_acc)

        # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
        if (next_arch_train_acc > curr_arch_train_acc):
          curr_arch_ops = next_arch_ops
          curr_arch_kernel = next_arch_kernel
          curr_arch_train_acc = next_arch_train_acc

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc,     

def search_kernels_and_operations(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info):

  logging.info('SEARCHING FOR KERNELS FIRST...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc

  logging.info('SEARCHING FOR OPERATIONS...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

def search_operations_and_kernels(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info):
  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args, classes, input_shape, train_queue, valid_queue, test_queue, model_info)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc
  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args, classes, train_queue, valid_queue, test_queue, model_info)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

def train_test(args, classes, model, train_queue, valid_queue, test_queue):

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_train_acc = 0.0
  best_test_acc = 0.0

  for epoch in range(args.epochs):
    scheduler.step()    

    start_time = time.time()
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    #logging.info('train_acc %f', train_acc)

    
    if epoch % args.report_freq == 0:
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])	
      logging.info('train_acc %f', train_acc)  
      # logging.info('valid_acc %f', valid_acc)
       

    end_time = time.time()
    duration = end_time - start_time
    print('Epoch time: %ds.' %duration)
    print('Train acc: %f ' %train_acc)

    if train_acc > best_train_acc:
      best_train_acc = train_acc
      utils.save(model, os.path.join(args.save, 'weights.pt'))


  logging.info('Best Training Accuracy %f', best_train_acc)
  utils.load(model, os.path.join(args.save, 'weights.pt'))

  return best_train_acc

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  
  model.train()

  for step, (input, target) in enumerate(train_queue):
    # print(input.dtype, target.dtype ,input.size(),target.size())
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)

    #if step % args.report_freq == 0:
    #  logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  start_time = time.time()
  main() 
  end_time = time.time()
  duration = end_time - start_time
  logging.info('Total Search Time: %ds', duration)
