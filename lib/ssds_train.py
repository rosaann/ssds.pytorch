from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

from tensorboardX import SummaryWriter

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import visdom
import math
import numpy as np
from lib.utils.data_augment import preproc_for_test
from lib.utils.box_utils import jaccard


class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, ifTrain = True):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        self.ifTrain = ifTrain
        if self.ifTrain:
            self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
            #self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        else:
            test_image_dir = os.path.join('./data/', 'ship_test_v2')
          #  transforms = transform.Compose([transform.Lambda(lambda x: cv2.cvtColor(np.asarray(x),cv2.COLOR_RGB2BGR)),transform.Resize([300,300]), transform.ToTensor()])

          #  test_set = torchvision.datasets.ImageFolder(test_image_dir, transform = transforms)
        
          #  self.test_loader = torch.utils.data.DataLoader(test_set,batch_size=8,shuffle=False,num_workers=8)
            self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
            #self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        #self.use_gpu = False
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                 self.model = torch.nn.DataParallel(self.model).module
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        #self.model = self.model.to(device)
        # Print the model architecture and parameters
        print('Model architectures:\n{}\n'.format(self.model))

        # print('Parameters and size:')
        # for name, param in self.model.named_parameters():
        #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param = self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)
       # print('trainable_param ', trainable_param)
        self.optimizer = self.configure_optimizer(trainable_param, cfg.TRAIN.OPTIMIZER)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
       # print('priors ', self.priors)
        self.criterion = MultiBoxLoss(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX


    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        # change the name of the weights which exists in other model
        # change_dict = {
        #         'conv1.weight':'base.0.weight',
        #         'bn1.running_mean':'base.1.running_mean',
        #         'bn1.running_var':'base.1.running_var',
        #         'bn1.bias':'base.1.bias',
        #         'bn1.weight':'base.1.weight',
        #         }
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k == k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)
        # change_dict = {'layer1.{:d}.'.format(i):'base.{:d}.'.format(i+4) for i in range(20)}
        # change_dict.update({'layer2.{:d}.'.format(i):'base.{:d}.'.format(i+7) for i in range(20)})
        # change_dict.update({'layer3.{:d}.'.format(i):'base.{:d}.'.format(i+11) for i in range(30)})
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k in k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
        # print("=> Resume weigths:")
        # print([k for k, v in list(pretrained_dict.items())])

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)
        #    if 'train' in cfg.PHASE:
            self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
        #    if 'eval' in cfg.PHASE:
        #        self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
        #    if 'test' in cfg.PHASE:
        #        self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu)
        #    if 'visualize' in cfg.PHASE:
        #    self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)

    def test_model(self):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
                    self.resume_checkpoint(resume_checkpoint)
                    #if 'eval' in cfg.PHASE:
                    #    self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
                    #if 'test' in cfg.PHASE:
                    self.test_epoch_2(self.model, self.detector, self.output_dir , self.use_gpu)
                    if 'visualize' in cfg.PHASE:
                        self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            #if 'eval' in cfg.PHASE:
            #    self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
            #if 'test' in cfg.PHASE:
            self.test_epoch_2(self.model, self.detector, self.output_dir , self.use_gpu)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        

        epoch_size = int( len(data_loader) / self.cfg.DATASET.TRAIN_BATCH_SIZE)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        train_end = int( epoch_size * 0.8);
        ###
        label = [list() for _ in range(model.num_classes)]
        gt_label = [list() for _ in range(model.num_classes)]
        score = [list() for _ in range(model.num_classes)]
        size = [list() for _ in range(model.num_classes)]
        npos = [0] * model.num_classes
        
        for iteration in iter(range((epoch_size))):
            images, targets = next(batch_iterator)
           # print('im ', images)
            if iteration > train_end and iteration < train_end + 10:
                self.visualize_epoch(model, int(iteration) * int(self.cfg.DATASET.TRAIN_BATCH_SIZE), self.priorbox, writer, epoch, use_gpu)
            if iteration <= train_end:
                if use_gpu:
                    images = Variable(images.cuda())
                    targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno, volatile=True) for anno in targets]
                model.train()
                #train:
                _t.tic()
                # forward
                out = model(images, phase='train')

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)

                # some bugs in coco train2017. maybe the annonation bug.
                if loss_l.data[0] == float("Inf"):
                    continue
                if math.isnan(loss_l.data[0]):
                    continue
                if math.isnan(loss_c.data[0]):
                    continue

                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

                time = _t.toc()
                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]

                # log per iter
                log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data[0], cls_loss=loss_c.data[0])

                sys.stdout.write(log)
                sys.stdout.flush()
                
                if iteration == train_end:
                    # log per epoch
                    sys.stdout.write('\r')
                    sys.stdout.flush()
                    lr = optimizer.param_groups[0]['lr']
                    log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                        time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
                    sys.stdout.write(log)
                    sys.stdout.flush()
                 #   print(log)
                    # log for tensorboard
                    writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
                    writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
                    writer.add_scalar('Train/lr', lr, epoch)
                    
                    loc_loss = 0
                    conf_loss = 0
            if iteration > train_end:
             #   self.visualize_epoch(model, images[0], targets[0], self.priorbox, writer, epoch, use_gpu)
                #eval:
                if use_gpu:
                    images = Variable(images.cuda())
                    targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno, volatile=True) for anno in targets]
                model.eval()
                out = model(images, phase='train')

                # loss
                loss_l, loss_c = criterion(out, targets)
                
                if loss_l.data[0] == float('nan'):
                    continue
                if loss_c.data[0] == float('nan'):
                    continue

                out = (out[0], model.softmax(out[1].view(-1, model.num_classes)))

                # detect
                detections = self.detector.forward(out)

                time = _t.toc()

                # evals
                label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
                size = cal_size(detections, targets, size)
                
                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]

                # log per iter
                log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data[0], cls_loss=loss_c.data[0])
                #print(log)
                sys.stdout.write(log)
                sys.stdout.flush()
                if train_end == (epoch_size - 1):
                    # eval mAP
                    prec, rec, ap = cal_pr(label, score, npos)

                    # log per epoch
                    sys.stdout.write('\r')
                    sys.stdout.flush()
                    log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
                      time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    print(log)
                    # log for tensorboard
                    writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
                    writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
                    writer.add_scalar('Eval/mAP', ap, epoch)
                    viz_pr_curve(writer, prec, rec, epoch)
                    viz_archor_strategy(writer, size, gt_label, epoch)
                    
                
    def test_epoch_2(self, model, detector, output_dir, use_gpu):
        model.eval()
        test_image_dir = os.path.join('./data/', 'ship_test_v2/data_test/')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        vis = visdom.Visdom(server="http://localhost", port=8888)
        check_i = 0;
        _t = Timer()
        for root, dirs, files in os.walk(test_image_dir):
            num_images = len(files)
            num_classes = detector.num_classes
            all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
            empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    img_dir = test_image_dir + file
                    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
                 #   vis.images(img, win=1, opts={'title': 'Reals'})
                    preproc_ = self.train_loader.dataset.preproc
                    #preproc_for_test(image, self.resize, self.means)
                    # preproc.add_writer(writer, epoch)
                    # preproc.p = 0.6

                    # preproc image & visualize preprocess prograss
                    if preproc_ is not None:   
                        img = preproc_for_test(img,self.cfg.DATASET.IMAGE_SIZE, self.cfg.DATASET.PIXEL_MEANS)
                        img = torch.from_numpy(img)
                    scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
                    img = Variable( img.unsqueeze(0), volatile=True)
                    if use_gpu:
                    #images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
                        images = img.cuda()
                    else:
                        images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

                _t.tic()
                if check_i == 3:
                    vis.images(images[0], win=2, opts={'title': 'Reals'})
                    self.visTest(model, images[0].unsqueeze(0), self.priorbox, self.writer, 1, use_gpu)
                    
                out = model(images, phase='eval')
                detections = detector.forward(out)
                time = _t.toc()
                for im ,this_img in enumerate( images):
                  if check_i == 3:
                      if im == 0:
                          print('de ', detections[im])
                         # return
                  for j in range(1, num_classes):
                      cls_dets = list()
                      for det in detections[im][j]:
                        #  if det[0] > 0.5:
                            d = det.cpu().numpy()
                            score, box = d[0], d[1:]
                           # box *= scale
                            box = np.append(box, score)
                            
                            if score >= 0.45:
                                cls_dets.append(box)
                                vis.images(this_img, win=1, opts={'title': 'Reals'})
                                print('box ', box)
                                print('score ', score)
                      if check_i == 3:
                          self.showTestResult(self.writer,img_dir, cls_dets)
                          return
                               # if check_i == 1:
                          #      return
                    
                      if len(cls_dets) == 0:
                        cls_dets = empty_array
                      all_boxes[j][0] = np.array(cls_dets)
                    
                  
                check_i += 1  
                
            with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

            # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
            print('Evaluating detections')
            data_loader.dataset.evaluate_detections(all_boxes, output_dir)
                    
    def showTestResult(self,writer, img_dir, cls_dets):
        image_show = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        real_box = []
        for box in cls_dets:
            dets =  box * 768
            real_box.append(dets)
            xs = dets[ 0]
            ys = dets[ 1]
            x2s = dets[ 2] 
            y2s = dets[ 3]
            cv2.rectangle(image_show, (int(xs), int(ys)), (int(x2s), int (y2s)), (0, 255, 0), 1)
            print(xs, ys, x2s, y2s)
                    
        cv2.imwrite(os.path.join('./data/','0.png'), image_show)
        
        ovlap_boxes = self.get_overlap_boxes(real_box)
        ovlap_boxes = self.get_overlap_boxes(ovlap_boxes)
        img2 = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        for ovlap_box in ovlap_boxes:
            cv2.rectangle(img2, (int(ovlap_box[0]), int(ovlap_box[1])), (int(ovlap_box[2]), int (ovlap_box[3])), (0, 255, 255), 1)

        cv2.imwrite(os.path.join('./data/','1.png'), img2)

       # image_show = Image.fromarray(cv2.cvtColor(image_show,cv2.COLOR_BGR2RGB)) 
       # image_show = transform.ToTensor()(image_show)
       # x = vutils.make_grid(image_show.cuda().data, normalize=True, scale_each=True)
        #writer.add_image('module_feature_maps/feature_extractors.{}'.format(img_dir),x, 67)
    def if_overlap(self, box1, box2):
        box1_point_1 = [box1[0], box1[1]]
        box1_point_2 = [box1[2], box1[3]]
        box2_point_1 = [box2[0], box2[1]]
        box2_point_2 = [box2[2], box2[3]]
        
        if (box1_point_1[0] - box2_point_1[0]) * (box2_point_2[0] - box1_point_1[0]) > 0:
            if(box1_point_1[1] - box2_point_1[1]) * (box2_point_2[1] - box1_point_1[1]) > 0:
                return True
        if (box1_point_2[0] - box2_point_1[0]) * (box2_point_2[0] - box1_point_2[0]) > 0:
            if(box1_point_2[1] - box2_point_1[1]) * (box2_point_2[1] - box1_point_2[1]) > 0:
                return True
        
        return False
    def get_overlap_boxes(self, boxes):
        out_boxes = []
        for box in boxes:
            if_has_overlop = False
            for o_i, out_box in enumerate( out_boxes):
                if self.if_overlap(box, out_box):
                    x = min(box[0], out_box[0])
                    y = min(box[1], out_box[1])
                    x2 = max(box[2], out_box[2])
                    y2 = max(box[3], out_box[3])
                    out_boxes[o_i] = [x, y, x2, y2]
                    if_has_overlop = True
            if if_has_overlop == False:
                out_boxes.append(box)
        return out_boxes           
        
    def visTest(self, model, images, priorbox, writer, epoch, use_gpu):
        print('image shpe', images.shape)
      #  images_to_writer(writer, images)

        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model(images, 'feature'), module_name='feature_extractors', epoch=epoch)

    def visualize_epoch(self, model, idx, priorbox, writer, epoch, use_gpu):
        model.eval()

        #img_index = random.randint(0, len(data_loader.dataset)-1)

        # get img
       # print('idx ', idx)
        imgIdx = idx #self.train_loader.dataset.ids[int(idx)]
        images = self.train_loader.dataset.pull_image(imgIdx)
       # print('vi ', images)
        anno = self.train_loader.dataset.pull_anno(imgIdx)

        # visualize archor box
        viz_prior_box(writer, priorbox, images, epoch)

        # get preproc
        preproc_ = self.train_loader.dataset.preproc
       # preproc.add_writer(writer, epoch)
        # preproc.p = 0.6

        # preproc image & visualize preprocess prograss
        if preproc_ is not None:
            
            images, target = preproc_(images, anno)
        images = Variable( images.unsqueeze(0), volatile=True)
        if use_gpu:
            images = images.cuda()

        # visualize feature map in base and extras
        print('image shpe', images.shape)
        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model(images, 'feature'), module_name='feature_extractors', epoch=epoch)

        model.train()
        images.requires_grad = True
        images.volatile=False
        #base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)
        base_out = viz_module_grads(writer, model, model.base, images, images, 0.5, module_name='base', epoch=epoch)

        # TODO: add more...

    
    def configure_optimizer(self, trainable_param, cfg):
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


    def export_graph(self):
        self.model.train(False)
        dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
        # Export the model
        torch_out = torch.onnx._export(self.model,             # model being run
                                       dummy_input,            # model input (or a tuple for multiple inputs)
                                       "graph.onnx",           # where to save the model (can be a file or file-like object)
                                       export_params=True)     # store the trained parameter weights inside the model file
        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))


def train_model():
    s = Solver(ifTrain = True)
    s.train_model()
    return True

def test_model():
    s = Solver(ifTrain = False)
    s.test_model()
    return True
