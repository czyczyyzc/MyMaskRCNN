BACKBONE                   = "resnet101"
BACKBONE_STRIDES           = [4, 8, 16, 32, 64]
POST_NMS_ROIS_TRAINING     = 2000
POST_NMS_ROIS_INFERENCE    = 1000
RPN_NMS_THRESHOLD          = 0.7
POOL_SIZE                  = 7
MASK_POOL_SIZE             = 14
TRAIN_BN                   = False  # Defaulting to False since batch size is often small
FPN_CLASSIF_FC_LAYERS_SIZE = 1024
TRAIN_ROIS_PER_IMAGE       = 200
ROI_POSITIVE_RATIO         = 0.33
MASK_SHAPE                 = [28, 28]
RPN_BBOX_STD_DEV           = np.array([0.1, 0.1, 0.2, 0.2])
BBOX_STD_DEV               = np.array([0.1, 0.1, 0.2, 0.2])
DETECTION_MAX_INSTANCES    = 100
DETECTION_MIN_CONFIDENCE   = 0.7
DETECTION_NMS_THRESHOLD    = 0.3
RPN_ANCHOR_SCALES          = (32, 64, 128, 256, 512)
RPN_ANCHOR_RATIOS          = [0.5, 1, 2]
RPN_ANCHOR_STRIDE          = 1
RPN_NMS_THRESHOLD          = 0.7
BACKBONE_STRIDES           = [4, 8, 16, 32, 64]
FPN_CLASSIF_FC_LAYERS_SIZE = 1024
TOP_DOWN_PYRAMID_SIZE      = 256

























import numpy as np
import tensorflow as tf
from .bbox import *

class BBoxesLayer(object):
    
    def __init__(self, img_shp=None, img_num=None):
        
        self.img_shp     = img_shp
        self.img_num     = img_num
        self.box_siz_min = 5
        self.box_prb_min = 0.5
        self.box_nms_pre = None
        self.box_nms_pst = 100   #200
        self.box_nms_max = 0.3   #0.2
        self.box_msk_min = 0.5
        self.box_msk_siz = [28, 28]
        
    def generate_boxs(self, rois=None, roi_prbs_pst=None, roi_prds_pst=None, roi_imxs=None):
        
        #取出最佳类的预测值
        box_clss = tf.argmax(roi_prbs_pst, axis=1)
        box_clss = tf.cast(box_clss, tf.int32)
        box_prbs = tf.reduce_max(roi_prbs_pst, axis=1)
        
        #设置一个box索引，避免大量的gather操作(prds、msks)，节省内存，提升速度
        box_idxs = tf.range(tf.shape(rois)[0])
        #剔除背景box
        idxs     = tf.where(box_clss>0)
        box_clss = tf.gather_nd(box_clss, idxs)
        box_prbs = tf.gather_nd(box_prbs, idxs)
        box_idxs = tf.gather_nd(box_idxs, idxs)
        #剔除得分较低的box
        if self.box_prb_min is not None:
            idxs     = tf.where(box_prbs>=self.box_prb_min)
            box_clss = tf.gather_nd(box_clss, idxs)
            box_prbs = tf.gather_nd(box_prbs, idxs)
            box_idxs = tf.gather_nd(box_idxs, idxs)
        #根据box_idxs进行剩余的gather操作
        rois         = tf.gather(rois,     box_idxs)
        box_imxs     = tf.gather(roi_imxs, box_idxs)
        box_idxs     = tf.stack([box_idxs, box_clss], axis=-1)  #如果box的预测是定类的话要加上这句
        roi_prds_pst = tf.gather(roi_prds_pst, box_idxs)
        #还原出box以进行后续的滤除
        boxs = bbox_transform_inv(rois, roi_prds_pst)
        boxs = bbox_clip(boxs, [0.0, 0.0, self.img_shp[0]-1.0, self.img_shp[1]-1.0])
        #剔除过小的box
        idxs     = bbox_filter(boxs, self.box_siz_min)
        boxs     = tf.gather_nd(boxs,     idxs)
        box_clss = tf.gather_nd(box_clss, idxs)
        box_prbs = tf.gather_nd(box_prbs, idxs)
        box_imxs = tf.gather_nd(box_imxs, idxs)
        
        #做逐img逐cls的nms
        #设置一个box索引，避免大量的concat操作(boxs、clss、prbs、imxs)，节省内存，提升速度
        box_idxs = tf.zeros(shape=[0], dtype=tf.int32)
        def cond0(i, boxs, box_clss, box_prbs, box_imxs, box_idxs):
            c = tf.less(i, self.img_num)
            return c

        def body0(i, boxs, box_clss, box_prbs, box_imxs, box_idxs):
            
            box_idxs_img = tf.where(tf.equal(box_imxs, i))
            boxs_img     = tf.gather_nd(boxs,     box_idxs_img) #和box_idxs_img对应
            box_clss_img = tf.gather_nd(box_clss, box_idxs_img)
            box_prbs_img = tf.gather_nd(box_prbs, box_idxs_img)
            
            #进一步剔除过多的roi
            if self.box_nms_pre is not None:
                box_nms_pre  = tf.minimum(self.box_nms_pre, tf.shape(boxs_img)[0])
                box_prbs_img, idxs = tf.nn.top_k(box_prbs_img, k=box_nms_pre, sorted=True)
                boxs_img     = tf.gather(boxs_img,     idxs)
                box_clss_img = tf.gather(box_clss_img, idxs)
                box_idxs_img = tf.gather(box_idxs_img, idxs)
            
            #####################################
            box_idxs_kep       = tf.zeros(shape=[0], dtype=tf.int32)
            box_clss_unq, idxs = tf.unique(box_clss_img)
            def cond1(j, boxs_img, box_clss_img, box_prbs_img, box_clss_unq, box_idxs_kep):
                box_cls_num = tf.shape(box_clss_unq)[0]
                c           = tf.less(j, box_cls_num)
                return c

            def body1(j, boxs_img, box_clss_img, box_prbs_img, box_clss_unq, box_idxs_kep):
                #选出对应类的rois
                box_cls      = box_clss_unq[j]
                box_idxs_cls = tf.where(tf.equal(box_clss_img, box_cls))
                boxs_cls     = tf.gather_nd(boxs_img,     box_idxs_cls)
                box_prbs_cls = tf.gather_nd(box_prbs_img, box_idxs_cls)
                #进行非极大值抑制操作
                idxs         = tf.image.non_max_suppression(boxs_cls, box_prbs_cls, self.box_nms_pst, self.box_nms_max)
                box_idxs_cls = tf.gather(box_idxs_cls, idxs)
                # 保存结果
                box_idxs_kep = tf.concat([box_idxs_kep, box_idxs_cls], axis=0)
                return [j+1, boxs_img, box_clss_img, box_prbs_img, box_clss_unq, box_idxs_kep]

            j = tf.constant(0)
            [j, boxs_img, box_clss_img, box_prbs_img, box_clss_unq, box_idxs_kep] = \
                tf.while_loop(cond1, body1, loop_vars=[j, boxs_img, box_clss_img, box_prbs_img, box_clss_unq, box_idxs_kep], \
                              shape_invariants=[j.get_shape(), boxs_img.get_shape(), box_clss_img.get_shape(), \
                                                box_prbs_img.get_shape(), box_clss_unq.get_shape(), tf.TensorShape([None])], \
                              parallel_iterations=10, back_prop=False, swap_memory=True)
            
            box_prbs_img = tf.gather(box_prbs_img, box_idxs_kep)
            box_idxs_img = tf.gather(box_idxs_img, box_idxs_kep)
            box_num_img  = tf.minimum(self.box_nms_pst, tf.shape(box_idxs_img)[0])
            box_prbs_img, idxs = tf.nn.top_k(box_prbs_img, k=box_num_img, sorted=True)
            box_idxs_img = tf.gather(box_idxs_img, idxs)
            # 保存结果
            box_idxs     = tf.concat([box_idxs, box_idxs_img], axis=0)
            return [i+1, boxs, box_clss, box_prbs, box_imxs, box_idxs]
            
        i = tf.constant(0)
        [i, boxs, box_clss, box_prbs, box_imxs, box_idxs] = \
            tf.while_loop(cond, body, loop_vars=[i, boxs, box_clss, box_prbs, box_imxs, box_idxs], \
                          shape_invariants=[i.get_shape(), boxs.get_shape(), box_clss.get_shape(), \
                                            box_prbs.get_shape(), box_imxs.get_shape(), tf.TensorShape([None])], \
                          parallel_iterations=10, back_prop=False, swap_memory=True)
            
        boxs     = tf.gather_nd(boxs,     box_idxs)
        box_clss = tf.gather_nd(box_clss, box_idxs)
        box_prbs = tf.gather_nd(box_prbs, box_idxs)
        box_imxs = tf.gather_nd(box_imxs, box_idxs)
        return boxs, box_clss, box_prbs, box_imxs
    
    def generate_msks(self, boxs=None, box_clss=None, box_msks_pst=None):
        
        return 
        