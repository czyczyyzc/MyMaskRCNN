import numpy as np
import tensorflow as tf
from .bbox import *

class BBoxesLayer(object):
    
    def __init__(self, img_shp=None):
        
        self.img_shp     = img_shp
        self.box_siz_min = 5
        self.box_prb_min = 0.7
        self.box_nms_pre = None
        self.box_nms_pst = 100   #200
        self.box_nms_max = 0.3   #0.2
        self.box_msk_min = 0.5
        self.box_msk_siz = [28, 28]
    
    def generate_boxs_img(self, elems=None):
        
        rois, roi_prbs_pst, roi_prds_pst, roi_num = elems
        rois         = rois        [0:roi_num]
        roi_prbs_pst = roi_prbs_pst[0:roi_num]
        roi_prds_pst = roi_prds_pst[0:roi_num]
        #取出最佳类的预测值
        box_clss = tf.argmax(roi_prbs_pst, axis=1)
        box_clss = tf.cast(box_clss, tf.int32)
        box_prbs = tf.reduce_max(roi_prbs_pst, axis=1)
        
        #设置一个roi索引，避免大量的gather操作(prds、msks)，节省内存，提升速度
        box_idxs = tf.range(tf.shape(rois)[0])
        #剔除背景box
        idxs     = tf.where(box_clss>0)
        box_idxs = tf.gather_nd(box_idxs, idxs)
        box_clss = tf.gather_nd(box_clss, idxs)
        box_prbs = tf.gather_nd(box_prbs, idxs)
        #剔除得分较低的box
        if self.box_prb_min is not None:
            idxs     = tf.where(box_prbs>=self.box_prb_min)
            box_idxs = tf.gather_nd(box_idxs, idxs)
            box_clss = tf.gather_nd(box_clss, idxs)
            box_prbs = tf.gather_nd(box_prbs, idxs)
        #进一步剔除过多的box
        if self.box_nms_pre is not None:
            box_nms_pre    = tf.minimum(self.box_nms_pre, tf.shape(box_idxs)[0])
            box_prbs, idxs = tf.nn.top_k(box_prbs, k=box_nms_pre, sorted=True)
            box_idxs = tf.gather(box_idxs, idxs)
            box_clss = tf.gather(box_clss, idxs)
        #根据box_idxs进行剩余的gather操作
        rois         = tf.gather(rois, box_idxs)
        box_idxs     = tf.stack([box_idxs, box_clss], axis=-1)  #如果box的预测是定类的话要加上这句
        roi_prds_pst = tf.gather_nd(roi_prds_pst, box_idxs)
        #还原出box以进行后续的滤除
        boxs = bbox_transform_inv(rois, roi_prds_pst)
        boxs = bbox_clip(boxs, [0.0, 0.0, self.img_shp[0]-1.0, self.img_shp[1]-1.0])
        #剔除过小的box
        idxs     = bbox_filter(boxs, self.box_siz_min)
        boxs     = tf.gather_nd(boxs,     idxs)
        box_clss = tf.gather_nd(box_clss, idxs)
        box_prbs = tf.gather_nd(box_prbs, idxs)
        #做逐类的nms
        #设置一个box索引，避免大量的concat操作(boxs、clss、prbs)，节省内存，提升速度
        box_idxs = tf.zeros(shape=[0, 1], dtype=tf.int64)
        box_clss_unq, idxs = tf.unique(box_clss)
        
        def cond(i, boxs, box_clss, box_prbs, box_clss_unq, box_idxs):
            box_cls_num = tf.shape(box_clss_unq)[0]
            c           = tf.less(i, box_cls_num)
            return c

        def body(i, boxs, box_clss, box_prbs, box_clss_unq, box_idxs):
            #选出对应类的boxs
            box_cls      = box_clss_unq[i]
            box_idxs_cls = tf.where(tf.equal(box_clss, box_cls))
            boxs_cls     = tf.gather_nd(boxs,     box_idxs_cls)
            box_prbs_cls = tf.gather_nd(box_prbs, box_idxs_cls)
            #进行非极大值抑制操作
            idxs         = tf.image.non_max_suppression(boxs_cls, box_prbs_cls, self.box_nms_pst, self.box_nms_max)
            box_idxs_cls = tf.gather(box_idxs_cls, idxs)
            # 保存结果
            box_idxs     = tf.concat([box_idxs, box_idxs_cls], axis=0)
            return [i+1, boxs, box_clss, box_prbs, box_clss_unq, box_idxs]

        i = tf.constant(0)
        [i, boxs, box_clss, box_prbs, box_clss_unq, box_idxs] = \
            tf.while_loop(cond, body, loop_vars=[i, boxs, box_clss, box_prbs, box_clss_unq, box_idxs], \
                          shape_invariants=[i.get_shape(), boxs.get_shape(), box_clss.get_shape(), box_prbs.get_shape(), \
                                            box_clss_unq.get_shape(), tf.TensorShape([None, 1])], \
                          parallel_iterations=10, back_prop=False, swap_memory=False)
        
        box_prbs = tf.gather_nd(box_prbs, box_idxs)
        box_num  = tf.minimum(self.box_nms_pst, tf.shape(box_idxs)[0])
        box_prbs, idxs = tf.nn.top_k(box_prbs, k=box_num, sorted=True)
        box_idxs = tf.gather   (box_idxs, idxs    )
        boxs     = tf.gather_nd(boxs,     box_idxs)
        box_clss = tf.gather_nd(box_clss, box_idxs)
        
        paddings = [[0, self.box_nms_pst-box_num], [0, 0]]
        boxs     = tf.pad(boxs,     paddings, "CONSTANT")
        paddings = [[0, self.box_nms_pst-box_num]]
        box_clss = tf.pad(box_clss, paddings, "CONSTANT")
        box_prbs = tf.pad(box_prbs, paddings, "CONSTANT")
        return boxs, box_clss, box_prbs, box_num
    
    def generate_boxs(self, rois=None, roi_prbs_pst=None, roi_prds_pst=None, roi_nums=None):
        
        elems = [rois, roi_prbs_pst, roi_prds_pst, roi_nums]
        boxs, box_clss, box_prbs, box_nums = \
            tf.map_fn(self.generate_boxs_img, elems, dtype=(tf.float32, tf.int32, tf.float32, tf.int32),
                      parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return boxs, box_clss, box_prbs, box_nums
    
    def generate_msks_img(self, elems=None):
        
        box_clss, box_msks_pst, box_num = elems
        #取出最佳类的预测值
        box_clss     = box_clss    [0:box_num]
        box_msks_pst = box_msks_pst[0:box_num]              #(M, C, 28, 28)
        box_idxs     = tf.range(box_num)
        box_idxs     = tf.stack([box_idxs, box_clss], axis=-1)
        #box_idxs    = tf.Print(box_idxs, [box_idxs, box_clss, box_num], message=None, first_n=None, summarize=100)
        box_msks_pst = tf.gather_nd(box_msks_pst, box_idxs) #(M, 28, 28)
        box_msks     = tf.sigmoid(box_msks_pst)
        paddings     = [[0, self.box_nms_pst-box_num], [0, 0], [0, 0]]
        box_msks     = tf.pad(box_msks, paddings, "CONSTANT")
        return box_msks
        
    def generate_msks(self, box_clss=None, box_msks_pst=None, box_nums=None):
        
        elems = [box_clss, box_msks_pst, box_nums]
        box_msks = tf.map_fn(self.generate_msks_img, elems, dtype=tf.float32,
                             parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return box_msks #(N, M, 28, 28)
        