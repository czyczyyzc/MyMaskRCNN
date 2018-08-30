import numpy as np
import tensorflow as tf
from .bbox import *

class RoisPoolingLayer(object):
    
    def __init__(self, mod_tra=None, img_shp=None):
        
        self.mod_tra = mod_tra
        self.img_shp = img_shp
        self.pol_dep = 256
        #self.pol_num = 200 if self.mod_tra else 600
        
    def rois_pooling_img(self, elems=None):
        
        rois     = elems[0 ]
        img_fets = elems[1:]
        
        roi_ymns, roi_xmns, roi_ymxs, roi_xmxs = tf.split(rois, 4, axis=-1)
        roi_hgts = roi_ymxs - roi_ymns + 1.0 #由于rois来源于proposals_layer，所以不需要再clip
        roi_wdhs = roi_xmxs - roi_xmns + 1.0
        roi_aras = roi_hgts * roi_wdhs
        roi_pmxs = tf.log(tf.sqrt(roi_aras)/224.0) / tf.log(2.0)
        roi_pmxs = tf.minimum(3, tf.maximum(0, 2+tf.cast(tf.round(roi_pmxs), dtype=tf.int32)))
        roi_pmxs = tf.squeeze(roi_pmxs, axis=-1)
        
        img_leh  = np.array([self.img_shp[0]-1.0, self.img_shp[1]-1.0], dtype=np.float32)
        img_leh  = np.tile(img_leh, [2])
        rois     = rois / img_leh
        
        fet_num  = len(img_fets)
        roi_fets = []
        roi_idxs = []
        for i in range(fet_num):
            img_fet      = img_fets[i]
            img_fet      = tf.expand_dims(img_fet, axis=0)
            idxs         = tf.where(tf.equal(roi_pmxs, i))
            rois_pmd     = tf.gather_nd(rois, idxs)
            roi_imxs     = tf.zeros(shape=[tf.shape(rois_pmd)[0]], dtype=tf.int32)
            rois_pmd     = tf.stop_gradient(rois_pmd)
            roi_imxs     = tf.stop_gradient(roi_imxs)
            roi_fets_pmd = tf.image.crop_and_resize(img_fet, rois_pmd, roi_imxs, self.roi_pol_siz, method='bilinear')
            roi_fets.append(roi_fets_pmd)
            roi_idxs.append(idxs        )
        roi_fets = tf.concat(roi_fets, axis=0)
        roi_idxs = tf.concat(roi_idxs, axis=0)[:, 0]
        _, idxs  = tf.nn.top_k(roi_idxs, k=tf.shape(roi_idxs)[0], sorted=True)
        idxs     = idxs[::-1]
        roi_fets = tf.gather(roi_fets, idxs) #(M, H, W, C)
        #roi_fets = tf.reshape(roi_fets, [self.pol_num]+self.roi_pol_siz+[self.pol_dep])
        return roi_fets

    def rois_pooling(self, rois=None, img_fets=None, roi_pol_siz=None):
        
        self.roi_pol_siz = roi_pol_siz
        elems    = [rois] + img_fets
        roi_fets = tf.map_fn(self.rois_pooling_img, elems, dtype=tf.float32,
                             parallel_iterations=10, back_prop=True, swap_memory=True, infer_shape=True)
        return roi_fets #(N, M, H, W, C)