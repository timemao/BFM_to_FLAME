'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import os
import cv2
import h5py
import numpy as np
import chumpy as ch
#from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
import pickle

class Mesh(object):
    def __init__(self,v=None,f=None,filename=None):
        if(v is not None and f is not None):
            self.v=v
            self.f=f

        if(filename is not None):
            self.v,self.f=read_obj(filename)

    def write_obj(self,path):
        save_obj(path,self.v,self.f+1)

def convert_mesh(mesh, corr_setup):
    v = np.vstack((mesh.v, np.zeros_like(mesh.v)))
    return Mesh(corr_setup['mtx'].dot(v), corr_setup['f_out'])

FLAME_model_fname = 'D:\download_chrome\BFM_to_FLAME-main/model/generic_model.pkl'
model = load_model(FLAME_model_fname)

cached_data = np.load('D:\download_chrome\BFM_to_FLAME-main/data/BFM_to_FLAME_corr.npz', allow_pickle=True, encoding="latin1")

BFM2017_corr = cached_data['BFM2017_corr'].item()
BFM2009_corr = cached_data['BFM2009_corr'].item()
BFM2009_cropped_corr = cached_data['BFM2009_cropped_corr'].item()

print('bfm 2017: ',BFM2017_corr['mtx'].shape[1]/2)
print('bfm 2009: ',BFM2009_corr['mtx'].shape[1]/2)
print('bfm 2009 crop: ',BFM2009_cropped_corr['mtx'].shape[1]/2)

FLAME_mask_ids = cached_data['FLAME_mask_ids']

def convert_BFM_mesh_to_FLAME(u,v):
    w_pose = 1e-4
    w_shape = 1e-3
    w_exp = 1e-4

    BFM_mesh = Mesh(u,v)
    conv_mesh = convert_mesh(BFM_mesh, BFM2009_cropped_corr)

    scale = ch.ones(1)
    v_target = scale*ch.array(conv_mesh.v)
    dist = v_target[FLAME_mask_ids]-model[FLAME_mask_ids]
    pose_reg = model.pose[3:]
    shape_reg = model.betas[:300]
    exp_reg = model.betas[300:]
    obj = {'dist': dist, 'pose_reg': w_pose*pose_reg,
           'shape_reg': w_shape*shape_reg,
           'exp_reg': w_exp*exp_reg}

    ch.minimize(obj, x0=[scale, model.trans, model.pose[:3]])
    print("pose-fit: ",model.pose[:3])
    ch.minimize(obj, x0=[scale, model.trans, model.pose[np.hstack((np.arange(3), np.arange(6,9)))],
                         model.betas])
    print("pose-exp-fit: ", model.pose[:3])

    v_out = model.r/scale.r
    print("save obj...")
    return v_out, model.f+1

def test():
    import sys
    sys.path.append(r"D:\code\yolov7_for_faguang\train_angle")
    from utils_obj import save_obj, read_obj

    u,v=read_obj("input.obj")
    v,f=convert_BFM_mesh_to_FLAME(u,v)
    save_obj("output.obj",v,f)

if __name__ == '__main__':
    test()
