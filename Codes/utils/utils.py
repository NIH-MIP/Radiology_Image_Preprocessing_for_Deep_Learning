#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Samira Masoudi
# Date:   11.07.2019
# -------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
from os.path import isdir,join,exists,split,dirname,basename
from os import makedirs,mkdir,getcwd,listdir
import cv2
import time
import random
import SimpleITK as sitk
from dicom import *
from .Annotation_utils import *
def Get_seed(A):
    Flag=True
    X,Y,Z=np.where(A>0)
    # print(A)
    if len(X)>0:
        return [X[0],Y[0],Z[0]],Flag
    else:
        Flag=False
        return [],Flag
def region_grow(vol, start_point, epsilon=5, fill_with=0):
    sizez = vol.shape[0] - 1
    sizex = vol.shape[1] - 1
    sizey = vol.shape[2] - 1

    items = []
    limited = [0]
    visited = np.zeros(vol.shape)
    # limit=0

    def enqueue(item):
        if item not in items:
            items.insert(0, item)

    def dequeue():
        s = items.pop()
        visited[s[0],s[1],s[2]]=1
        limited[0]+=1
        return s

    enqueue((start_point[0], start_point[1], start_point[2]))

    while not items == []:
        # print(items)
        z, x, y = dequeue()
        voxel = vol[z, x, y]
        #print( z, x, y,voxel)
        vol[z, x, y] = fill_with

        if x < sizex:
            tvoxel = vol[z, x + 1, y]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z, x + 1, y))
            if tvoxel==255:  enqueue((z, x + 1, y))

        if x > 0:
            tvoxel = vol[z, x - 1, y]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z, x - 1, y))
            if tvoxel==255:  enqueue((z, x - 1, y))

        if y < sizey:
            tvoxel = vol[z, x, y + 1]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z, x, y + 1))
            if tvoxel==255:  enqueue((z, x, y + 1))


        if y > 0:
            tvoxel = vol[z, x, y - 1]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z, x, y - 1))
            if tvoxel==255:  enqueue((z, x , y - 1))

        if z < sizez:
            tvoxel = vol[z + 1, x, y]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z + 1, x, y))
            if tvoxel==255:  enqueue((z + 1, x , y))

        if z > 0:
            tvoxel = vol[z - 1, x, y]
            # if abs(tvoxel - voxel) < epsilon:  enqueue((z - 1, x, y))
            if tvoxel==255:  enqueue((z-1, x, y))
    if limited[0]<50:
        visited=[]
    return visited,vol

def normalize(arr, N=255, eps=1e-6):
    """
    TO normalize an image by mapping its [Min,Max] into the interval [0,255]
    :param arr: Input (2D or 3D) array of image
    :param N: Scaling factor
    :param eps:
    :return: Normalized Image
    """
    # N=255
    # eps=1e-6
    arr = arr.astype(np.float32)
    output=N*(arr+600)/2000
    # output = N*(arr-np.min(arr))/(np.max(arr)-np.min(arr)+eps)
    return output

def Physical2array(array, origin, spacing):
    """ To convert an input physical values based on the Origin and Spacing of the original image
    :param array: Physical values
    :param origin: Origin of axes in the original image
    :param spacing: Spacing of the axes in the original image
    :return: Converted physical values into array
    """
    origin = origin(...,np.newaxis)
    spacing = spacing(..., np.newaxis)
    array -= origin
    output = array//spacing
    return output.astype(int)

def standardize(array, axis=0,  # axis where the z component occurs
                type='image'
                # options=['image','slide'] to do teh standardization over the whole image(2D or 3D) or per slice where we have 3D image
                ):
    if type =='image':
        MEAN = np.mean(array)
        array -= MEAN
        sd = np.sqrt(np.mean(np.square(array)))
        array /= sd + 1e-6
    else:
        Mean0 = np.mean(array, axis=(axis + 1 if axis < 2 else axis - 2), keepdims=True)
        Mean = np.mean(Mean0, axis=(axis + 2 if axis < 1 else axis - 1), keepdims=True)
        array -= Mean
        sd0 = np.mean(np.square(array), axis=(axis + 1 if axis < 2 else axis - 2), keepdims=True)
        sd = np.mean(sd0, axis=(axis + 2 if axis < 1 else axis - 1), keepdims=True)
        sd = np.sqrt(sd)
        array /= sd + 1e-6
    return np.clip(array, -3 * sd, 3 * sd)

def clip(Slice,WL,WW):
    """
    Clipping or windowing the Input 2D slice
    :param Slice: Input 2D array
    :param WL: Wl is the center of the threshold interval
    :param WW: WW is half the length of the threshold interval
    :return: Input 2D array which values are clipped at [WL-WW, WL+WW]
    """
    Slice[Slice < (WL - WW)] = np.floor(WL - WW)
    Slice[Slice > (WL + WW)] = np.floor(WL + WW)
    return Slice

def Non_zero_Slice(Slice):
    if np.sum(Slice)>30:
        # print(np.sum(Slice))
        return True
    return False
def save_masks_as_png(shape,Pol,output_dir):
    if not exists(output_dir):
        makedirs(output_dir)
    for key,Polygon in Pol.items():
        Mask=255*(polygon2mask(shape, Polygon).astype(np.uint8))
        mask=cv2.resize(Mask, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(join(output_dir,str(key)+'.png'),mask)
    return(1)
def save_annotations(array,Bounds,output_dir):
    if not exists(output_dir):
        makedirs(output_dir)
    for key, B in Bounds.items():
        img = cv2.rectangle(normalize(array[key,:,:]), (B[0],B[1]), (B[2],B[3]), (255, 0, 0), 3)
        cv2.imwrite(join(output_dir, str(key) + '.png'), img)
    return True
def save_slides_as_png(Image, axis=0, method='Normalize', output_dir='home/mip/Desktop/New_folder2'):
    if not exists(output_dir):
        makedirs(output_dir)
    if method=='standardization_per_image':
        Image=standardize(Image)
    for k in range(Image.shape[axis]):
        slide = Image[k, :, :]
        if method == 'Standardization_per_slide':
            P_slide = standardize(slide)
        elif method =='standardization_per_image':
            P_slide = slide
        elif method =='Normalize':
            P_slide=normalize(slide)
        else:
            P_slide = slide
        cv2.imwrite(join(output_dir,str(k)+'.png'),P_slide)
    return 1

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print
        "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print
        "Toc: start time not set"

def ensureDir(f):
    d = dirname(f)
    if not  exists(d):
        makedirs(d)


def Get_the_MAX_and_MIN(patient_list, dir):
    MAX=0
    MIN=1000
    for i, patient in enumerate(patient_list):
            Input_path = join(dir,patient)
            # Reading the image at the Input path
            try :
                CT = sitk.ReadImage(Input_path+'.nii')
            except Exception:
                CT = DicomRead(Input_path)
            CT_array=sitk.GetArrayFromImage(CT)
            MAX=np.maximum(MAX,np.max(CT_array))
            MIN = np.minimum(MIN, np.min(CT_array))
    return MAX,MIN
def Get_the_MEAN_and_SD(patient_list, dir):
    MEAN=0
    SD=0
    for i, patient in enumerate(patient_list):
            Input_path = join(dir,patient)
            # Reading the image at the Input path
            try :
                CT = sitk.ReadImage(Input_path+'.nii')
            except Exception:
                CT = DicomRead(Input_path)
            CT_array=sitk.GetArrayFromImage(CT)
            MEAN += np.mean(CT_array)
    MEAN = MEAN / (len(patient_list))
    for i, patient in enumerate(patient_list):
            Input_path = join(dir,patient)
            # Reading the image at the Input path
            try :
                CT = sitk.ReadImage(Input_path+'.nii')
            except Exception:
                CT = DicomRead(Input_path)
            CT_array=sitk.GetArrayFromImage(CT)
            SD += np.sqrt(np.mean(np.square(CT_array - MEAN)))
    return MEAN,SD/len(patient_list)