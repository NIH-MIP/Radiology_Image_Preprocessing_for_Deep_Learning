#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Samira Masoudi
# Date:   11.07.2020
# -------------------------------------------------------------------------------
from __future__ import print_function
import argparse
import sys
import json
import glob
import random
from dicom.Dicom_Tools import *
from utils.utils import *
import csv
from utils.Annotation_utils import *
from image.Nyul_preprocessing import *

def arg_def():
    # ---------------------------------------------------------------------------
    # Parse the commandline
    # ---------------------------------------------------------------------------
    Organ_Dic = {'brain': [80, 40], 'head subdural': [215, 75], 'head stroke1': [8, 32], 'head stroke2': [40, 40],
                 'head temporal bones': [2800, 600], 'head soft tissues': [375, 40],
                 'lungs': [1500, 600], 'chest mediastinum': [350, 50], 'abdomen soft tissues': [400, 50],
                 'liver': [150, 30], 'spinal soft tissues': [250, 50], 'spinal bone': [1800, 400], 'Bone': [2000, 300],'None':[4000,0]}
    parser = argparse.ArgumentParser(
        description='Extracting Labeled patches from dicom images at dicom_folder and save the results at target_folder')
    parser.add_argument('--dicom_folder', type=str,
                        default='C:\\Users\masoudis2\\Desktop\\white_paper\\Data\\Original', help='Input path')
    parser.add_argument('--target_folder', type=str,
                        default='C:\\Users\\masoudis2\\Desktop\\white_paper\\Data',
                        help='Output path')
    parser.add_argument('--mask_folder', type=str,
                        default='C:\\Users\\masoudis2\\Desktop\\white_paper\\Data\\Masks',
                        help='Path to Masks folder')
    parser.add_argument('--image_type', type=str, default='MRI',
                        help='Image type received at the intput, options are \'MRI\', \'CT\'')
    parser.add_argument('--Masking', type=bool, default=False,
                        help='If Masking is True a certain organ will be masked from the input image, given the masks already available with the same 3D size at \'mask folder\'')
    parser.add_argument('--output_image_extension', type=str, default=['png'],
                        help='Image type requested at the output, options are \'png\', \'jpeg\' and \'npy\'')
    parser.add_argument('--split_ratio', type=float, default=[.6, .2, 0.2], help='Random split for training, vaidation, and test')
    parser.add_argument('--Path_to_train_set', type=str, default=[], help='directory to json file implying the training set of folders: /home/mip/Mdrive_mount/MIP/MRIClinical/development/consecutive/train_set.json')
    parser.add_argument('--Path_to_validation_set', type=str, default=[], help='json file implying the training set of folders:/home/mip/Mdrive_mount/MIP/MRIClinical/development/consecutive/validation_set.json')
    parser.add_argument('--Path_to_test_set', type=str, default=[], help='directory to json file implying the training set of folders: /home/mip/Mdrive_mount/MIP/MRIClinical/development/consecutive/test_set.json')
    parser.add_argument('--Classes', default=[0, 1], help='Classes or Labels')
    parser.add_argument('--WL', type=int, default=None, help='Window Level for thresholding')
    parser.add_argument('--WW', type=int, default=None, help='Window Width for thresholding')
    parser.add_argument('--Windowing_Organ', type=str, default='None',
                        help='Organ windowing which can be one of the' + str(Organ_Dic.keys()))
    parser.add_argument('--Spacing', type=float, default=[1,1,1], help='No resizing if empty [], otherwise images are resized to have target spacing of [x,y,z]')
    parser.add_argument('--desired_zero_padded_size', type=int, default=[512, 512, 128],
                        help='No zero padding if empty [], otherwise zero pad the images into a final desired size')
    # parser.add_argument('--Cropping_size', type=int, default=[512,512], help='Images are cropped at bottom_right to this size')
    parser.add_argument('--Normalization_type', type=str, default='MIN_MAX',
                        help='Options are "MIN_MAX" and "STND", for CT Images"')
    parser.add_argument('--Normalization_population', type=str, default='Overall',
                        help='Options are "Overall" and "Per_image", for CT Images"')
    parser.add_argument('--Output_Pixel_Type', type=str, default='Uint8',
                        help='Options are "Uint8" and "Uint16", for CT Images"')
    parser.add_argument('--Image_format', type=str, default='Nifti',
                        help='Output image format, options are \'Nifti\' or \'Dicom\' ')
    parser.add_argument('--Image_format_PNG', type=bool, default=True,
                        help='If True, output image Slices will be saved in PNG format')
    parser.add_argument('--Standardization_type', type=str, default='Nyul',
                        help='Options are "MIN_MAX","Nyul", and "STND" for MRI Images')
    parser.add_argument('--Save_image_specifications', type=bool, default=True,
                        help='If True, saves image specifications in a CSV file "image_specifications.csv" for the preprocessing')

    args = parser.parse_args()
    print('[i] Input directory:         ', args.dicom_folder)
    print('[i] Output directory:       ', args.target_folder)
    if args.image_type == 'CT':
        masking_statement=''
        if args.Masking:
            masking_statement='masking'
            if not exists(args.mask_folder):
                ValueError('Please specify the path to masks as instructed')
        print(
            'Pre-processing the CT images includes windowing followed by normalization, ',masking_statement,', resizing, and zero padding. Images are going to be:')
        if args.WL is None:
            args.WL = Organ_Dic[args.Windowing_Organ][1]
        if args.WW is None:
            args.WW = Organ_Dic[args.Windowing_Organ][0] / 2
        print('                          clipped within the window of: [' + str(
                args.WL) + '-' + str(int(args.WW)) + ',' + str(args.WL) + '+' + str(int(args.WW)) + ']')
        print('                          normalized using: ', args.Normalization_type)

    else:
        masking_statement=''
        if args.Masking:
            masking_statement='masking'
            if not exists(args.mask_folder):
                ValueError('Please specify the path to masks as instructed')
        print(
            'Pre-processing for MR images includes N4-Bias Correction, Nyul Standarization, ',masking_statement,', space-matching followed by zero padding. Images are going to be:')
        print('                          processed by N4 bias correction to lose their bias field')

        print('                          standardized using Nyul standardization method')
    if len(args.Spacing) == 3:
        print('                          resampled to have the target spacing:', args.Spacing)
    elif len(args.Spacing) == 0:
        print('                          remain the same spacing')
    else:
        ValueError(args.Spacing, ' should include 3 elements in x, y, and z directions, respectively')
    # if args.Windowing_Organ is 'costumized' and (args.WL is None or args.WW is None):
    if len(args.desired_zero_padded_size) == 0:
        print('                          remain the same size')
    else:
        print('                          zero-padded to be of size: ', args.desired_zero_padded_size)
    return args


def main():
    args = arg_def()
    if not exists(args.target_folder):
        makedirs(args.target_folder)

    # Get a list of all patients
    Patient_List = [ f for f in listdir(args.dicom_folder) if isdir(join(args.dicom_folder,f))]
    Patient_List.sort()
    random.shuffle(Patient_List)
    try:
        all_patients = json.load(open(join(args.target_folder,'all_patients.json'),'r'))
    except Exception:
        all_patients = Patient_List
        with open(join(args.target_folder,'all_patients.json'), 'w') as output:
            json.dump(all_patients, output)
    try:
        train_patients = json.load(open(args.Path_to_train_set,'r'))
    except Exception:
        train_patients = Patient_List[:int(args.split_ratio[0]* len(Patient_List))]
        with open(join(args.target_folder, 'train_set.json'), 'w') as output:
            json.dump(train_patients, output)
    #
    try:
        validation_patients = json.load(open(args.Path_to_validation_set, 'r'))
    except Exception:
        validation_patients = Patient_List[int(args.split_ratio[0]* len(Patient_List)):int((args.split_ratio[0]+args.split_ratio[1])* len(Patient_List))]
        with open(join(args.target_folder, 'validation_set.json'), 'w') as output:
            json.dump(validation_patients, output)
    #
    try:
        test_patients = json.load(open(args.test_set, 'r'))
    except Exception:
        test_patients = Patient_List[int((args.split_ratio[0] + args.split_ratio[1]) * len(Patient_List)):]
        with open(join(args.target_folder, 'test_set.json'), 'w') as output:
            json.dump(test_patients, output)

    with open(join(args.target_folder,'image_specifications.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Original_Spacing','Original_Size','Organ_Size','Split']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        if args.image_type=='MRI':
            for i, patient in enumerate(Patient_List):
                Input_path = join(args.dicom_folder, patient, 't2')
                if isdir(Input_path):
                        image1 = DicomRead(Input_path)
                        data = sitk.GetArrayFromImage(image1)
                        # shift the data up so that all intensity values turn positive
                        data -= np.min(data)
                        # Removing the outliers with a probability of occuring less than 5e-3 through histogram computation
                        histo, bins = np.histogram(data.flatten(), 10)
                        histo = normalize(histo)
                        Bin = bins[np.min(np.where(histo < 5e-3))]
                        data = np.clip(data, 0, Bin)
                        image = sitk.GetImageFromArray(data)
                        image.SetSpacing(image1.GetSpacing())
                        image.SetOrigin(image1.GetOrigin())
                        image.SetDirection(image1.GetDirection())
                        #Bias Correction
                        # image_B = Dicom_Bias_Correct(image)
                        image_B = image
                        if args.Image_format=='Nifti':
                                NiftiWrite(image_B, join(args.target_folder,'Bias_field_corrected'),output_name = patient+'.nii', OutputPixelType=args.Output_Pixel_Type)
                        elif args.Image_format=='Dicom':
                                DicomWrite(image_B, join(args.target_folder,'Bias_field_corrected',patient),
                                   Referenced_dicom_image_directory=Input_path, OutputPixelType=args.Output_Pixel_Type)

                else:
                    continue

            train(train_patients, dir1=join(args.target_folder,'Bias_field_corrected'),
                               dir2=join(args.target_folder,'trained_model'+args.image_type+'.npz'))
            Model_Path = join(args.target_folder,'trained_model'+args.image_type+'.npz')
            f = np.load(Model_Path, allow_pickle=True)
            Model = f['trainedModel'].all()
            meanLandmarks = Model['meanLandmarks']

            Patient_List = json.load(open(join(args.target_folder,'all_patients.json'),'r'))
            for i, patient in enumerate(Patient_List):

                Input_path = join(args.target_folder, 'Bias_field_corrected', patient)
                print('Standardizing ...', basename(Input_path))
                try:
                    image_B = sitk.ReadImage(Input_path+'.nii')
                except Exception:
                    image_B = DicomRead(Input_path)

                image_B_S= transform(image_B,meanLandmarks=meanLandmarks)
                if args.Masking:
                    mask = sitk.ReadImage(join(args.mask_folder,  patient))
                    image_B_S_m= Dicom_matmul(mask, image_B_S)
                else:
                    image_B_S_m = image_B_S
                if len(args.Spacing):
                        image_B_S_m_R  = Resample_3D(image_B_S_m, [args.Spacing[0], args.Spacing[1], args.Spacing[2]])
                else:
                    image_B_S_m_R = image_B_S_m
                if len(args.desired_zero_padded_size):
                    image_B_S_m_R_Z = zero_pad_3D(image_B_S_m_R, [args.desired_zero_padded_size[0],args.desired_zero_padded_size[1],args.desired_zero_padded_size[2]], mask=None)
                # print(image_B_S_m_R_Z.GetSize())
                if args.Image_format == 'Nifti':
                        NiftiWrite(image_B_S_m_R_Z, join(args.target_folder, 'Pre_Processed'),
                                   output_name=patient+'.nii', OutputPixelType=args.Output_Pixel_Type)
                if args.Image_format == 'Dicom':
                        DicomWrite(image_B_S_m_R_Z, join(args.target_folder,'Pre_Processed',patient),
                                   Referenced_dicom_image_directory=join(args.dicom_folder,patient,'t2'), OutputPixelType=args.Output_Pixel_Type)
                if args.Image_format_PNG:
                          save_dicom_as_png_slices(image_B_S_m_R_Z,  join(args.target_folder,'Pre_Processed_PNG'), patient, OutputPixelType=args.Output_Pixel_Type)

                Split='Training'
                if patient in validation_patients:
                    Split='Validation'
                elif patient in test_patients:
                    Split='Test'
                writer.writerow({'Image': patient, 'Original_Spacing':image_B.GetSpacing(),'Original_Size':image_B.GetSize(),'Organ_Size':None,'Split':Split})
        elif args.image_type=='CT':
                for i, patient in enumerate(Patient_List):
                    Input_path = join(args.dicom_folder, patient)
                    if isdir(Input_path):
                        image = DicomRead(Input_path)  # Here we have our image
                        if args.Masking:
                            mask = sitk.ReadImage(join(args.mask_folder, patient))
                            image_m = Dicom_matmul(mask, image)
                        else:
                            image_m = image
                        if len(args.Spacing):
                            image_m_R = Resample_3D(image_m,
                                                        [args.Spacing[0], args.Spacing[1], args.Spacing[2]])
                        else:
                            image_m_R = image_m
                        image_m_R_W= clip_dicom(image_m_R,args.WL,args.WW)
                        NiftiWrite(image_m_R_W, join(args.target_folder, 'Windowed'),
                                   output_name=patient + '.nii', OutputPixelType=args.Output_Pixel_Type)
                        DicomWrite(image_m_R_W, join(args.target_folder, 'Windowed', patient),
                                   Referenced_dicom_image_directory=Input_path, OutputPixelType=args.Output_Pixel_Type)

                MEAN = 0
                VAR = 1
                MIN = 0
                MAX = 1
                if args.Normalization_population == 'overall':
                    if args.Normalization_type == 'STND':
                        print('Computing the Mean and SD values ...')
                        MEAN, SD = Get_the_MEAN_and_SD(train_patients,join(args.target_folder, 'Windowed'))
                        print('The Mean Value is obtained to be: {:.2f}'.format(MEAN))
                        print('The Standard Deviation Value is: {:.2f}'.format(SD))
                    elif args.Normalization_type == 'MIN_MAX':
                        print('Computing the Max and Min values ...')
                        MAX,MIN = Get_the_MAX_and_MIN(train_patients,join(args.target_folder, 'Windowed'))
                        print('The Max Value is obtained to be: {:.2f}'.format(MAX))
                        print('The Min Value is: {:.2f}'.format(MIN))
                Patient_List = json.load(open(join(args.target_folder, 'all_patients.json'), 'r'))
                for i, patient in enumerate(Patient_List):
                    Input_path = join(args.target_folder, 'Windowed', patient)
                    try:
                        image_m_R_W = sitk.ReadImage(Input_path + '.nii')
                    except Exception:
                        image_m_R_W = DicomRead(Input_path)
                    CT_array = sitk.GetArrayFromImage(image_m_R_W)
                    if args.Normalization_population == 'per_image':
                        if args.Normalization_type == 'STND':
                            print('Computing the Mean and SD values ...')
                            MEAN, SD = Get_the_MEAN_and_SD([patient], join(args.target_folder, 'Windowed'))
                        elif args.Normalization_type == 'MIN_MAX':
                            print('Computing the Max and Min values ...')
                            MAX, MIN = Get_the_MAX_and_MIN([patient], join(args.target_folder, 'Windowed'))
                    if args.Normalization_type == 'MIN_MAX':
                        CT_array= (CT_array - MIN) / (
                                (MAX - MIN) + int((MAX - MIN) == 0) * 1e-6)
                    if args.Normalization_type == 'STND':
                        CT_array = (CT_array - MEAN) / (SD + int(SD == 0) * 1e-6)

                    CT_array=((2**int(args.Output_Pixel_Type[4:]))-1)*CT_array
                    image_m_R_W_N=sitk.GetImageFromArray(CT_array)
                    image_m_R_W_N.SetSpacing(image_m_R_W.GetSpacing())
                    image_m_R_W_N.SetOrigin(image_m_R_W.GetOrigin())
                    image_m_R_W_N.SetDirection(image_m_R_W.GetDirection())

                    if len(args.desired_zero_padded_size):
                        image_m_R_W_N_Z = zero_pad_3D(image_m_R_W_N, [args.desired_zero_padded_size[0],
                                                                      args.desired_zero_padded_size[1],
                                                                      args.desired_zero_padded_size[2]], mask=None)
                    NiftiWrite(image_m_R_W_N_Z, join(args.target_folder, 'Pre_Processed'),
                               output_name=patient + '.nii', OutputPixelType=args.Output_Pixel_Type)
                    DicomWrite(image_m_R_W_N_Z, join(args.target_folder, 'Pre_Processed', patient),
                               Referenced_dicom_image_directory=join(args.dicom_folder,patient), OutputPixelType=args.Output_Pixel_Type)
                    if args.Image_format_PNG:
                        save_dicom_as_png_slices(image_m_R_W_N_Z, join(args.target_folder, 'Pre_Processed_PNG'),
                                                 patient, OutputPixelType=args.Output_Pixel_Type)

                    Split = 'Training'
                    if patient in validation_patients:
                        Split = 'Validation'
                    elif patient in test_patients:
                        Split = 'Test'
                    writer.writerow(
                        {'Image': patient, 'Original_Spacing':  image_m_R_W .GetSpacing(), 'Original_Size':  image_m_R_W .GetSize(),
                         'Organ_Size': None, 'Split': Split})

    return 0
if __name__ == '__main__':
    sys.exit(main())

