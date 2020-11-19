#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Samira Masoudi
# Date:   11.07.2019
# -------------------------------------------------------------------------------
from __future__ import print_function
import re
from PIL import Image, ImageDraw
from .utils import *

def mask2polygan(Mask):
    coor = cv2.findContours(Mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(coor)
    coord = np.squeeze(coor[0][0])
    return coord
def polygon2mask(image_shape, polygon):
    """Compute a mask from polygon.
    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    polygon : array_like.
        The polygon coordinates of shape (N, 2) where N is
        the number of points.
    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    Notes
    -----
    This function does not do any border checking, so that all
    the vertices need to be within the given shape.
    Examples
    --------
    >>> image_shape = (128, 128)
    >>> polygon = np.array([[60, 100], [100, 40], [40, 40]])
    >>> mask = polygon2mask(image_shape, polygon)
    >>> mask.shape
    (128, 128)
    """
    polygon = np.asarray(polygon)
    polygon0 = []
    for _i in range(polygon.shape[0]):
        polygon0.append((polygon[_i, 0], polygon[_i, 1]))
    polygon = polygon0

    img = Image.new('L', image_shape, 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    mask = ndimage.binary_fill_holes(mask)
    # vertex_row_coords, vertex_col_coords = polygon.T
    # fill_row_coords, fill_col_coords = draw.polygon(
    #     vertex_row_coords, vertex_col_coords, image_shape)
    # mask = np.zeros(image_shape, dtype=np.bool)
    # mask[fill_row_coords, fill_col_coords] = True
    return mask.T
def get_3D_mask_from_voi(Mask, voi_path, lesion_score=1):
        """
        :param Mask: a 3D matrix to which we want to add the annotaion that can be found at voi_path
        :param voi_path: Path to the .voi file where labels are stored. Note that labels descibe the of a certain lesion is specified at different slices,
        :param lesion_score: is a float number in (0,1) inetrval. score 1 lesions are tend to be more aggressive
        you can find the sample version of the labels in data folder.
        :return: Mask: Updated  3D matrix mask
        """
        voi_df = pd.read_fwf(voi_path)
        first_line = []
        last_line = []
        slice_number = []

        # collecting the slice numbers with the first and last line describing the label coordinates in .voi file
        for line in range(len(voi_df)):  # Go through the .voi file line by line
            line_specific = voi_df.iloc[line, :]  # Read each line
            # We used the specific characteristics in the content of the .voi file which may be specific to our annotation tool (Radiant)
            # Sample organization of the first 13 lines in our .voi file:
            # MIPAV VOI FILE
            # 255  # color of VOI - red component
            # 0  # color of VOI - green component
            # 0  # color of VOI - blue component
            # 255  # color of VOI - alpha component
            # 1  # number of slices for the VOI
            # 5  # slice number
            # 1  # number of contours in slice
            # 64  # number of pts in contour <Chain-element-type>1</Chain-element-type>
            # 261.409  309.846
            # 261.01 309.775
            # 260.583 309.564

            # Please change these code according to the contents of your annotation file
            as_list = line_specific.str.split(r"\t\t")[0]  # Get the describing title at each line
            if "# slice number" in as_list:  # In case we are at the line which specifies the slice number,
                slice_number.append(int(as_list[0]))  # take the slice number
                first_line.append(
                    line + 3)  # First line of coordinates starts right after 3 lines under the slice number
            if "# number of pts in contour <Chain-element-type>1</Chain-element-type>" in as_list:  # Where the coordinates end
                last_line.append(line + int(as_list[0]))
            if "# unique ID of the VOI" in as_list:
                UID=as_list[0]


        # Now we start reading the coordinates from first to the last line for each slice.
        for i in range(len(first_line)):
            # For each labeled slice:
            counter = 0
            X = np.zeros((last_line[i] + 1 - first_line[i], 1))
            Y = np.zeros((last_line[i] + 1 - first_line[i], 1))
            for j in range(first_line[i], last_line[i] + 1):
                # reading coordinates line by line
                line_specific = voi_df.iloc[j, :]
                Coords_str = line_specific.str.split(r"\t")[0][0]
                for m in re.finditer("\d+.\d+\s", Coords_str):
                    Y[counter] = float(m.group(0))
                for m in re.finditer("\s\d+.\d+", Coords_str):
                    X[counter] = float(m.group(0))
                counter += 1
            # Here we have the labels aligned with arrays, otherwise we had to convert the physical values specified
            # by labels into their equivalent index of the arrays
            coords = np.hstack((X, Y)).astype(int)
            # print(coords)
            # print(slice_number[i])
            Mask[slice_number[i],:,:] = 255 * lesion_score*(polygon2mask((Mask.shape[1],Mask.shape[2]), coords).astype(np.uint8))
            # print(Mask[slice_number[i],:,:].max())
            del X, Y, coords
        return Mask,UID


def get_3D_mask_from_voi_at_slice(Mask, loc_dict):
        # print(slice_number[i])
        keys = np.array(list(loc_dict.keys()))
        keys.sort()
        for i,key in enumerate(keys):
            Mask[i, :, :] = 255 * (
                polygon2mask((Mask.shape[1], Mask.shape[2]), loc_dict[key]).astype(np.uint8))
        # Mask[1, :, :] = 255 * (
        #     polygon2mask((Mask.shape[1], Mask.shape[2]), loc_dict[slice_number2]).astype(np.uint8))
        return Mask
def Update_labels(Dic,dir,dir_save,labels):
    for i,lbl in enumerate(labels):
        label=join(dir,lbl)
        voi_df = pd.read_fwf(label)
        with open(join(dir_save,'New_'+lbl),mode='w') as f:
            f.write('MIPAV VOI FILE\n')
            for line in range(len(voi_df)):  # Go through the .voi file line by line
                line_specific = voi_df.iloc[line, :]
                line_sp= voi_df.iloc[line, :].str.split(r"\n")[0][0] # Read each line
                as_list = line_specific.str.split(r"\t\t")[0]  # Get the describing title at each line
                if "# slice number" in as_list:  # In case we are at the line which specifies the slice number,
                    slice_number_old=as_list[0]
                    # print(slice_number_old)
                    slice_number_new= Dic[int(slice_number_old)]
                    line_sp=str(slice_number_new)+'\t\t# slice number'
                f.write(line_sp+'\n')
    return 1

def Update_label(Mask_array,UID, dir_save,label):
        if not exists(dir_save):
            makedirs(dir_save)
        with open(join(dir_save,label),mode='w') as f:
            f.write('MIPAV VOI FILE\n')
            f.write('0\t\t# curvelement_type of the VOI <Source-image></Source-image><ViewId>0</ViewId><z-flipped>0</z-flipped>\n')
            f.write('255\t\t# color of VOI - red component\n')
            f.write('0\t\t# color of VOI - green component\n')
            f.write('0\t\t# color of VOI - blue component\n')
            f.write('255\t\t# color of VOI - alpha component\n')
            Index=[]
            for i in range(Mask_array.shape[0]):
                if Non_zero_Slice(Mask_array[i,...]):
                    Index.append(i)
            # print(Index)
            f.write(str(len(Index))+'\t\t# number of slices for the VOI\n')
            for i, ind in enumerate(Index):
                print(ind)
                f.write(str(ind)+'\t\t# slice number\n')
                f.write('1\t\t# number of contours in slice\n')
                # print(Mask_array[ind,...].shape)
                # cv2.imwrite(join(dir_save,str(ind)+str(i)+'.png'),Mask_array[ind,...])
                coor=cv2.findContours((np.array(Mask_array[ind,...])).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                print(coor)
                true_count=[]
                TC=0
                if len(coor[0])>1:
                    for tc in range(len(coor[0])):
                        true_count.append((coor[0][tc]).shape[0])
                    TC=np.argmax(true_count)
                coord=np.squeeze(coor[0][TC])
                    # mas=polygon2mask((Mask_array.shape[1],Mask_array.shape[2]), coord.astype(int))
                    # cv2.imwrite(join(dir_save,str(ind)+str(i)+'_new.png'),255.0*mas)
                print(coord)
                if len(coord.shape)>=2:
                    f.write(str(coord.shape[0])+'\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>\n')
                    for j in range(coord.shape[0]):
                        # print(str(coord[i]))
                        f.write(str(coord[j][0].astype(float))+' '+str(coord[j][1].astype(float))+'\n')
            f.write(UID+'\t\t# unique ID of the VOI\n')
        return 1

def get_UID(voi_path):
    voi_df = pd.read_fwf(voi_path)
    first_line = []
    last_line = []
    slice_number = []

    # collecting the slice numbers with the first and last line describing the label coordinates in .voi file
    for line in range(len(voi_df)):  # Go through the .voi file line by line
        line_specific = voi_df.iloc[line, :]  # Read each line
        as_list = line_specific.str.split(r"\t\t")[0]  # Get the describing
        if "# unique ID of the VOI" in as_list:
            UID = as_list[0]
        # title at each line
    return UID
def Update_label_at_slice(Mask_array, slice_number):
    loc_dict={}
    for i in range(Mask_array.shape[0]):
        coor = cv2.findContours((np.array(Mask_array[i, ...])).astype('uint8'), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        true_count = []
        TC = 0
        if len(coor[0]) > 1:
            for tc in range(len(coor[0])):
                true_count.append((coor[0][tc]).shape[0])
            TC = np.argmax(true_count)
        coord = np.squeeze(coor[0][TC])
        # print(coord.shape)
        # print(np.shape(np.vstack((coord[:,1],coord[:,0]))))
        loc_dict.update({slice_number+i:np.vstack((coord[:,1],coord[:,0])).T})
    return loc_dict
def Update_all_labels(loc_dict,dir_save,name,UID):
    keys = np.array(list(loc_dict.keys()))
    keys.sort()
    with open(join(dir_save,name), mode='w') as f:
        f.write('MIPAV VOI FILE\n')
        f.write(
            '0\t\t# curvelement_type of the VOI <Source-image></Source-image><ViewId>0</ViewId><z-flipped>0</z-flipped>\n')
        f.write('255\t\t# color of VOI - red component\n')
        f.write('0\t\t# color of VOI - green component\n')
        f.write('0\t\t# color of VOI - blue component\n')
        f.write('255\t\t# color of VOI - alpha component\n')
        f.write(str(len(keys)) + '\t\t# number of slices for the VOI\n')

        for i,ind in enumerate(keys):
                print(ind)
                f.write(str(ind) + '\t\t# slice number\n')
                f.write('1\t\t# number of contours in slice\n')
                # coor = cv2.findContours((np.array(loc_dict[ind])).astype('uint8'), cv2.RETR_EXTERNAL,
                #                         cv2.CHAIN_APPROX_NONE)
                # true_count = []
                # TC = 0
                # if len(coor[0]) > 1:
                #     for tc in range(len(coor[0])):
                #         true_count.append((coor[0][tc]).shape[0])
                #     TC = np.argmax(true_count)
                # coord = np.squeeze(coor[0][TC])
                # print(coord)
                # mas=polygon2mask((Mask_array.shape[1],Mask_array.shape[2]), coord.astype(int))
                # cv2.imwrite(join(dir_save,str(ind)+str(i)+'_new.png'),255.0*mas)
                # print(coord)
                coord=loc_dict[ind]
                if len(coord.shape) >= 2:
                    f.write(str(coord.shape[0]) + '\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>\n')
                for j in range(coord.shape[0]):
                        # print(str(coord[i]))
                        f.write(str(coord[j][1].astype(float)) + ' ' + str(coord[j][0].astype(float)) + '\n')
        f.write(UID + '\t\t# unique ID of the VOI\n')
    return 1


def get_ROI_slice_loc(voi_path):
    voi_df = pd.read_fwf(voi_path)
    first_line = []
    last_line = []
    slice_number = []
    Bound_dict = {}
    loc_dict = {}

    # Find the location of the slice numbers and last line
    for line in range(len(voi_df)):
        line_specific = voi_df.iloc[line, :]
        as_list = line_specific.str.split(r"\t\t")[0]
        # print(as_list)
        if "# slice number" in as_list:
            slice_number.append(int(as_list[0]))
            first_line.append(line + 3)
        if "# number of pts in contour <Chain-element-type>1</Chain-element-type>" in as_list:
            last_line.append(line + int(as_list[0]))
    print(first_line,last_line)
    for i in range(len(first_line)):
        counter = 0
        X = np.zeros((last_line[i] + 1 - first_line[i], 1))
        Y = np.zeros((last_line[i] + 1 - first_line[i], 1))
        # print(slice_number[i])
        for j in range(first_line[i], last_line[i] + 1):
            # if counter<2:
            line_specific = voi_df.iloc[j, :]
            Coords_str = line_specific.str.split(r"\t")[0][0]
            for m in re.finditer("\d+.\d+\s", Coords_str):
                X[counter] = float(m.group(0))
            for m in re.finditer("\s\d+.\d+", Coords_str):
                Y[counter] = float(m.group(0))
            counter += 1
        coord = np.hstack((Y, X))
        # coords = Physical2array(coord, org, spc)
        # Start_x=MEAN[0]-64
        # loc_dict.update({slice_number[i]: coords})
        Min = np.floor(np.min(coord, axis=0)).astype(int)
        Max = np.floor(np.max(coord, axis=0)).astype(int)
        loc_dict.update({slice_number[i]: coord})
        Bound_dict.update({slice_number[i]: [Min[1], Min[0], Max[1], Max[0]]})
        del X, Y, coord
    return loc_dict, Bound_dict
