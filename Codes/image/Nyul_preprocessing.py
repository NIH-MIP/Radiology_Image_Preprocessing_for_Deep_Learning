#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Samira Masoudi
# Date:   11.07.2019
# Based on Nyul et al. 2000: New variants of a method of MRI scale standardization
# Full version can be found at https://github.com/sergivalverde/MRI_intensity_normalization
# -------------------------------------------------------------------------------
from __future__ import print_function
import SimpleITK as sitk
from os import listdir
from scipy.interpolate import interp1d
import time
from utils import *
from dicom.Dicom_Tools import *
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


def getCdf(hist):
    """
        Given a histogram, it returns the cumulative distribution function.
    """
    aux = np.cumsum(hist)
    aux = aux / aux[-1] * 100
    return aux


def getPercentile(cdf, bins, perc):
    """
        Given a cumulative distribution function obtained from a histogram,
        (where 'bins' are the x values of the histogram and 'cdf' is the
        cumulative distribution function of the original histogram), it returns
        the x center value for the bin index corresponding to the given percentile,
        and the bin index itself.

        Example:

            import numpy as np
            hist = np.array([204., 1651., 2405., 1972., 872., 1455.])
            bins = np.array([0., 1., 2., 3., 4., 5., 6.])

            cumHist = getCdf(hist)
            print cumHist
            val, bin = getPercentile(cumHist, bins, 50)

            print "Val = " + str(val)
            print "Bin = " + str(bin)

    """
    b = len(bins[cdf <= perc])
    return bins[b] + ((bins[1] - bins[0]) / 2)



def getLandmarks(image, mask=None, showLandmarks=False,nbins=1024, pLow=1, pHigh=99,numPoints=10):
        """
            This Private function obtain the landmarks for a given image and returns them
            in a list like:
                [lm_pLow, lm_perc1, lm_perc2, ... lm_perc_(numPoints-1), lm_pHigh] (lm means landmark)

            :param image    SimpleITK image for which the landmarks are computed.
            :param mask     [OPTIONAL] SimpleITK image containing a mask. If provided, the histogram will be computed
                                    taking into account only the voxels where mask > 0.
            :param showLandmarks    Plot the landmarks using matplotlib on top of the histogram.

        """

        data = sitk.GetArrayFromImage(image)
        if mask is None:
            # Calculate useful statistics
            stats = sitk.StatisticsImageFilter()
            stats.Execute(image)
            mean =stats.GetMean()

            # Compute the image histogram
            histo, bins = np.histogram(data.flatten(), nbins)

            # Calculate the cumulative distribution function of the original histogram
            cdfOriginal = getCdf(histo)

            # Truncate the histogram (put 0 to those values whose intensity is less than the mean)
            # so that only the foreground values are considered for the landmark learning process
            histo[bins[:-1] < mean] = 0.0
        # else:
        #     # Calculate useful statistics
        #     dataMask = sitk.GetArrayFromImage(mask)
        #
        #     # Compute the image histogram
        #     histo, bins = np.histogram(data[dataMask > 0].flatten(), nbins, normed=True)
        #
        #     # Calculate the cumulative distribution function of the original histogram
        #     cdfOriginal = getCdf(histo)

        # Calculate the cumulative distribution function of the truncated histogram, where outliers are removed
        cdfTruncated = getCdf(histo)

        # Generate the percentile landmarks for  m_i
        perc = [x for x in range(0, 100, 100 // numPoints)]
        # Remove the first landmark that will always correspond to 0
        perc = perc[1:]

        # Generate the landmarks. Note that those corresponding to pLow and pHigh (at the beginning and the
        # end of the list of landmarks) are generated from the cdfOriginal, while the ones
        # corresponding to the percentiles are generated from cdfTruncated, meaning that only foreground intensities
        # are considered.

        landmarks = [getPercentile(cdfOriginal, bins[:-1], pLow)] + [getPercentile(cdfTruncated, bins[:-1], x) for x in perc] + [getPercentile(cdfOriginal, bins[:-1], pHigh)]
        # landmarks_org =  [getPercentile(cdfOriginal, bins[:-1], x) for x in [pLow]+perc+[pHigh]]
        return landmarks

def  landmarksSanityCheck(landmarks):
        Flag=True
        if not (np.unique(landmarks).size == len(landmarks)):
            for i in range(1, len(landmarks) - 1):
                if landmarks[i] == landmarks[i + 1]:
                    landmarks[i] = (landmarks[i - 1] + landmarks[i + 1]) / 2.0

                print( "WARNING: Fixing duplicate landmark.")

            if not (np.unique(landmarks).size == len(landmarks)):
                raise Exception('ERROR NyulNormalizer landmarks sanity check : One of the landmarks is duplicate. You can try increasing the number of bins in the histogram \
                (NyulNormalizer.nbins) to avoid this behaviour. Landmarks are: ' + str(landmarks))

        elif not (sorted(landmarks) == list(landmarks)):
            Flag=False

        return Flag
            # raise Exception(
            #     'ERROR NyulNormalizer landmarks sanity check: Landmarks in the list are not sorted, while they should be. Landmarks are: ' + str(
            #         landmarks))
def train(image_list,dir1,dir2,pLow=1, pHigh=99, sMin=1, sMax=99, numPoints=10,
              showLandmarks=False,nbins=1024):

        # Percentiles used to trunk the tails of the histogram
        if pLow > 10:
            raise ("NyulNormalizer Error: pLow may be bigger than the first lm_pXX landmark.")
        if pHigh < 90:
            raise ("NyulNormalizer Error: pHigh may be bigger than the first lm_pXX landmark.")

        allMappedLandmarks = []

        for F2, image in enumerate(image_list):
                if True:#image!='7148914_20180608':
                        try:
                            print('Learning the landmarks from: ', image + '.nii')
                            img = sitk.ReadImage(join(dir1, image+'.nii'))
                        except Exception:
                            img=DicomRead(join(dir1, image))
                        landmarks = getLandmarks(img, showLandmarks=showLandmarks,nbins=nbins,pHigh=pHigh,pLow=pLow,numPoints=numPoints)
                        # Check the obtained landmarks ...
                        if landmarksSanityCheck(landmarks):
                                # Construct the linear mapping function
                                mapping = interp1d([landmarks[0], landmarks[-1]], [sMin, sMax], fill_value=(0,100))
                                # Map the landmarks to the standard scale
                                mappedLandmarks = mapping(landmarks)
                                # Add the mapped landmarks to the working set
                                allMappedLandmarks.append(mappedLandmarks)
                        else:
                            print('some issue with:', image)
                            continue
                    # print ("ALL MAPPED LANDMARKS: ")
                    # print  ( allMappedLandmarks)

        meanLandmarks = np.array(allMappedLandmarks).mean(axis=0)
            # Check the obtained landmarks ...
        landmarksSanityCheck(meanLandmarks)
        trainedModel = {
                'pLow': pLow,
                'pHigh': pHigh,
                'sMin': sMin,
                'sMax': sMax,
                'numPoints': numPoints,
                'meanLandmarks': meanLandmarks}

        np.savez(dir2, trainedModel=[trainedModel])
        return True
def shif_by_negative_value(array):
    array-=np.min(array)
    return array
def transform(image,meanLandmarks,mask=None):
    # Get the raw data of the image
    data = sitk.GetArrayFromImage(image)
    # data = standardize(data, type='image')
    # Calculate useful statistics
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)


    # Get the landmarks for the current image
    landmarks = getLandmarks(image, mask=mask, nbins=1024,pHigh=99,pLow=1,numPoints=10)
    landmarks = np.array(landmarks)
    # print(landmarks)
    # Check the obtained landmarks ...
    landmarksSanityCheck(landmarks)

    # Recover the standard scale landmarks
    standardScale = meanLandmarks


    # Construct the piecewise linear interpolator to map the landmarks to the standard scale
    mapping = interp1d(landmarks, standardScale, fill_value="extrapolate")

    # Map the input image to the standard space using the piecewise linear function

    flatData = data.ravel()
    tic()
    mappedData = mapping(flatData)
    mappedlandmarks = mapping(landmarks)
    histo,bins=np.histogram(mappedData, 1024)
    toc()
    mappedData = mappedData.reshape(data.shape)

    output = sitk.GetImageFromArray(shif_by_negative_value(mappedData.astype(int)))
    output.SetSpacing(image.GetSpacing())
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())

    return output


