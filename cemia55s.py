#CeMiA Core v 0.5.5s
#Update 050720
#Kashatus Lab@UVA

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, os.path
from skimage import measure,color
from skimage.morphology import thin, skeletonize, medial_axis
from skimage.measure import label, regionprops
from scipy.spatial import distance
import math
from copy import deepcopy
from scipy.ndimage import gaussian_filter
import random
from matplotlib.gridspec import GridSpec
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage.morphology import convex_hull_image
import datetime
import pandas as pd
from scipy.stats import multivariate_normal
from scipy import ndimage
from scipy import stats
import shutil


if __name__ == '__main__':
    print('This is a helper file for CeMiA, and it is not supposed to run directly.')
    print("  _  __        _         _             _         _    ")
    print(" | |/ /__ _ __| |_  __ _| |_ _  _ ___ | |   __ _| |__ ")
    print(" | ' </ _` (_-< ' \\/ _` |  _| || (_-< | |__/ _` | '_ \\")
    print(" |_|\\_\\__,_/__/_||_\\__,_|\\__|\\_,_/__/ |____\\__,_|_.__/")
    print("                            \N{COPYRIGHT SIGN} Kashatus lab, UVA, 2019\n")


# Set Address folder
def address():

    while True:
        address = input('Where are the files located?\n\n')

        try:
            file_list = os.listdir(address)
            print('\nAll set! You may run the next cell.')
            return address, file_list
            break
        except:
            print('###########')
            print('Please make sure the address that you entered exists.')
            print('###########')

# Set the max number of files to be samples from the target directory
def how_many(file_list):
    print('\nHow many files to sample? (If possible)')

    while True:
        how_many = input('Please enter a valid integer value below {}\n'.format(len(file_list)))

        try:
            how_many = int(how_many)
            if how_many <= len(file_list):
                return how_many
                break
            else:
                print('\nLets forget about mitochondria for a second and focus on math!')
        except:
            print('\nT{} is not an integer!'.format(how_many))

# Randomly selecting files from the target folder
def random_files(address,file_list,how_many):
    random_files = []

    if len(file_list) >= 7:
        a = list(range(len(file_list)))
        random.shuffle(a)
        for _ in range(len(file_list)):
            try:
                x = a.pop()
                sx = cv2.imread(address + '/'+file_list[x])
                sx.shape
                random_files.append(file_list[x])
                if len(random_files) >=how_many:
                    break
            except:
                pass

    else:
        a = list(range(len(file_list)))
        random.shuffle(a)
        for _ in range(len(file_list)):
            try:
                x = a.pop()
                sx = cv2.imread(address + '/'+file_list[x])
                sx.shape
                random_files.append(file_list[x])
                if len(random_files) >=how_many:
                    break
            except:
                pass

    print('\nThese files were validated and selected:\n\n',random_files)
    return random_files


# Standardize the shape and size of the input images
def shapeAdjust(img, size=(1024,1024), padColor=(0,0,0,255)):

        # Method to resize input images to a standard size

        # Image Size, and desired image size
        h, w = img.shape[:2]
        sh, sw = size

        #Choosing interpolation methods for shrinking and expanding
        if h > sh or w > sw:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC

        aspect = w / h

        # Accomodating differnt image sizes
        # Horizontal image
        if aspect > 1:
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0

        # Vertical image
        elif aspect < 1:
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0

        # Square image
        else:
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # Adjusting the pad
        if len(img.shape) is 3 and not isinstance(padColor, (
                list, tuple, np.ndarray)):
            padColor = [padColor] * 3

        # Scale and pad the input image
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

# Cleaning the nuclei
def nucleus_filtering(fullpath_input, abspath, intensity_thresh,size_thresh,show_images, nuc_correction=True, diffused=False):

    # Reading the image and extracting its size
    image = cv2.imread(fullpath_input)
    image_size = max(image.shape[:2])

    # Replace the image with its adjusted version
    image = shapeAdjust(image)

    # Making copies for future use
    image_cp0 = deepcopy(image)

    # Keeping the nuclei (Blue)
    image_cp0[:,:,1] = 0
    image_cp0[:,:,2] = 0

    # Cleaning nuclei channel

    # Apply tophat to remove uneven illumination and make it sharper
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    image_cp0 = cv2.morphologyEx(image_cp0, cv2.MORPH_TOPHAT, kernel_tophat)


    # Create a gray image of nuclei, and making copies for future use
    nuc_gray = cv2.cvtColor(image_cp0, cv2.COLOR_BGR2GRAY)
    #nuc_gray = cv2.GaussianBlur(nuc_gray,(5,5),5)
    nuc_gray_cp = deepcopy(nuc_gray)

    if diffused:
        nuc_gray_tmp = np.reshape(nuc_gray,-1)
        v,p = np.histogram(nuc_gray_tmp[nuc_gray_tmp>0],25)
        mxLoc = np.argmax(v)
        diff = int(round((p[mxLoc] + p[mxLoc+1])/2))
        nuc_gray_cp = nuc_gray_cp - diff
        nuc_gray_cp[nuc_gray_cp<0]=0
        nuc_gray_cp = 255 - nuc_gray_cp
        nuc_gray_cp = nuc_gray_cp.astype('uint8')

    # Binarize the nuclei channel and remove noise
    _, nuc_bw = cv2.threshold(nuc_gray_cp, np.percentile(nuc_gray_cp[nuc_gray_cp>0], intensity_thresh), 255, cv2.THRESH_BINARY)  # to binary image
    nuc_bw  = nuc_bw.astype('uint8')
    nuc_bw = cv2.medianBlur(nuc_bw,5)


    labels = measure.label(nuc_bw, neighbors=8, background=0)
    mask1 = np.zeros(nuc_bw.shape, dtype="uint8")

    for label in np.unique(labels):

        if label == 0:
            continue

        labelMask = np.zeros(nuc_bw.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if diffused:
            sz = 150
        else:
            sz = 450
        if numPixels > sz:
            kernel_mask1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            labelMask = cv2.morphologyEx(labelMask, cv2.MORPH_CLOSE, kernel_mask1)
            mask1 = cv2.add(mask1, labelMask)

    kernel_mask1_erode = np.ones((5,5),np.uint8)
    mask1 = cv2.erode(mask1,kernel_mask1_erode,iterations = 1)

    labels = measure.label(mask1, neighbors=8, background=0)
    mask = np.zeros(mask1.shape, dtype="uint8")

    for label in np.unique(labels):

        if label == 0:
            continue

        labelMask = np.zeros(mask1.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > size_thresh:
            if nuc_correction:
                chull = np.array(convex_hull_image(labelMask), dtype='uint8')
                mask = cv2.add(mask, chull)
            else:
                mask = cv2.add(mask, labelMask)

    if show_images:
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('Processed and cleaned nuclei',fontsize=16)
        plt.imshow(mask,'gray')

        plt.subplot(122)
        plt.axis('off')
        plt.title('Original image',fontsize=16)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        plt.show()

    return mask

def mysig_th(height,harshness,compactness,value,shift,offset):
    return (offset)+(height*((1/(1+np.exp(-(harshness*compactness)*(value-shift))))))

# Algorithm for binarizing the mitochondrial channel
def behind_the_moon_filter(img, thresh,thresh2, thresh3,bg_harshness=-0.5,sig_harshness=2,sig_strength=1, cleanup=False, method=1, adaptive=True, window_size=16,equalizer=0.15,steepness=0.5, remove_debries=True):

    # Method == 1 : Let it go
    # Method == 2 : Not one us

    # Separating Channels
    red = img[:,:,0]
    rd_sm = np.sum(red)
    green = img[:,:,1]
    gr_sm = np.sum(green)

    blue = cv2.GaussianBlur(img[:,:,2],(15,15),0)

    # Identifying the mitochondria channel

    if rd_sm > gr_sm:
        green = red.copy()

    # Making cleaner image, removing uneven illumination, and make sharper image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    green = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #green = clahe.apply(green)

    # Defining the mask for sampling mitochondiral background in the nucleus area

    (thresh_blue, blue_bw) = cv2.threshold(blue, np.percentile(blue[blue>0],thresh), 255, cv2.THRESH_BINARY)
    labels = measure.label(blue_bw, neighbors=8, background=0)

    # Use the largest mask to sample background (in case there is noise in nucleus channel)
    numPixels = []
    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(blue_bw.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels.append(cv2.countNonZero(labelMask))

    numPixels.index(max(numPixels))
    mask_blue = np.zeros(blue_bw.shape, dtype="uint8")
    mask_blue[labels == numPixels.index(max(numPixels))+1] = 255

    # Clean the mask and apply it to mitochondrial channel
    kernel =  np.ones((5,5),np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    g_temp =cv2.bitwise_and(mask_blue,green)

    # Measure the mean pixel intensity in masked area of the mitochondrial channel
    green_thresh = thresh2*np.mean(g_temp[g_temp>0])
    #print("\nInternal Thresh: ", thresh2)
    (thresh_green, green_bw) = cv2.threshold(green, green_thresh, 255, cv2.THRESH_BINARY)
    green_bw = cv2.medianBlur(green_bw,5)

    try:

        # Measuring the statistics of the background sampling area
        x = g_temp.reshape(g_temp.shape[0]*g_temp.shape[1])
        x = x[x>0]

        # Signal threshold for Not one of US Method
        bg_mean = np.mean(x)
        bg_std = np.std(x)
        sig1 = sig_strength * bg_mean + sig_harshness * bg_std

        iqr = np.percentile(x,75) - np.percentile(x,25)
        x = x[x<(np.percentile(x,75)+1.5*iqr)]
        bg_mean = np.mean(x)
        bg_std = np.std(x)
        print('\nAverage background: ',bg_mean)

        # Signal threshold method 1- Assuming normal distribution
        sig = bg_mean + 2 * bg_std

        if np.isnan(sig):
            pass
    except:

        # Use the default values if measurements fail
        print('Using default values')

        sig = 3
        sig1 = sig_strength * 1 + sig_harshness * 1

    if np.isnan(sig):

        #Use the default value if measurements fail
        print('Using default values')
        sig = 3
        sig1 = sig_strength * 1 + sig_harshness * 1

    # Thresholding the whole image (single cell) method 1 - Let it go method
    x_g = green.reshape(g_temp.shape[0]*g_temp.shape[1])
    iqr = np.percentile(x_g,75) - np.percentile(x_g,25)
    x_g = x_g[x_g>0]
    x_g = x_g[x_g<(np.percentile(x_g[x_g>sig],75)+1.5*iqr)]
    sig_mean = np.mean(x_g)
    sig_std = np.std(x_g)
    bg = sig_mean - bg_harshness * sig_std

    # Thresholding the whole image (single cell)

    if adaptive:
        values_mat = np.zeros(((int((green.shape[0]/window_size))), (int((green.shape[0]/window_size)))))
        values_array = []
        pre_processed = []
        post_processed = []
        adapt_factors = []

        canvas = np.zeros((green.shape[0],green.shape[1]), dtype='uint8')

        sum_img = deepcopy(green)

        for i in range(0,int(green.shape[0]/window_size)):


            for j in range(0,int(green.shape[0]/window_size)):

                sub_sum_img = sum_img[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size]
                pre_processed.append(sub_sum_img)


                if np.mean(sub_sum_img) != 0:

                    #ave = np.mean(sub_sum_img[sub_sum_img>0])
                    ave = np.mean(sub_sum_img)

                    if np.isnan(ave):
                        values_mat[i,j] = 0
                        values_array.append(0)

                    else:
                        values_mat[i,j] = ave
                        values_array.append(ave)
                else:
                    values_mat[i,j] = 0
                    values_array.append(0)

        vals = np.array(values_array)

        vals = vals[vals>0]

        g_iqr = np.percentile(vals,75) - np.percentile(vals,25)


        if np.percentile(values_array,25) - 1.5*g_iqr < 0:
            min_okay = 0

        else:
            min_okay = np.percentile(values_array,25) - 1.5*g_iqr

        max_okay = np.percentile(values_array,75) + 1.5*g_iqr

        values_cleaned = []

        for i in values_array:
            if i < max_okay and i > min_okay:
                values_cleaned.append(i)

        compactness = 12/(np.max(values_cleaned) - np.min(values_cleaned))

        if method == 1:
            shifts = bg

        else:
            shifts = sig1

        for i in range(len(values_array)):

            adapt = mysig_th(2*equalizer,steepness,compactness,values_array[i],shifts,-equalizer)
            adapt_factors.append(1+adapt)

        for i in range(len(pre_processed)):
            if method == 1:
                new_sig = bg * adapt_factors[i]
            else:
                new_sig = sig1 * adapt_factors[i]

            _, masked = cv2.threshold(pre_processed[i], new_sig, 255, cv2.THRESH_BINARY)

            if remove_debries:
                kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # previously (3,3)
                masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_final)

            post_processed.append(masked)

        counter = 0

        for i in range(0,int(img.shape[0]/window_size)):

            for j in range(0,int(img.shape[0]/window_size)):

                canvas[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size] = post_processed[counter]
                counter = counter + 1

        pic_bw = deepcopy(canvas)

        adapt_factors = []

        for i in range(len(values_array)):

            adapt = mysig_th(2*equalizer,steepness,compactness,values_array[i],shifts,-equalizer)
            adapt_factors.append(adapt)

    else:

        if method == 1:
            # Let it go
            _,pic_bw = cv2.threshold(green, thresh3*int(bg), 255, cv2.THRESH_BINARY)

        elif method == 2:
            # Not one of us
            _,pic_bw = cv2.threshold(green, thresh3*int(sig1), 255, cv2.THRESH_BINARY)

    # Removing unwanted noise from the output
    pic_bw = cv2.medianBlur(pic_bw,3)

    output_image = deepcopy(pic_bw)

    updated_list_points = []
    list_points = list(zip(np.nonzero(pic_bw)[1],np.nonzero(pic_bw)[0]))

    dist_mean = np.mean(np.transpose(list_points),axis=1)
    dist_cov = np.cov(np.transpose(list_points))
    dist_rv = multivariate_normal(dist_mean,dist_cov)
    normalizer = dist_rv.pdf(dist_mean)

    for point in list_points:
        if (dist_rv.pdf(point)/normalizer) > 0.01:
            updated_list_points.append(point)

    output = np.zeros(pic_bw.shape, dtype="uint8")

    for pixel in updated_list_points:
            output[pixel[1],pixel[0]] = 255

    output_image = output
    output_image = cv2.medianBlur(output_image, 5)

    original = deepcopy(img)

    green = cv2.bitwise_and(green,output)
    # Identifying the mitochondria channel

    if rd_sm > gr_sm:
        red = cv2.bitwise_and(red,output)

    original[:,:,0] = red
    original[:,:,1] = green

    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    if cleanup == True:

        # Removing blobs from mitochondrial channel
        kernelb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        dial = cv2.dilate(pic_bw, kernelb)

        labelsp = measure.label(dial, connectivity=2)
        propsm = regionprops(labelsp)
        blob_area = []
        blob_distance = []
        blob_label = []

        for r in range(len(propsm)):
                blob_area.append(propsm[r].area)
                blob_distance.append(np.sqrt(np.power(propsm[r].centroid[0],2) + np.power(propsm[r].centroid[1],2)))
                blob_label.append(propsm[r].label)

        blob = np.zeros(dial.shape, dtype="uint8")

        for r in range(len(blob_label)):

            if (blob_area[r] > 0.33 * np.max(blob_area) and (blob_distance[r] > np.percentile(blob_distance,50)) and (blob_label[r] !=0)):

                labelMask = np.zeros(blue_bw.shape, dtype="uint8")
                labelMask[labelsp == blob_label[r]] = 255
                #plt.figure()
                #plt.imshow(labelMask)
                #plt.show()
                blob = cv2.add(blob,labelMask)

        blob = 255 - blob
        output_image = cv2.bitwise_and(pic_bw,blob)

        try:
            # Output with blobs removed
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            pic_bw = cv2.morphologyEx(pic_bw, cv2.MORPH_OPEN, kernel2)

            clean_pic_bw = cv2.bitwise_and(pic_bw,blob)

            #Removing outlier mitochondria
            labelsp = measure.label(clean_pic_bw, connectivity=2)
            propsm = regionprops(labelsp)

            pts = pd.DataFrame(columns=['x','y'])
            lbs = pd.DataFrame(columns=['label'])

            for r in range(len(propsm)):
                op = pd.concat([pd.DataFrame([[propsm[r].centroid[0],propsm[r].centroid[1]]],columns=['x','y'])]*propsm[r].area,ignore_index=True)
                lb = pd.DataFrame([[propsm[r].centroid[0],propsm[r].centroid[1],propsm[r].label]],columns=['x','y','label'])
                pts = pts.append(op,ignore_index=True)
                lbs = lbs.append(lb,ignore_index=True)

            pts['mahala'] = mahalanobis(x=pts, data=pts)
            pts = pts[pts['mahala']>6]
            pts.drop_duplicates(inplace=True)
            pts = pts.merge(lbs,left_index=False, right_index=False, left_on=['x','y'], right_on=['x','y'])
            labelsm = pts['label']

            clean_pic_bw_new = np.zeros(pic_bw.shape, dtype="uint8")

            for label in np.unique(labelsp):
                if ((label in labelsm)&(label != 0)):

                    labelMask = np.zeros(blue_bw.shape, dtype="uint8")
                    labelMask[labelsp == label] = 255
                    #plt.figure()
                    #plt.imshow(labelMask)
                    #plt.show()
                    clean_pic_bw_new = cv2.add(clean_pic_bw_new,labelMask)

            new_pic_bw = clean_pic_bw - clean_pic_bw_new

            output_image = new_pic_bw

        except:
            print('Could not further clean the image')
            pass

    return original,green_thresh,output_image,mask_blue


# Sigmoid function for adaptive thresholding
def mysig(height,harshness,compactness,value,shift,offset):
    return (1+offset)+(height*(1 - (1/(1+np.exp(-(harshness*compactness)*(value-shift))))))


# Cell Segmentation from multi-cell images
def auto_segmentation(fullpath_input, abspath, namestring,filt,showimg,dilation_size,correct, nuc_intensity_threshold, nuc_size_threshold,empty_cell_thresh, hide, nuc_correct,diffused, mito_diffused, entangled):

    # Read and reshape image
    image = cv2.imread(fullpath_input)
    imageSize = max(image.shape[:2])
    image = shapeAdjust(image)

    # Copy images for manipulation
    image_cp1 = deepcopy(image)
    image_cp2 = deepcopy(image)
    image_cp3 = deepcopy(image)
    image_cp4 = deepcopy(image)

    # Keeping Nuclei
    image_cp3[:,:,1] = 0
    image_cp3[:,:,2] = 0

    # Detecting Mitochondria Channel
    red = image[:,:,1]
    rd_sm = np.sum(red)

    mito_channel = image[:,:,2]
    gr_sm = np.sum(mito_channel)

    if rd_sm > gr_sm:
        mito_channel = red.copy()

    # Removing blobs
    _,t2t = cv2.threshold(mito_channel, 0.5*np.max(mito_channel), 255, cv2.THRESH_BINARY)

    mitoLabels = measure.label(t2t, connectivity=2)
    propMito = regionprops(mitoLabels)
    good_mask = np.zeros(t2t.shape, dtype="uint8")

    for r in range(len(propMito)):
        if propMito[r].area > 2500:
            if propMito[r].label == 0:
                continue
            else:
                glabelMask = np.zeros(t2t.shape, dtype="uint8")
                glabelMask[mitoLabels == propMito[r].label] = 255
                good_mask = cv2.add(good_mask,glabelMask)

    good_mask = 255 - good_mask
    mito_channel = cv2.bitwise_and(mito_channel,good_mask)
    GR = deepcopy(mito_channel)

    # Apply nuclei mask from nuclei_filter
    nuclear_mask = nucleus_filtering(fullpath_input, abspath, nuc_intensity_threshold, nuc_size_threshold,False,nuc_correct,diffused)

    blue = cv2.bitwise_and(image_cp3[:,:,0],nuclear_mask)
    #blue = cv2.GaussianBlur(blue,(15,15),0)
    # Use nuclei to measure the mitochodrial background level across the image (Multi cell)

    _, blue_bw = cv2.threshold(blue, int(np.percentile(blue[blue>0],60)), 255, cv2.THRESH_BINARY)
    g_temp =cv2.bitwise_and(blue_bw,mito_channel)

    # Mitochondrial background signal estimation
    x = g_temp.reshape(g_temp.shape[0]*g_temp.shape[1])
    bg_mean = np.mean(x[x>0])
    bg_std = np.std(x[x>0])
    sig = bg_mean + 2 * bg_std
    #print('\nSampled threshold for empty cell removal: ',sig)

    # Detecting and removing deserted cells

    labels = measure.label(nuclear_mask, neighbors=8, background=0)

    mask = np.zeros(nuclear_mask.shape, dtype="uint8")
    mask_blue = np.zeros(nuclear_mask.shape, dtype="uint8")
    cnt = 0
    for label in np.unique(labels):
        cnt = cnt+1
        if label == 0:
            continue

        labelMask = np.zeros(nuclear_mask.shape, dtype="uint8")
        labelMask[labels == label] = 255

        # Growing each nuclei to search for overlapping mitochondria
        kernel_nuc_growth = np.ones((dilation_size, dilation_size), np.uint8)
        labelMaskd = cv2.dilate(labelMask,kernel_nuc_growth)

        # Search for mitochondria in the overlap area
        g_temp =cv2.bitwise_and(labelMaskd,mito_channel)
        _,local_mito_channel = cv2.threshold(g_temp, int(sig), 255, cv2.THRESH_BINARY )

        if np.sum(local_mito_channel) >= empty_cell_thresh: #250000
            mask_blue = cv2.add(mask_blue, labelMask)
    #print('cnt',cnt)
    kernel_nuc_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_blue = cv2.erode(mask_blue,kernel_nuc_erode)

    #EXTRA

    #plt.figure(figsize=(10,10))
    #plt.imshow(mask_blue)
    #plt.title('MASK BLUE')

    # Remove empty cells from the image
    if correct == True:
        image_cp1[:,:,0] = cv2.bitwise_and(image_cp1[:,:,0],mask_blue)
        image_cp2[:,:,0] = cv2.bitwise_and(image_cp2[:,:,0],mask_blue)

    nuclei_centers = []

    labels = measure.label(np.array(mask_blue), neighbors=8, background=0)
    nuc_labels = len(np.unique(labels))

    lbl = np.zeros(mask_blue.shape, dtype="uint8")


    for label in np.unique(labels):

        if label == 0:
            continue

        lbl2_tmp = np.zeros(mask_blue.shape, dtype="uint8")
        lbl2_tmp[labels == label] = label

        lbl = cv2.add(lbl,lbl2_tmp)

        mask_blue = deepcopy(lbl)
        mask_blue[mask_blue>0] = 255


    # Create the labeled image
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(image_cp1, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    if nuc_labels != 0:
        r, c = np.vstack(ndimage.center_of_mass(np.array(mask_blue), lbl, np.arange(nuc_labels) + 1)).T
        nuclei_centers = np.array(ndimage.center_of_mass(np.array(mask_blue), lbl, np.arange(nuc_labels) + 1))
        nuclei_centers[:, [0, 1]] = nuclei_centers[:, [1, 0]]

        # Annotating the nuclei
        for ri, ci, li in zip(r, c, range(1, nuc_labels + 1)):

            ax.annotate(li, xy=(ci, ri), fontsize=14, color='white')

        File_Path = abspath

        if not os.path.exists(File_Path):
            os.makedirs(File_Path)

        plt.savefig(File_Path + '/' + namestring.strip('.tif') + '_labeled.tif', dpi=300,
                    bbox_inches='tight')

        # Sharpening mito channel
        kernel_mito_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))
        mito_channel = cv2.morphologyEx(mito_channel, cv2.MORPH_TOPHAT, kernel_mito_th)

        image_cp1[:,:,0] = 0

        if np.sum(image_cp1[:,:,1]) > np.sum(image_cp1[:,:,2]):
            image_cp1[:,:,1] = mito_channel
        else:
            image_cp1[:,:,2] = mito_channel



        # Preparing mito channel for cell separation

        # Make grayscale image of mito
        mito_gray = cv2.cvtColor(image_cp1, cv2.COLOR_BGR2GRAY)
        # blur_mito = cv2.GaussianBlur(mito_gray, (25, 25),25)

        # Binarize and denoise mito
        _,mito_bw = cv2.threshold(mito_gray, np.percentile(mito_gray[mito_gray>0], filt), 255, cv2.THRESH_BINARY)
        mito_bw = cv2.medianBlur(mito_bw, 5)
        mito_bw = cv2.medianBlur(mito_bw, 3)

        if mito_diffused:

            ch,mt = improve_mito(image_cp4)
            _,mito_bw = cv2.threshold(mt, np.percentile(mt[mt>0], filt), 255, cv2.THRESH_BINARY)
            mito_bw = cv2.medianBlur(mito_bw, 3)
            mito_bw = cv2.medianBlur(mito_bw, 5)




        plt.figure(figsize=(20,10))
        if mito_diffused:
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(image_cp1,cv2.COLOR_RGB2BGR))
            plt.title('3X Amplified Mito Channel',fontsize=16)
        else:
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(image_cp1,cv2.COLOR_RGB2BGR))
            plt.title('Mito Channel',fontsize=16)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(mito_bw)
        plt.axis('off')
        plt.title('Global Mitochondrial Mask.\n Everything within the yellow masks would be captured and assigned to cells.',fontsize=16)

        if showimg:

            print('\nPlease wait...')
            # Separation
            ##################################################################
            mitoLabels = measure.label(mito_bw, connectivity=2)
            mitoProperties = regionprops(mitoLabels)

            nucLabels = measure.label(mask_blue, neighbors=8, background=0)
            nucProperties = regionprops(nucLabels)
            print('Number of Nuclei: ', len(nucProperties))


            sure_link_matrix = np.zeros((len(mitoProperties),len(nucProperties)))
            #print('sure_link_size: ', np.shape(sure_link_matrix))
            print(f'Linking {len(mitoProperties)} Mitochondria to {len(nucProperties)} Cells')
            # Reconstructing nuclei with bad staining

            list_dict_nuc = dict()
            list_dict_nuc2 = dict()
            list_dict_mito = dict()

            for nn in range(len(nucProperties)):

                nucMask = np.zeros(mito_bw.shape, dtype="uint8")
                nucMask2 = np.zeros(mito_bw.shape, dtype="uint8")

                try:
                    if int(nucProperties[nn].minor_axis_length) == 0:
                        ratio = int(nucProperties[nn].major_axis_length/1)
                    else:
                        ratio = int(nucProperties[nn].major_axis_length)/int(nucProperties[nn].minor_axis_length)

                    if ratio > 1.2:
                        radius = int(0.33*(nucProperties[nn].major_axis_length+nucProperties[nn].minor_axis_length))
                        centroid = nucProperties[nn].centroid
                        centroid = list(map(lambda x: int(x), centroid))
                        centroid = centroid[::-1]

                        cv2.circle(nucMask, tuple(centroid), radius,255,-1)
                        nucMask[nucLabels == nucProperties[nn].label] = 255

                        nucMask2[nucLabels == nucProperties[nn].label] = 255
                    else:
                        nucMask[nucLabels == nucProperties[nn].label] = 255
                        nucMask2[nucLabels == nucProperties[nn].label] = 255
                except:
                    pass
            # Keep track of nuclei points
                list_dict_nuc[nucProperties[nn].label] = list(zip(np.nonzero(nucMask)[1],np.nonzero(nucMask)[0]))
                list_dict_nuc2[nucProperties[nn].label] = list(zip(np.nonzero(nucMask2)[1],np.nonzero(nucMask2)[0]))

            #Keep track of mitochondrial points

            for mm in range(len(mitoProperties)):
                mitoMask = np.zeros(mito_bw.shape, dtype="uint8")
                mitoMask[mitoLabels == mitoProperties[mm].label] = 255
                list_dict_mito[mitoProperties[mm].label] = list(zip(np.nonzero(mitoMask)[1],np.nonzero(mitoMask)[0]))

            # Creating mitochondria-Nucleus linking matrix

            r = np.add(10000*nucLabels,mitoLabels)

            # Linking mitochondria to nuclei with obvious overlap

            sure_link_matrix = np.zeros((len(mitoProperties),len(nucProperties)))
            #print('sure_link_size2: ', np.shape(sure_link_matrix))

            for i in np.unique(r):
                if (int(i/10000)!=0) & (int(i%10000)!=0):
                    #print(int(i/1000), int(i%1000))
                    sure_link_matrix[int(i%10000)-1,int(i/10000)-1] = 1

            sure_link_matrix = pd.DataFrame(sure_link_matrix)
            sure_link_matrix.index = (sure_link_matrix.index + 1)
            sure_link_matrix.columns = sure_link_matrix.columns + 1

            # Identifying mitochondria with obvious linking which need more accurate treatment

            if entangled:
                mito_bypass = 3000
            else:
                mito_bypass = 100000


            single_link_special_indices = []
            single_link_indices = sure_link_matrix[sure_link_matrix.sum(axis=1) == 1].index
            for i in single_link_indices:
                if len(list_dict_mito[i]) > mito_bypass:
                    single_link_special_indices.append(i)
                    sure_link_matrix.loc[i] = 0

            single_link_indices = sure_link_matrix[sure_link_matrix.sum(axis=1) == 1].index

            print('\nNumber of mitochondria with obvious links to nuclei: ',len(single_link_indices))

            # Assigning mitochondria to nuclei with obvious linking

            print('Assigning mitochondria with obvious links to respective nuclei...')

            for i in single_link_indices:
                mito_num = i
                nuc_num = (sure_link_matrix.loc[i,:] != 0).to_numpy().nonzero()[0][0] + 1
                list_dict_nuc[nuc_num] = list_dict_nuc[nuc_num] + list_dict_mito[i]
                list_dict_nuc2[nuc_num] = list_dict_nuc2[nuc_num] + list_dict_mito[i]

            print('\nNumber of mitochondria with less obvious links: ',len(single_link_special_indices))

            # Assigning mitochondria to nuclei with less obvious linking

            print('Finding the best nuclei to assign mitochondria with less obvious links...')

            list_dict_nuc_temp = deepcopy(list_dict_nuc)
            list_dict_nuc_temp2 = deepcopy(list_dict_nuc2)

            nuc_means_list = []
            nuc_covs_list = []
            nuc_rv_list = []

            for nuc in sure_link_matrix.columns:
                nuc_means_list.append(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1))
                nuc_covs_list.append(np.cov(np.transpose(list_dict_nuc_temp[nuc])))
                nuc_rv_list.append(multivariate_normal(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1),
                                                       np.cov(np.transpose(list_dict_nuc_temp[nuc]))))

            pixel_prob_list = []
            points = dict()

            for i in single_link_special_indices:

                for nuc in sure_link_matrix.columns:

                    prob_pixels = list(map(lambda x: 10*(nuc_rv_list[nuc-1].pdf(x)),list_dict_mito[i]))

                    for j in range(len(prob_pixels)):
                        points[list_dict_mito[i][j]] =  points.get(list_dict_mito[i][j],[]) + [(nuc, prob_pixels[j])]


            assignments = []
            for k in points.keys():

                b = np.argmax(list(map(lambda x: (x[1]),points.get(k))))
                assignments.append((k,points.get(k)[b][0]))
                list_dict_nuc_temp[points.get(k)[b][0]] = list_dict_nuc_temp[points.get(k)[b][0]] + [k]
                list_dict_nuc_temp2[points.get(k)[b][0]] = list_dict_nuc_temp2[points.get(k)[b][0]] + [k]


            # Idetifying floating mitochondria

            no_link_indices = sure_link_matrix[sure_link_matrix.sum(axis=1) == 0].index

            print('\nNumber of floating mitochondria: ',len(no_link_indices))

            # Assigning mitochondria with no obvious linking

            print('Assigning floating mitochondria to the most probable nuclei...')

            nuc_means_list = []
            nuc_covs_list = []
            nuc_rv_list = []

            # Measuring cell distribution stats

            for nuc in sure_link_matrix.columns:
                nuc_means_list.append(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1))
                nuc_covs_list.append(np.cov(np.transpose(list_dict_nuc_temp[nuc])))
                nuc_rv_list.append(multivariate_normal(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1),
                                                       np.cov(np.transpose(list_dict_nuc_temp[nuc]))))

            list_dict_nuc_temp = deepcopy(list_dict_nuc)
            list_dict_nuc_temp2 = deepcopy(list_dict_nuc2)

            prob_means_list = []
            connections = []

            for i in no_link_indices:
                for nuc in sure_link_matrix.columns:
                    prob_mean = np.sum(list(map(lambda x: 10*(nuc_rv_list[nuc-1].pdf(x)),list_dict_mito[i])))
                    prob_means_list.append((i,nuc,prob_mean))

                b = np.argmax(list(map(lambda x: (x[2]),prob_means_list)))
                connections.append((prob_means_list[b][0],prob_means_list[b][1]))
                prob_means_list = []

            for c in connections:
                list_dict_nuc_temp[c[1]] = list_dict_nuc_temp[c[1]] + list_dict_mito[c[0]]
                list_dict_nuc_temp2[c[1]] = list_dict_nuc_temp2[c[1]] + list_dict_mito[c[0]]

            # Updating the linking matrix

            for link in connections:
                sure_link_matrix.loc[link[0],link[1]] = 1


            # Identifying multi-link mitochondria

            multi_link_indices = sure_link_matrix[sure_link_matrix.sum(axis=1) > 1].index

            print('\nNumber of mitochondria common between multiple nuclei: ',len(multi_link_indices))

            # Assigning mitochondria common between multiple nuclei

            print('Finding the best nuclei to assign mitochondria with multiple links')

            nuc_means_list = []
            nuc_covs_list = []
            nuc_rv_list = []

            for nuc in sure_link_matrix.columns:
                nuc_means_list.append(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1))
                nuc_covs_list.append(np.cov(np.transpose(list_dict_nuc_temp[nuc])))
                nuc_rv_list.append(multivariate_normal(np.mean(np.transpose(list_dict_nuc_temp[nuc]),axis=1),
                                                       np.cov(np.transpose(list_dict_nuc_temp[nuc]))))


            pixel_prob_list = []
            points = dict()

            for i in multi_link_indices:

                nucs = sure_link_matrix.loc[i,:].to_numpy().nonzero()[0]+1

                for nuc in nucs:
                    prob_pixels = list(map(lambda x: 10*(nuc_rv_list[nuc-1].pdf(x)),list_dict_mito[i]))

                    for j in range(len(prob_pixels)):
                        points[list_dict_mito[i][j]] =  points.get(list_dict_mito[i][j],[]) + [(nuc, prob_pixels[j])]

            assignments = []
            for k in points.keys():
                #print(k,list(map(lambda x: (x[1]),points.get(k))))
                b = np.argmax(list(map(lambda x: (x[1]),points.get(k))))
                assignments.append((k,points.get(k)[b][0]))
                list_dict_nuc_temp[points.get(k)[b][0]] = list_dict_nuc_temp[points.get(k)[b][0]] + [k]
                list_dict_nuc_temp2[points.get(k)[b][0]] = list_dict_nuc_temp2[points.get(k)[b][0]] + [k]

            #Creating quality control tags
            #################################################################

            quality_tags = dict()

            for key in list_dict_nuc.keys():

                x = np.array(list(map(lambda x: x[0],list_dict_nuc[key])))
                y = np.array(list(map(lambda x: x[1],list_dict_nuc[key])))

                quality_tags[key] = 'Good'

            #     if ((len(x[x<4]) + len(y[y<4]) + len(x[x>1020]) + len(y[y>1020])) > 8):

            #         quality_tags[key] = 'Bad'

            #     else:

            #         quality_tags[key] = 'Good'

            # for key in list_dict_nuc_temp.keys():

            #     x = np.array(list(map(lambda x: x[0],list_dict_nuc_temp[key])))
            #     y = np.array(list(map(lambda x: x[1],list_dict_nuc_temp[key])))

            #     #Measuring the ratio of the mitochondrial network touching the frame
            #     touching = (len(x[x<2]) + len(y[y<2]) + len(x[x>1022]) + len(y[y>1022]))

            #     if (np.round(touching/(len(x)+len(y)),4) > 0.0025):
            #         print('This needs work, may be we should only list it, and deal with it later.')
            #         quality_tags[key] = 'Bad'

            #################################################################

            cellBinaryMask = np.zeros(mito_bw.shape, dtype="uint8")
            singleCellMask = np.zeros(image.shape, dtype="uint8")
            image_rev = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rev[:,:,2] = cv2.bitwise_and(mask_blue,image_rev[:,:,2])

            for i in range(len(list_dict_nuc_temp2)):
                for pixel in list_dict_nuc_temp2[i+1]:
                    cellBinaryMask[pixel[1],pixel[0]] = 255

                updated_list_points = []
                list_points = list(zip(np.nonzero(cellBinaryMask)[1],np.nonzero(cellBinaryMask)[0]))

                dist_mean = np.mean(np.transpose(list_points),axis=1)
                dist_cov = np.cov(np.transpose(list_points))
                dist_rv = multivariate_normal(dist_mean,dist_cov)
                normalizer = dist_rv.pdf(dist_mean)

                for point in list_points:

                    #Outlier Removal
                    if (dist_rv.pdf(point)/normalizer) > 0.05:
                        updated_list_points.append(point)

                output = np.zeros(cellBinaryMask.shape, dtype="uint8")

                for pixel in updated_list_points:
                        output[pixel[1],pixel[0]] = 255

                cellBinaryMask = deepcopy(output)

                kernel_fill_cell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))  # kernel1cl = np.ones((12,12),np.uint8)
                cellBinaryMask = cv2.morphologyEx(cellBinaryMask, cv2.MORPH_CLOSE, kernel_fill_cell)


                singleCellMask[:,:,0] = cv2.bitwise_and(cellBinaryMask,image_rev[:,:,0])
                singleCellMask[:,:,1] = cv2.bitwise_and(cellBinaryMask,image_rev[:,:,1])
                singleCellMask[:,:,2] = cv2.bitwise_and(cellBinaryMask,image_rev[:,:,2])

                rows,cols = singleCellMask.shape[0],singleCellMask.shape[1]
                shift_x = singleCellMask.shape[0]/2 - nucProperties[i].centroid[1]
                shift_y = singleCellMask.shape[1]/2 - nucProperties[i].centroid[0]
                M = np.float32([[1,0,shift_x],[0,1,shift_y]])
                singleCellMask = cv2.warpAffine(singleCellMask,M,(cols,rows))

                # print(f'Cell {i+1} QC: {quality_tags[i+1]}')

                if quality_tags[i+1] == 'Good':

                    File_Path = abspath

                    try:
                        os.makedirs(File_Path + '/to_analyze/')
                        print()
                    except:
                        pass
                    full_path_to_image = File_Path + '/to_analyze/' + namestring.strip('.tif') + '_cell{}_{}.tif'.format(i+1,imageSize) #singleCellMask.shape[0]
                    cv2.imwrite(full_path_to_image , cv2.cvtColor(singleCellMask, cv2.COLOR_RGB2BGR))
                    single_cell_QC(abspath, full_path_to_image, entangled, hide)

                elif quality_tags[i+1] == 'Bad':

                    File_Path = abspath

                    try:
                        os.makedirs(File_Path + '/to_discard/')
                    except:
                        pass

                    cv2.imwrite(File_Path + '/to_discard/' + namestring.strip('.tif') + '_cell{}_{}.tif'.format(i+1,imageSize) , cv2.cvtColor(singleCellMask, cv2.COLOR_RGB2BGR))

                if hide == False:
                    # plt.figure(figsize=(10,10))
                    # plt.imshow(singleCellMask)
                    # plt.title(f'Cell {i+1}')
                    pass

                cellBinaryMask = np.zeros(mito_bw.shape, dtype="uint8")



    return mask_blue

# Garbage Collector and QC control for Cell Catcher export.
# Garbage Collector and QC control for Cell Catcher export.
def single_cell_QC(abspath, full_path_to_image ,entangled,hide=True):

    # Creating Junkyard
    try:
        os.makedirs(os.path.join(abspath,'to_discard'))
    except:
        pass

    try:
        # Read image
        image_sc_in = cv2.imread(full_path_to_image)
        image_sc = cv2.cvtColor(image_sc_in, cv2.COLOR_BGR2RGB)
        # Detecting Mitochondria Channel
        red = image_sc[:,:,0]
        rd_sm = np.sum(red)

        mito_channel_sc = image_sc[:,:,1]
        gr_sm = np.sum(mito_channel_sc)

        if rd_sm > gr_sm:
            mito_channel_sc = red.copy()

        # Identifying aliens
        _,t2t_sc = cv2.threshold(mito_channel_sc, 0.01*np.max(mito_channel_sc), 255, cv2.THRESH_BINARY)
        _,nuc_mask_sc = cv2.threshold(image_sc[:,:,2], 0.01*np.max(image_sc[:,:,2]), 255, cv2.THRESH_BINARY)

        mitoLabels_sc = measure.label(t2t_sc, connectivity=2)
        propMito_sc = regionprops(mitoLabels_sc)
        good_mask_sc = np.zeros(t2t_sc.shape, dtype="uint8")

        #Size based filtering
        area_list = []
        mask_counter = 0
        for r in range(len(propMito_sc)):
            area_list.append(propMito_sc[r].area)

        for r in range(len(propMito_sc)):
            if (propMito_sc[r].area > 0.001 * np.max(area_list)):
                if propMito_sc[r].label == 0:
                    continue
                else:
                    globalMask_sc = np.zeros(t2t_sc.shape, dtype="uint8")
                    globalMask_sc[mitoLabels_sc == propMito_sc[r].label] = 255

                    is_main = cv2.bitwise_and(nuc_mask_sc, globalMask_sc)
                    if np.sum(is_main) == 0:
                        good_mask_sc = cv2.add(good_mask_sc,globalMask_sc)
                    else:
                        mask_counter += 1

        #if no mito-blob touches the nucleus
        if mask_counter == 0:
            for r in range(len(propMito_sc)):
                if (propMito_sc[r].area > 0.001 * np.max(area_list)) and (propMito_sc[r].area < np.max(area_list)):
                    if propMito_sc[r].label == 0:
                        continue
                    else:
                        globalMask_sc = np.zeros(t2t_sc.shape, dtype="uint8")
                        globalMask_sc[mitoLabels_sc == propMito_sc[r].label] = 255
                        good_mask_sc = cv2.add(good_mask_sc,globalMask_sc)

        good_mask_sc = cv2.medianBlur(good_mask_sc, 3)
        good_mask_sc = 255 - good_mask_sc

        image_sc[:,:,1] = cv2.bitwise_and(image_sc[:,:,1],good_mask_sc)
        image_sc[:,:,2] = cv2.bitwise_and(image_sc[:,:,2],good_mask_sc)

        #Identifying and discarding cells that are touching image edges or missing mitochondria
        #Finding cell boundary
        xmx = np.max([np.max(np.where(image_sc[:,:,2].any(axis=1))[0]),np.max(np.where(image_sc[:,:,1].any(axis=1))[0])])
        xmn = np.min([np.min(np.where(image_sc[:,:,2].any(axis=1))[0]),np.min(np.where(image_sc[:,:,1].any(axis=1))[0])])

        ymx = np.max([np.max(np.where(image_sc[:,:,2].any(axis=0))[0]),np.max(np.where(image_sc[:,:,1].any(axis=0))[0])])
        ymn = np.min([np.min(np.where(image_sc[:,:,2].any(axis=0))[0]),np.min(np.where(image_sc[:,:,1].any(axis=0))[0])])

        qc_mask = np.zeros(t2t_sc.shape, dtype="uint8")
        qc_mask[xmx:,:] = 255
        qc_mask[:xmn,:] = 255
        qc_mask[:,ymx:] = 255
        qc_mask[:,:ymn] = 255

        c1 = cv2.bitwise_and(image_sc[:,:,0],qc_mask)
        c2 = cv2.bitwise_and(image_sc[:,:,1],qc_mask)
        c3 = cv2.bitwise_and(image_sc[:,:,2],qc_mask)
        touch_count = len(c1[c1!=0]) + len(c2[c2!=0]) + len(c3[c3!=0])

        mito_sum = np.sum(image_sc[:,:,0]) + np.sum(image_sc[:,:,1])

        if entangled:
            touch_thresh = 50
        else:
            touch_thresh = 200

        if ((mito_sum < 50000) or (len(c3[c3!=0]) > 2) or (touch_count > touch_thresh)):
            shutil.move(full_path_to_image, os.path.join(abspath,'to_discard', os.path.split(full_path_to_image)[1]))
            print(f'QC Failed: {full_path_to_image}')

            if (len(c3[c3!=0]) > 5) or (touch_count > touch_thresh):
                print('Reason: The cell is touching the image frame.', len(c3[c3!=0]),touch_count)

            else:
                print('Reason: Not enough mitochondrial content in the cell.')
            print(f'{full_path_to_image} discarded. (moved to "to_discard" direcoty)\n')

        else:
            print(f'QC Passed: {full_path_to_image}')

            if hide == False:
                plt.figure(figsize=(10,10))
                plt.imshow(image_sc)
                plt.title(f'{full_path_to_image}',fontsize=14)

            os.remove(full_path_to_image)
            cv2.imwrite(full_path_to_image , cv2.cvtColor(image_sc, cv2.COLOR_BGR2RGB))
    except:
        print(f'Saving {full_path_to_image}')
        pass

# Improve mito network quality for low signal images
def improve_mito(img2):

    img2[:,:,0] = 0

    # Detecting Mitochondria Channel
    red = img2[:,:,1]
    rd_sm = np.sum(red)

    mito_channel = img2[:,:,2]
    gr_sm = np.sum(mito_channel)
    ch = 2

    if rd_sm > gr_sm:
        ch = 1
        mito_channel = red.copy()

    img2 = cv2.normalize(img2, dst=None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    tmp = np.reshape(img2, (1,-1))
    h,b = np.histogram(tmp,bins=128)
    b_diff = np.round(b[1]).astype('uint8')
    img2 = img2 - b_diff

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img2)

    img2[img2>=0.9*np.max(img2)] = 0

    #Removing noise resulted from filters
    img2 = img2 * 1 - 1 * np.min(img2)#img2[:,:,0] * 2 - 2 * np.min(img2[:,:,0])

    return ch,img2





# # Garbage Collector and QC control for Cell Catcher export.
# def single_cell_QC(abspath, hide=True):

#     # Creating Junkyard
#     try:
#         os.makedirs(os.path.join(abspath,'to_discard'))
#     except:
#         pass

#     files_to_check = os.listdir(os.path.join(abspath, 'to_analyze'))

#     for fullpath_input in files_to_check:
#         print(os.path.join(abspath, 'to_analyze',fullpath_input))
#         try:

#             # Read image
#             image = cv2.imread(os.path.join(abspath, 'to_analyze',fullpath_input))

#             # Detecting Mitochondria Channel
#             red = image[:,:,1]
#             rd_sm = np.sum(red)

#             mito_channel = image[:,:,2]
#             gr_sm = np.sum(mito_channel)

#             if rd_sm > gr_sm:
#                 mito_channel = red.copy()

#             # Identifying aliens
#             _,t2t = cv2.threshold(mito_channel, 0.01*np.max(mito_channel), 255, cv2.THRESH_BINARY)

#             mitoLabels = measure.label(t2t, connectivity=2)
#             propMito = regionprops(mitoLabels)
#             good_mask = np.zeros(t2t.shape, dtype="uint8")

#             #Size based filtering
#             area_list = []
#             for r in range(len(propMito)):
#                 area_list.append(propMito[r].area)

#             for r in range(len(propMito)):
#                 if (propMito[r].area > 0.005 * np.max(area_list)) and (propMito[r].area < np.max(area_list)):
#                     if propMito[r].label == 0:
#                         continue
#                     else:
#                         globalMask = np.zeros(t2t.shape, dtype="uint8")
#                         globalMask[mitoLabels == propMito[r].label] = 255
#                         good_mask = cv2.add(good_mask,globalMask)

#             good_mask = cv2.medianBlur(good_mask, 3)
#             good_mask = 255 - good_mask

#             image[:,:,1] = cv2.bitwise_and(image[:,:,1],good_mask)
#             image[:,:,2] = cv2.bitwise_and(image[:,:,2],good_mask)

#             #Identifying and discarding cells that are touching image edges or missing mitochondria

#             qc_mask = np.zeros(t2t.shape, dtype="uint8")
#             qc_mask[4:1020,4:1020] = 255
#             qc_mask = 255 - qc_mask

#             edge_sum = np.sum(cv2.bitwise_and(image[:,:,1],qc_mask)) + np.sum(cv2.bitwise_and(image[:,:,2],qc_mask))
#             mito_sum = np.sum(image[:,:,1]) + np.sum(image[:,:,2])

#             if (mito_sum < 10000) or (edge_sum) > 8:
#                 shutil.move(fullpath_input, os.path.join(abspath,'to_discard', os.path.split(fullpath_input)[1]))
#                 print(f'Did not pass QC: {fullpath_input}')
#                 print(f'{fullpath_input} discarded. (check "to_discard" direcoty)')

#             else:
#                 print(f'QC Passed: {fullpath_input}')
#                 cv2.imwrite(os.path.join(abspath,'to_analyze', os.path.split(fullpath_input)[1]) , image ) #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 if hide == False:
#                     plt.figure(figsize=(10,10))
#                     plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#                     plt.title(f'{fullpath_input}')
#         except:
#             # print(f'{fullpath_input} is not a valid image.')
#             pass




def eight_neighbors(x, y, image):
        VIII_neighbors = []
        VIII_neighbors = [image[x, y - 1], image[x - 1, y - 1], image[x - 1, y], image[x - 1, y + 1],
                          image[x, y + 1], image[x + 1, y + 1], image[x + 1, y], image[x + 1, y - 1]]
        return VIII_neighbors
        # i_width, i_height = image.shape[0], image.shape[1]

def getSkeletonIntersection(skeleton):
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0]];
    image = skeleton.copy();
    image = image / 255;
    intersections = list();
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):

            neighbours = []
            if image[x][y] == 1:

                neighbours = eight_neighbors(x, y, image);
                valid = True;
                if neighbours in validIntersection:
                    intersections.append((y, x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections;

# Single cell quality control
def good_check(address):

    try:

        list_of_cells = os.listdir(address + '/cells/processed/to_analyze/')

        for file in list_of_cells:
            nuc_area = []
            bad_cell = False
            if not(file.endswith('_mask.tif')):

                img = cv2.imread(address+'/cells/processed/to_analyze/'+file)
                img2 = cv2.imread(address+'/cells/processed/to_analyze/'+file[:file.rfind('.tif')]+'_mask.tif')
                img = img[:,:,0]
                #_,im_bw = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY )

                (thresh, im_bw) = cv2.threshold(img, np.percentile(img, 50), 255, cv2.THRESH_BINARY)  # to binary image

                im_bw = cv2.medianBlur(im_bw,5)

                kernel1c0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                im_bw0 = cv2.dilate(im_bw, kernel1c0, iterations=1)

                kernel1op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))  # kernel1cl = np.ones((12,12),np.uint8)
                opennuc1 = cv2.morphologyEx(im_bw0, cv2.MORPH_OPEN, kernel1op)

                kernel1cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # kernel1cl = np.ones((12,12),np.uint8)
                close1 = cv2.morphologyEx(opennuc1, cv2.MORPH_CLOSE, kernel1cl)

                kernel1op = np.ones((5, 5), np.uint8)  # open operation
                opening1 = cv2.morphologyEx(close1, cv2.MORPH_OPEN, kernel1op)

                kernel1op = np.ones((5, 5), np.uint8)  # open operation
                closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel1op)

                imgfill, contours, hierarchy = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    cv2.drawContours(imgfill, [cnt], 0, 255, -1)

                opening = deepcopy(imgfill)  # find and draw contours
                labels = measure.label(opening, connectivity=2)
                Nuclei_Area_Data = []
                propsnuc = regionprops(labels)
                for r in range(len(propsnuc)):
                    Nuclei_Area_Data.append(propsnuc[r].area)
                #print('Nuclei_Area_Data:',Nuclei_Area_Data)
                minor_nuclei_area = []
                #print('Max nuc',max(Nuclei_Area_Data))
                for i in range(len(Nuclei_Area_Data)):
                    if Nuclei_Area_Data[i] < 0.1*int(max(Nuclei_Area_Data)): #800
                        minor_nuclei_area.append(i + 1)

                if len(minor_nuclei_area) != 0:
                    minornucarea = (labels == (minor_nuclei_area[0]))
                    for i in range(len(minor_nuclei_area)):
                        minornucarea = minornucarea | (labels == (minor_nuclei_area[i]))

                    openingfill = deepcopy(opening)
                    get_high_vals = (minornucarea != 0)
                    openingfill[get_high_vals] = 0
                else:
                    openingfill = deepcopy(opening)


                close2 = deepcopy(openingfill)
                cv2.morphologyEx(openingfill, cv2.MORPH_CLOSE, kernel1op)
                close2 = cv2.morphologyEx(close2, cv2.MORPH_ERODE, kernel1op, iterations=5)

                close2 = cv2.medianBlur(close2, 3)


                label_mito = measure.label(img2, connectivity=2)
                label_nuc = measure.label(close2, connectivity=2)
                properties_mito = regionprops(label_mito)
                properties_nuc = regionprops(label_nuc)

                for i in range(len(properties_nuc)):
                    nuc_area.append(properties_nuc[i].area)

                nuc_ratios = np.divide(nuc_area,np.max(nuc_area))

                if len(nuc_ratios[nuc_ratios>0.3]) > 1:
                    bad_cell = True
                    print('Multiple cells packed as one cell.')
                if (np.sum(img2) < 0.75 * np.sum(close2)):
                    bad_cell = True
                    print('Sparse mitochondrial network.')
                if bad_cell:
                    print('Moving {} to bad folder.'.format(file))

                    os.rename(address+'/cells/processed/to_analyze/'+file, address+'/cells/processed/bad/'+file)
                    os.rename(address+'/cells/processed/to_analyze/'+file[:file.rfind('.tif')]+'_mask.tif', address+'/cells/processed/bad/'+file[:file.rfind('.tif')]+'_mask.tif')
    except:
        pass


def measurement(address,cell_list,output_filename):
    database = pd.DataFrame([[0]*110], columns = ['cell_name','cell_mito_count', 'cell_total_mito_area','cell_mean_mito_area',
                             'cell_median_mito_area','cell_std_mito_area','cell_mean_mito_eccentricity',
                             'cell_median_mito_eccentricity','cell_std_mito_eccentricity',
                             'cell_mean_mito_equi_diameter','cell_median_mito_equi_diameter',
                             'cell_std_mito_equi_diameter','cell_mean_mito_euler_number',
                             'cell_median_mito_euler_number','cell_std_mito_euler_number','cell_mean_mito_extent',
                             'cell_median_mito_extent','cell_std_mito_extent','cell_mean_mito_major_axis',
                             'cell_median_mito_major_axis','cell_std_mito_major_axis','cell_mean_mito_minor_axis',
                             'cell_median_mito_minor_axis','cell_std_mito_minor_axis','cell_mean_mito_orientation',
                             'cell_median_mito_orientation','cell_std_mito_orientation','cell_mean_mito_perimeter',
                             'cell_median_mito_perimeter','cell_std_mito_perimeter','cell_mean_mito_solidity',
                             'cell_median_mito_solidity','cell_std_mito_solidity','cell_mean_mito_centroid_x',
                             'cell_median_mito_centroid_x','cell_std_mito_centroid_x','cell_mean_mito_centroid_y',
                             'cell_median_mito_centroid_y','cell_std_mito_centroid_y','cell_mean_mito_distance',
                             'cell_median_mito_distance','cell_std_mito_distance','cell_mean_mito_weighted_cent_x',
                             'cell_median_mito_weighted_cent_x','cell_std_mito_weighted_cent_x',
                             'cell_mean_mito_weighted_cent_y','cell_median_mito_weighted_cent_y',
                             'cell_std_mito_weighted_cent_y','cell_mean_mito_weighted_distance',
                             'cell_median_mito_weighted_distance','cell_std_mito_weighted_distance',
                             'cell_mean_mito_form_factor','cell_median_mito_form_factor',
                             'cell_std_mito_form_factor','cell_mean_mito_roundness','cell_median_mito_roundness',
                             'cell_std_mito_roundness','cell_mean_mito_branch_count','cell_median_mito_branch_count',
                             'cell_std_mito_branch_count','cell_mean_mito_mean_branch_length',
                             'cell_median_mito_mean_branch_length','cell_std_mito_mean_branch_length',
                             'cell_mean_mito_total_branch_length','cell_median_mito_total_branch_length',
                             'cell_std_mito_total_branch_length','cell_mean_mito_median_branch_length',
                             'cell_median_mito_median_branch_length','cell_std_mito_median_branch_length',
                             'cell_mean_mito_std_branch_length','cell_median_mito_std_branch_length',
                             'cell_std_mito_std_branch_length','cell_mean_mito_mean_branch_angle',
                             'cell_median_mito_mean_branch_angle','cell_std_mito_mean_branch_angle',
                             'cell_mean_mito_median_branch_angle','cell_median_mito_median_branch_angle',
                             'cell_std_mito_median_branch_angle','cell_mean_mito_std_branch_angle',
                             'cell_median_mito_std_branch_angle','cell_std_mito_std_branch_angle',
                             'cell_mean_mito_total_density','cell_median_mito_total_density',
                             'cell_std_mito_total_density','cell_mean_mito_average_density',
                             'cell_median_mito_average_density','cell_std_mito_average_density',
                             'cell_mean_mito_median_density','cell_median_mito_median_density',
                             'cell_std_mito_median_density','cell_kurtosis_x','cell_weighted_kurtosis_x',
                             'cell_kurtosis_y','cell_weighted_kurtosis_y','cell_kurtosis_squared',
                             'cell_weighted_kurtosis_squared','cell_skewness_x','cell_weighted_skewness_x',
                             'cell_skewness_y','cell_weighted_skewness_y','cell_skewness_squared',
                             'cell_weighted_skewness_squared','cell_network_orientation','cell_network_major_axis',
                             'cell_network_minor_axis', 'cell_network_eccentricity',
                             'cell_network_effective_extent','cell_network_effective_solidity',
                             'cell_network_fractal_dimension','scale'])



    database_raw = pd.DataFrame([[0]*39], columns = ['cell_name','scale','mito_area','mito_centroid','mito_eccentricity',
                             'mito_equi_diameter','mito_euler_number','mito_extent','mito_major_axis',
                             'mito_minor_axis','mito_orientation','mito_perimeter','mito_solidity',
                             'mito_centroid_x','mito_centroid_y','mito_distance','mito_weighted_cent_x',
                             'mito_weighted_cent_y','mito_weighted_distance','mito_form_factor',
                             'mito_roundness','mito_branch_count','mito_total_branch_length',
                             'mito_mean_branch_length','mito_median_branch_length','mito_std_branch_length',
                             'mito_mean_branch_angle','mito_median_branch_angle','mito_std_branch_angle',
                             'mito_total_density','mito_average_density' ,'mito_median_density',
                             'mito_branch_count','mito_distance','mito_weighted_cent_x' ,'mito_weighted_cent_y',
                             'mito_weighted_distance','mito_form_factor' ,'mito_roundness'])

    for file in cell_list:
        if '_mask' in file:

            fullpath_input = file
            abspath = address

            try:
                print(file)
                img = cv2.imread(abspath+'/output/processed/single_cells_binary/'+file)
                img = img[:,:,0]
                print('Now quantifying:', file)
                scale = int(file[file.rfind('_')+1:file.rfind('_mask')])/1024

                #Mitochondria level measurements

                mito_labels = measure.label(np.array(img),connectivity=2)
                mito_props = regionprops(mito_labels)

                mito_area = []
                mito_centroid = []
                mito_eccentricity = []
                mito_equi_diameter = []
                mito_euler_number = []
                mito_extent = []
                mito_major_axis = []
                mito_minor_axis = []
                mito_orientation = []
                mito_perimeter = []
                mito_solidity = []
                mito_centroid_x = []
                mito_centroid_y = []
                mito_distance = []
                mito_weighted_cent_x = []
                mito_weighted_cent_y = []
                mito_weighted_distance = []
                mito_form_factor = []
                mito_roundness = []
                mito_branch_count = []
                mito_total_branch_length = []
                mito_mean_branch_length = []
                mito_median_branch_length = []
                mito_std_branch_length = []
                mito_mean_branch_angle = []
                mito_median_branch_angle = []
                mito_std_branch_angle = []
                mito_total_density = []
                mito_average_density = []
                mito_median_density = []
                mito_branch_count = []

                for r in range(len(mito_props)):
                    if mito_props[r].area > 16:

                        mito_area.append(mito_props[r].area)
                        mito_eccentricity.append(mito_props[r].eccentricity)
                        mito_equi_diameter.append(mito_props[r].equivalent_diameter)
                        mito_euler_number.append(mito_props[r].euler_number)
                        mito_extent.append(mito_props[r].extent)
                        mito_major_axis.append(mito_props[r].major_axis_length)
                        mito_minor_axis.append(mito_props[r].minor_axis_length)
                        mito_orientation.append(mito_props[r].orientation)
                        mito_perimeter.append(mito_props[r].perimeter)
                        mito_solidity.append(mito_props[r].solidity)
                        mito_centroid.append(mito_props[r].centroid)
                        mito_centroid_x.append(mito_props[r].centroid[0])
                        mito_centroid_y.append(mito_props[r].centroid[1])

                        if mito_props[r].label == 0:
                                continue

                        else:
                            labelMask = np.zeros(img.shape, dtype="uint8")
                            labelMask[mito_labels == mito_props[r].label] = 255
                            #print('lebel1', mito_props[r].label)

                            BranchPointsPositions = []
                            branch_points_ctr = []
                            num_branch_points = 0
                            number_branches = 0
                            branch_length = []
                            branch_angle = []

                            try:
                                #Finding the Skeleton
                                imagebw8 = labelMask
                                imagebw8 = imagebw8.astype(np.int32)

                                Skel2 = thin(labelMask)
                                Skel2 = 255 * Skel2

                                branch_pointsn = getSkeletonIntersection(Skel2)
                                number_branchpoints = len(branch_pointsn)

                                #Recreating Branch Points
                                bp = np.zeros(shape=(imagebw8.shape[0], imagebw8.shape[1]), dtype=np.uint8)

                                for ii in range(len(branch_pointsn)):
                                    xi = branch_pointsn[ii][0]
                                    yi = branch_pointsn[ii][1]
                                    # print(xi,yi)
                                    BranchPointsPositions.append([yi, xi])
                                    bp[yi, xi] = 255

                                kernelbp = np.ones((3, 3), np.uint8)
                                IM = cv2.dilate(bp, kernelbp, iterations=1)


                                #Finding independent branches
                                BranchLengthMatrix = Skel2 - IM
                                BranchMatrix = BranchLengthMatrix > 0

                                imagebwlabels2 = measure.label(np.array(BranchMatrix),connectivity = 2)
                                NUMimagebw1 = imagebwlabels2.max()

                                propsbmm = regionprops(imagebwlabels2)

                                dist_transform2 = cv2.distanceTransform(labelMask,cv2.DIST_L2,5)
                                #print('label2',mito_props[r].label)
                                for rq in range(len(propsbmm)):
                                    branch_length.append((propsbmm[rq].area)+4)
                                    #print('label3',mito_props[r].label)
                                    branch_angle.append(propsbmm[rq].orientation)
                                    #print('orientation ',propsbmm[rq].orientation)
                                    #print('eanch angle ', branch_angle)

                                if len(branch_length) == 1:
                                    #print('initial bl', branch_length)
                                    branch_length[0] =  mito_props[r].major_axis_length
                                    #print('new bl',branch_length)

                                branch_angle = np.multiply(branch_angle,(180/np.pi))
                                num_branch_points = number_branchpoints
                                number_branches = NUMimagebw1
                                total_branch_length = np.sum(branch_length)
                                mean_branch_length = np.mean(branch_length)
                                median_branch_length = np.median(branch_length)
                                std_branch_length = np.std(branch_length)
                                mean_branch_angle = np.mean(branch_angle)
                                median_branch_angle = np.median(branch_angle)
                                std_branch_angle = np.std(branch_angle)

                            except:
                                branch_points_ctr.append([])
                                num_branch_points = 0
                                number_branches = 0
                                branch_length.append([])
                                branch_angle.append([])

                                total_branch_length = 0
                                mean_branch_length = 0
                                median_branch_length = 0
                                std_branch_length = 0
                                mean_branch_angle = 0
                                median_branch_angle = 0
                                std_branch_angle = 0

                    if mito_props[r].area > 16:
                        mito_branch_count.append(number_branches)

                        mito_total_branch_length.append(total_branch_length)
                        mito_mean_branch_length.append(mean_branch_length)
                        mito_median_branch_length.append(median_branch_length)
                        mito_std_branch_length.append(std_branch_length)
                        mito_mean_branch_angle.append(mean_branch_angle)
                        mito_median_branch_angle.append(median_branch_angle)
                        mito_std_branch_angle.append(std_branch_angle)

                        mito_total_density.append(np.sum(dist_transform2[dist_transform2>0]))
                        mito_average_density.append(np.mean(dist_transform2[dist_transform2>0]))
                        mito_median_density.append(np.median(dist_transform2[dist_transform2>0]))

                mito_area = np.multiply(np.power(scale,2),mito_area)
                mito_equi_diameter = np.multiply(scale,mito_equi_diameter)
                mito_major_axis = np.multiply(scale,mito_major_axis)
                mito_minor_axis = np.multiply(scale,mito_minor_axis)
                mito_perimeter = np.multiply(scale,mito_perimeter)
                mito_centroid_x = np.multiply(scale,mito_centroid_x)
                mito_centroid_y = np.multiply(scale,mito_centroid_y)

                mito_distance = np.sqrt(np.power(mito_centroid_x,2)+np.power(mito_centroid_y,2))
                mito_weighted_cent_x = np.divide(np.multiply(mito_centroid_x,mito_area),np.sum(mito_area))
                mito_weighted_cent_y = np.divide(np.multiply(mito_centroid_y,mito_area),np.sum(mito_area))
                mito_weighted_distance = np.sqrt(np.power(mito_weighted_cent_x,2)+np.power(mito_weighted_cent_y,2))
                mito_form_factor = (np.divide(np.power(mito_perimeter,2),mito_area))/(4*np.pi)
                mito_roundness = ((4/np.pi)*np.divide(mito_area, np.power(mito_major_axis,2)))

                mito_total_branch_length = np.multiply(scale,mito_total_branch_length)
                mito_mean_branch_length = np.multiply(scale,mito_mean_branch_length)
                mito_median_branch_length = np.multiply(scale,mito_median_branch_length)
                mito_std_branch_length = np.multiply(scale,mito_std_branch_length)
                mito_total_density = np.multiply(scale,mito_total_density)
                mito_average_density = np.multiply(scale,mito_average_density)
                mito_median_density = np.multiply(scale,mito_median_density)

                #Cell level measurements
                cell_mito_count = len(mito_area)
                cell_total_mito_area = np.sum(mito_area)
                cell_mean_mito_area = np.mean(mito_area)
                cell_median_mito_area = np.median(mito_area)
                cell_std_mito_area = np.std(mito_area)
                cell_mean_mito_eccentricity = np.mean(mito_eccentricity)
                cell_median_mito_eccentricity = np.median(mito_eccentricity)
                cell_std_mito_eccentricity = np.std(mito_eccentricity)
                cell_mean_mito_equi_diameter = np.mean(mito_equi_diameter)
                cell_median_mito_equi_diameter = np.median(mito_equi_diameter)
                cell_std_mito_equi_diameter = np.std(mito_equi_diameter)
                cell_mean_mito_euler_number = np.mean(mito_euler_number)
                cell_median_mito_euler_number = np.median(mito_euler_number)
                cell_std_mito_euler_number = np.std(mito_euler_number)
                cell_mean_mito_extent = np.mean(mito_extent)
                cell_median_mito_extent = np.median(mito_extent)
                cell_std_mito_extent = np.std(mito_extent)
                cell_mean_mito_major_axis = np.mean(mito_major_axis)
                cell_median_mito_major_axis = np.median(mito_major_axis)
                cell_std_mito_major_axis = np.std(mito_major_axis)
                cell_mean_mito_minor_axis = np.mean(mito_minor_axis)
                cell_median_mito_minor_axis = np.median(mito_minor_axis)
                cell_std_mito_minor_axis = np.std(mito_minor_axis)
                cell_mean_mito_orientation = np.mean(mito_orientation)
                cell_median_mito_orientation = np.median(mito_orientation)
                cell_std_mito_orientation = np.std(mito_orientation)
                cell_mean_mito_perimeter = np.mean(mito_perimeter)
                cell_median_mito_perimeter = np.median(mito_perimeter)
                cell_std_mito_perimeter = np.std(mito_perimeter)
                cell_mean_mito_solidity = np.mean(mito_solidity)
                cell_median_mito_solidity = np.median(mito_solidity)
                cell_std_mito_solidity = np.std(mito_solidity)
                cell_mean_mito_centroid_x = np.mean(mito_centroid_x)
                cell_median_mito_centroid_x = np.median(mito_centroid_x)
                cell_std_mito_centroid_x = np.std(mito_centroid_x)
                cell_mean_mito_centroid_y = np.mean(mito_centroid_y)
                cell_median_mito_centroid_y = np.median(mito_centroid_y)
                cell_std_mito_centroid_y = np.std(mito_centroid_y)
                cell_mean_mito_distance = np.mean(mito_distance)
                cell_median_mito_distance = np.median(mito_distance)
                cell_std_mito_distance = np.std(mito_distance)
                cell_mean_mito_weighted_cent_x = np.mean(mito_weighted_cent_x)
                cell_median_mito_weighted_cent_x = np.median(mito_weighted_cent_x)
                cell_std_mito_weighted_cent_x = np.std(mito_weighted_cent_x)
                cell_mean_mito_weighted_cent_y = np.mean(mito_weighted_cent_y)
                cell_median_mito_weighted_cent_y = np.median(mito_weighted_cent_y)
                cell_std_mito_weighted_cent_y = np.std(mito_weighted_cent_y)
                cell_mean_mito_weighted_distance = np.mean(mito_weighted_distance)
                cell_median_mito_weighted_distance = np.median(mito_weighted_distance)
                cell_std_mito_weighted_distance = np.std(mito_weighted_distance)
                cell_mean_mito_form_factor = np.mean(mito_form_factor)
                cell_median_mito_form_factor = np.median(mito_form_factor)
                cell_std_mito_form_factor = np.std(mito_form_factor)
                cell_mean_mito_roundness = np.mean(mito_roundness)
                cell_median_mito_roundness = np.median(mito_roundness)
                cell_std_mito_roundness = np.std(mito_roundness)
                cell_mean_mito_branch_count = np.mean(mito_branch_count)
                cell_median_mito_branch_count = np.median(mito_branch_count)
                cell_std_mito_branch_count = np.std(mito_branch_count)
                cell_mean_mito_mean_branch_length = np.mean(mito_mean_branch_length)
                cell_median_mito_mean_branch_length = np.median(mito_mean_branch_length)
                cell_std_mito_mean_branch_length = np.std(mito_mean_branch_length)
                cell_mean_mito_total_branch_length = np.mean(mito_total_branch_length)
                cell_median_mito_total_branch_length = np.median(mito_total_branch_length)
                cell_std_mito_total_branch_length = np.std(mito_total_branch_length)
                cell_mean_mito_median_branch_length = np.mean(mito_median_branch_length)
                cell_median_mito_median_branch_length = np.median(mito_median_branch_length)
                cell_std_mito_median_branch_length = np.std(mito_median_branch_length)
                cell_mean_mito_std_branch_length = np.mean(mito_std_branch_length)
                cell_median_mito_std_branch_length = np.median(mito_std_branch_length)
                cell_std_mito_std_branch_length = np.std(mito_std_branch_length)
                cell_mean_mito_mean_branch_angle = np.mean(mito_mean_branch_angle)
                cell_median_mito_mean_branch_angle = np.median(mito_mean_branch_angle)
                cell_std_mito_mean_branch_angle = np.std(mito_mean_branch_angle)
                cell_mean_mito_median_branch_angle = np.mean(mito_median_branch_angle)
                cell_median_mito_median_branch_angle = np.median(mito_median_branch_angle)
                cell_std_mito_median_branch_angle = np.std(mito_median_branch_angle)
                cell_mean_mito_std_branch_angle = np.mean(mito_std_branch_angle)
                cell_median_mito_std_branch_angle = np.median(mito_std_branch_angle)
                cell_std_mito_std_branch_angle = np.std(mito_std_branch_angle)
                cell_mean_mito_total_density = np.mean(mito_total_density)
                cell_median_mito_total_density = np.median(mito_total_density)
                cell_std_mito_total_density = np.std(mito_total_density)
                cell_mean_mito_average_density = np.mean(mito_average_density)
                cell_median_mito_average_density = np.median(mito_average_density)
                cell_std_mito_average_density = np.std(mito_average_density)
                cell_mean_mito_median_density = np.mean(mito_median_density)
                cell_median_mito_median_density = np.median(mito_median_density)
                cell_std_mito_median_density = np.std(mito_median_density)
                cell_kurtosis_x = kurtosis(mito_centroid_x)
                cell_weighted_kurtosis_x = kurtosis(mito_weighted_cent_x)
                cell_kurtosis_y = kurtosis(mito_centroid_y)
                cell_weighted_kurtosis_y = kurtosis(mito_weighted_cent_y)
                cell_kurtosis_squared = np.add(np.power(cell_kurtosis_x,2),np.power(cell_kurtosis_y,2))
                cell_weighted_kurtosis_squared = np.add(np.power(cell_weighted_kurtosis_x,2),np.power(cell_weighted_kurtosis_y,2))
                cell_skewness_x = skew(mito_centroid_x)
                cell_weighted_skewness_x = skew(mito_weighted_cent_x)
                cell_skewness_y = skew(mito_centroid_y)
                cell_weighted_skewness_y = skew(mito_weighted_cent_y)
                cell_skewness_squared = np.add(np.power(cell_skewness_x,2),np.power(cell_skewness_y,2))
                cell_weighted_skewness_squared = np.add(np.power(cell_weighted_skewness_x,2),np.power(cell_weighted_skewness_y,2))

                chull = convex_hull_image(img)

                cell_labels = measure.label(np.array(chull),connectivity=2)
                cell_props = regionprops(cell_labels)
                cell_network_orientation = cell_props[0].orientation * 180/np.pi
                cell_network_major_axis = cell_props[0].major_axis_length
                cell_network_minor_axis = cell_props[0].minor_axis_length
                cell_network_eccentricity = cell_props[0].eccentricity

                cell_scaled_area = np.multiply(np.power(scale,2),cell_props[0].area)
                cell_network_effective_extent = (np.sum(mito_area)/cell_scaled_area) * cell_props[0].extent
                cell_network_effective_solidity = np.sum(mito_area)/cell_scaled_area

                cell_network_major_axis = np.multiply(scale,cell_network_major_axis)
                cell_network_minor_axis = np.multiply(scale,cell_network_minor_axis)

                pixels=[]

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i,j]>0:
                            pixels.append((i,j))

                Lx=img.shape[1]
                Ly=img.shape[0]

                pixels=np.array(pixels)

                scales=np.logspace(0.01, 1, num=10, endpoint=False, base=2)
                Ns=[]
                for scale1 in scales:

                    H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale1),np.arange(0,Ly,scale1)))
                    Ns.append(np.sum(H>0))

                coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
                cell_network_fractal_dimension = -coeffs[0]

                temp_dataset = pd.DataFrame([[file, cell_mito_count, cell_total_mito_area, cell_mean_mito_area,
                          cell_median_mito_area, cell_std_mito_area, cell_mean_mito_eccentricity,
                          cell_median_mito_eccentricity, cell_std_mito_eccentricity,
                          cell_mean_mito_equi_diameter, cell_median_mito_equi_diameter,
                          cell_std_mito_equi_diameter, cell_mean_mito_euler_number,
                          cell_median_mito_euler_number, cell_std_mito_euler_number,
                          cell_mean_mito_extent ,cell_median_mito_extent, cell_std_mito_extent,
                          cell_mean_mito_major_axis, cell_median_mito_major_axis, cell_std_mito_major_axis,
                          cell_mean_mito_minor_axis, cell_median_mito_minor_axis,cell_std_mito_minor_axis,
                          cell_mean_mito_orientation, cell_median_mito_orientation, cell_std_mito_orientation,
                          cell_mean_mito_perimeter, cell_median_mito_perimeter,cell_std_mito_perimeter,
                          cell_mean_mito_solidity, cell_median_mito_solidity, cell_std_mito_solidity,
                          cell_mean_mito_centroid_x, cell_median_mito_centroid_x, cell_std_mito_centroid_x,
                          cell_mean_mito_centroid_y, cell_median_mito_centroid_y,cell_std_mito_centroid_y,
                          cell_mean_mito_distance, cell_median_mito_distance, cell_std_mito_distance,
                          cell_mean_mito_weighted_cent_x, cell_median_mito_weighted_cent_x,
                          cell_std_mito_weighted_cent_x, cell_mean_mito_weighted_cent_y,
                          cell_median_mito_weighted_cent_y, cell_std_mito_weighted_cent_y,
                          cell_mean_mito_weighted_distance, cell_median_mito_weighted_distance,
                          cell_std_mito_weighted_distance, cell_mean_mito_form_factor,
                          cell_median_mito_form_factor, cell_std_mito_form_factor,
                          cell_mean_mito_roundness, cell_median_mito_roundness, cell_std_mito_roundness,
                          cell_mean_mito_branch_count, cell_median_mito_branch_count,
                          cell_std_mito_branch_count, cell_mean_mito_mean_branch_length,
                          cell_median_mito_mean_branch_length, cell_std_mito_mean_branch_length,
                          cell_mean_mito_total_branch_length, cell_median_mito_total_branch_length,
                          cell_std_mito_total_branch_length, cell_mean_mito_median_branch_length,
                          cell_median_mito_median_branch_length, cell_std_mito_median_branch_length,
                          cell_mean_mito_std_branch_length, cell_median_mito_std_branch_length,
                          cell_std_mito_std_branch_length, cell_mean_mito_mean_branch_angle,
                          cell_median_mito_mean_branch_angle, cell_std_mito_mean_branch_angle,
                          cell_mean_mito_median_branch_angle, cell_median_mito_median_branch_angle,
                          cell_std_mito_median_branch_angle, cell_mean_mito_std_branch_angle,
                          cell_median_mito_std_branch_angle,cell_std_mito_std_branch_angle,
                          cell_mean_mito_total_density,cell_median_mito_total_density,
                          cell_std_mito_total_density,cell_mean_mito_average_density,
                          cell_median_mito_average_density, cell_std_mito_average_density,
                          cell_mean_mito_median_density, cell_median_mito_median_density,
                          cell_std_mito_median_density, cell_kurtosis_x, cell_weighted_kurtosis_x,
                          cell_kurtosis_y, cell_weighted_kurtosis_y, cell_kurtosis_squared,
                          cell_weighted_kurtosis_squared, cell_skewness_x, cell_weighted_skewness_x,
                          cell_skewness_y, cell_weighted_skewness_y, cell_skewness_squared,
                          cell_weighted_skewness_squared, cell_network_orientation, cell_network_major_axis,
                          cell_network_minor_axis, cell_network_eccentricity, cell_network_effective_extent,
                          cell_network_effective_solidity,cell_network_fractal_dimension, scale]],
                          columns = ['cell_name','cell_mito_count', 'cell_total_mito_area' ,'cell_mean_mito_area',
                         'cell_median_mito_area','cell_std_mito_area','cell_mean_mito_eccentricity',
                         'cell_median_mito_eccentricity','cell_std_mito_eccentricity',
                         'cell_mean_mito_equi_diameter','cell_median_mito_equi_diameter',
                         'cell_std_mito_equi_diameter','cell_mean_mito_euler_number',
                         'cell_median_mito_euler_number','cell_std_mito_euler_number','cell_mean_mito_extent',
                         'cell_median_mito_extent','cell_std_mito_extent','cell_mean_mito_major_axis',
                         'cell_median_mito_major_axis','cell_std_mito_major_axis','cell_mean_mito_minor_axis',
                         'cell_median_mito_minor_axis','cell_std_mito_minor_axis','cell_mean_mito_orientation',
                         'cell_median_mito_orientation','cell_std_mito_orientation','cell_mean_mito_perimeter',
                         'cell_median_mito_perimeter','cell_std_mito_perimeter','cell_mean_mito_solidity',
                         'cell_median_mito_solidity','cell_std_mito_solidity','cell_mean_mito_centroid_x',
                         'cell_median_mito_centroid_x','cell_std_mito_centroid_x','cell_mean_mito_centroid_y',
                         'cell_median_mito_centroid_y','cell_std_mito_centroid_y','cell_mean_mito_distance',
                         'cell_median_mito_distance','cell_std_mito_distance','cell_mean_mito_weighted_cent_x',
                         'cell_median_mito_weighted_cent_x','cell_std_mito_weighted_cent_x',
                         'cell_mean_mito_weighted_cent_y','cell_median_mito_weighted_cent_y',
                         'cell_std_mito_weighted_cent_y','cell_mean_mito_weighted_distance',
                         'cell_median_mito_weighted_distance','cell_std_mito_weighted_distance',
                         'cell_mean_mito_form_factor','cell_median_mito_form_factor',
                         'cell_std_mito_form_factor','cell_mean_mito_roundness','cell_median_mito_roundness',
                         'cell_std_mito_roundness','cell_mean_mito_branch_count','cell_median_mito_branch_count',
                         'cell_std_mito_branch_count','cell_mean_mito_mean_branch_length',
                         'cell_median_mito_mean_branch_length','cell_std_mito_mean_branch_length',
                         'cell_mean_mito_total_branch_length','cell_median_mito_total_branch_length',
                         'cell_std_mito_total_branch_length','cell_mean_mito_median_branch_length',
                         'cell_median_mito_median_branch_length','cell_std_mito_median_branch_length',
                         'cell_mean_mito_std_branch_length','cell_median_mito_std_branch_length',
                         'cell_std_mito_std_branch_length','cell_mean_mito_mean_branch_angle',
                         'cell_median_mito_mean_branch_angle','cell_std_mito_mean_branch_angle',
                         'cell_mean_mito_median_branch_angle','cell_median_mito_median_branch_angle',
                         'cell_std_mito_median_branch_angle','cell_mean_mito_std_branch_angle',
                         'cell_median_mito_std_branch_angle','cell_std_mito_std_branch_angle',
                         'cell_mean_mito_total_density','cell_median_mito_total_density',
                         'cell_std_mito_total_density','cell_mean_mito_average_density',
                         'cell_median_mito_average_density','cell_std_mito_average_density',
                         'cell_mean_mito_median_density','cell_median_mito_median_density',
                         'cell_std_mito_median_density','cell_kurtosis_x','cell_weighted_kurtosis_x',
                         'cell_kurtosis_y','cell_weighted_kurtosis_y','cell_kurtosis_squared',
                         'cell_weighted_kurtosis_squared','cell_skewness_x','cell_weighted_skewness_x',
                         'cell_skewness_y','cell_weighted_skewness_y','cell_skewness_squared',
                         'cell_weighted_skewness_squared','cell_network_orientation','cell_network_major_axis',
                         'cell_network_minor_axis', 'cell_network_eccentricity',
                         'cell_network_effective_extent','cell_network_effective_solidity',
                         'cell_network_fractal_dimension','scale'])

                temp_dataset_raw = pd.DataFrame([[file,scale, mito_area, mito_centroid,
                          mito_eccentricity, mito_equi_diameter, mito_euler_number, mito_extent,
                          mito_major_axis, mito_minor_axis, mito_orientation, mito_perimeter,
                          mito_solidity, mito_centroid_x, mito_centroid_y, mito_distance,
                          mito_weighted_cent_x, mito_weighted_cent_y, mito_weighted_distance,
                          mito_form_factor, mito_roundness, mito_branch_count, mito_total_branch_length,
                          mito_mean_branch_length, mito_median_branch_length, mito_std_branch_length,
                          mito_mean_branch_angle, mito_median_branch_angle,mito_std_branch_angle,
                          mito_total_density, mito_average_density,mito_median_density, mito_branch_count,
                          mito_distance, mito_weighted_cent_x, mito_weighted_cent_y, mito_weighted_distance,
                          mito_form_factor, mito_roundness]],
                          columns = ['cell_name', 'scale','mito_area','mito_centroid','mito_eccentricity',
                         'mito_equi_diameter','mito_euler_number','mito_extent','mito_major_axis',
                         'mito_minor_axis','mito_orientation','mito_perimeter','mito_solidity',
                         'mito_centroid_x','mito_centroid_y','mito_distance','mito_weighted_cent_x',
                         'mito_weighted_cent_y','mito_weighted_distance','mito_form_factor',
                         'mito_roundness','mito_branch_count','mito_total_branch_length',
                         'mito_mean_branch_length','mito_median_branch_length','mito_std_branch_length',
                         'mito_mean_branch_angle','mito_median_branch_angle','mito_std_branch_angle',
                         'mito_total_density','mito_average_density' ,'mito_median_density',
                         'mito_branch_count','mito_distance','mito_weighted_cent_x' ,'mito_weighted_cent_y',
                         'mito_weighted_distance','mito_form_factor' ,'mito_roundness'])


                database = database.append(temp_dataset,ignore_index=True)
                database_raw = database_raw.append(temp_dataset_raw,ignore_index=True)
            except:
                print('Couldn\'t Analize {0}'.format(file))

    detail = str(datetime.datetime.now().year) + '-' + str(datetime.datetime.now().month) + '-' + str(datetime.datetime.now().day) + '-' + output_filename
    database.drop(database.index[0],inplace=True)
    database.to_csv(address+'/'+detail+'.csv', sep=',')

    database_raw.drop(database_raw.index[0],inplace=True)
    database_raw.to_csv(address+'/'+detail+'_raw.tsv', sep='\t')

    print('Done with Measurements!\nYour data is Ready! Enjoy :)')


#Measuring Mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):

    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

#Check if required directory structure is present
def check_dir_requirements_miner(address):
    try:
        os.makedirs(os.path.join(address,'output','to_analyze'))
    except:
        print('Test1: Directory requirements already satisfied')
        pass
#Check if files exist where they should exist
def check_requirements_miner(plan,address):
    counter = 0
    if plan == 'I am using Mito Tracker to segment the cells that I previously isolated by Cell Catcher.':
        try:
            temp_address = os.path.join(address, 'output','to_analyze')
            print(f'Your Plan:\n"{plan}"')
            file_list = os.listdir(os.path.join(address, 'output','to_analyze'))
            for f in file_list:
                try:
                    img = plt.imread(os.path.join(address, 'output','to_analyze',f))
                    if img.shape:
                        counter += 1
                except:
                    pass
            print(f'\nWe found {counter} images based on your plan.')

            if counter == 0:
                print('\n>>> Are you sure that you have already used Cell Catcher to separate cells from your images?')
                print(f'\n>>> There are no files in {os.path.join(address, "output", "to_analyze")}')
                print('\n>>> Consider changing your plan and following the instructions.')
        except:
            print('\n>>> Are you sure that you have already used Cell Catcher to separate cells from your images?')
            print(f'\n>>> There are no files in {os.path.join(address, "output","to_analyze")}')
            print('\n>>> Consider changing your plan and following the instructions.')
            pass
    else:
        try:
            temp_address = os.path.join(address, 'output','to_analyze')
            print(f'Your Plan:\n"{plan}"')
            file_list = os.listdir(os.path.join(address, 'output','to_analyze'))
            for f in file_list:
                try:
                    img = plt.imread(os.path.join(address, 'output','to_analyze',f))
                    if img.shape:
                        counter += 1
                except:
                      pass

            if counter != 0:
                print('\n>>> It seems you already have some cells from your previous analysis using Cell Catcher.')
                print(f"\n>>> Your files are located in: {os.path.join(address, 'output','to_analyze')}")
                print('\n>>> If you want to use new set of images, consider removing files in this folder, and placing your single-cell images there.')
            else:
                print(f"\n>>> Please put your single cell images here: {os.path.join(address, 'output','to_analyze')}")
                print('>>> After putting your files in the above address, run the next cell, to ensure that requirements are satisfied.')

        except:
                print(f"\n>>> Looking for files in {os.path.join(address, 'output','to_analyze')}. The folder does not exist or it is empty.")
                print(f"\n>>> Please put your single-cell images here: {os.path.join(address, 'output','to_analyze')}")
                print('\n>>> After putting your files in the above address, run the next cell, to ensure that requirements are satisfied.')
                pass

    return counter, temp_address

#Confirm the existance of the files based on user's plan
def confirm_requirements_miner(plan, address, counter):
    if plan == 'I have my own set of single cell images.':
        try:
            file_list = []
            file_list = os.listdir(os.path.join(address, 'output','to_analyze'))
            if counter == 0:
                for f in file_list:
                    try:
                        img = plt.imread(os.path.join(address, 'output','to_analyze',f))
                        if img.shape:
                            counter += 1
                    except:
                        pass
                print('\nIt seems we are still missing the files that you want to analyze.')
                print(f"\nMake sure your desired files are located here: {os.path.join(address, 'output','to_analyze')}")
                print('\nOnce you put your files in the above location, run this cell again to ensure you are all set!')

            else:
                print(f'\nWe found {counter} images based on your plan. It seems your are all set!')
                print('\nYou are all set!')
        except:
            print('\nIt seems we are still missing the files that you want to analyze.')
            print(f"\nMake sure your desired files are located here: {os.path.join(address, 'output','to_analyze')}")
            print('\nOnce you put your files in the above location, run this cell again to ensure you are all set!')

    if plan == 'I am using Mito Tracker to segment the cells that I previously isolated by Cell Catcher.':
        try:
            file_list = []
            file_list = os.listdir(os.path.join(address, 'output','to_analyze'))
            if counter == 0:
                for f in file_list:
                    try:
                        img = plt.imread(os.path.join(address, 'output','to_analyze',f))
                        if img.shape:
                            counter += 1
                    except:
                        pass
                print('\nIt seems we are still missing the files that you want to analyze.')
                print(f"\nMake sure your desired files are located here: {os.path.join(address, 'output','to_analyze')}")
                print('\nOnce you put your files in the above location, run this cell again to ensure you are all set!')

            else:
                print(f'\nWe found {counter} images based on your plan. It seems your are all set!')
                print('\nYou are all set!')
        except:
            print('\nIt seems we are still missing the files that you want to analyze.')
            print(f"\nMake sure your desired files are located here: {os.path.join(address, 'output','to_analyze')}")
            print('\nOnce you put your files in the above location, run this cell again to ensure you are all set!')

catcher_initial_params = {
'Intensity_threshold' : [75],
'Size_threshold': [1000],
'correct_cells' : [True],
'neighorhood' : [50],
'mito_threshold' : [75],
'dialation' : 50,
'empty_cell_thresh' : [0],
'nuc_correction': [True],
'showimg' : True,
'diffused_bg': [False],
'correct_cells' : [True],
'neighorhood' : [45],
'mito_threshold' : [65],
'mito_low' : [False],
'sparse' : [True],
}

mito_miner_initial_params = {
'thresh_moon1' : [5],
'thresh_moon2' : [3],
'thresh_adamask1' : [3],
'thresh_adamask2' : [0.5],
'thresh_adamask3' : [0],
'thresh_median_mask' : [2],
'thresh_median_mask2' : [1],
'thresh_median_signal' : [1],
'out' : [],
'adaptive_th' : [True],
'window_size_th' : [16],
'steepness_th' : [0.5],
'equalizer_th' : [0.15],
'remove_debries_th' : [True],
}

def filter_plot_mito_miner(pic,binarized,diffused):
    plt.figure(figsize=(20,20))
    plt.subplot(121)
    if diffused:
        plt.title('3X Brighter Original Image',fontsize=16)
        plt.imshow(pic*3,'gray')
    else:
        plt.title('Original Image',fontsize=16)
        plt.imshow(pic,'gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('Mitochondrial Network Mask',fontsize=16)
    plt.axis('off')
    plt.imshow(binarized,'gray')
    plt.show()

def check_files_before_filter(temp_address):
    sample_files = []
    try:
        for file in os.listdir(temp_address):
            sample_files.append(file)
    except:
        print('\nThere is nothing to analyze!')
        print('\nYou must at least segment one cell before you proceed!')
        print('\nPlease go to the last step and select "Segment Cell" for at least one cell.\n')
    return sample_files

# Final parameter report for Mito Miner
def report_mito_miner(filt,params):
    print('The following settings will be used to analyze all the images!')
    print('**************************************************************')
    print('\nFilter for binarizing the cells: ',filt)

    if filt == 'Not One of Us':
        try:
            print('\nHarshness for "Not One of Us" algorithm: ',params['thresh_median_mask'][-1])
            print('\nStrength for "Not One of Us" algorithm: ',params['thresh_median_mask2'][-1])
            if params['adaptive_th'][-1]:
                print('\nAdaptive Thgresholding Selected')
                print('\nTile Size: ',params['window_size_th'][-1])
                print('\nAdaptive Function Power: ',params['equalizer_th'][-1])
                print('\nAdaptive Function Footprint: ',params['steepness_th'][-1])
                print('\nRemove Leftover debries: ',params['remove_debries_th'][-1])
            else:
                print('Adaptive threshold was not selected')

        except:
            print('\n### ERROR: It seems you have not set your treshold values!\nIf you continue, the analysis will fail!\nGo back and run the previous cells.')
            pass

    elif filt == 'Let It Go':
        try:
            print('\nHarshness for "Let It Go" algorithm: ',params['thresh_median_signal'][-1])
            if params['adaptive_th'][-1]:
                print(1)
                print('\nAdaptive Thgresholding Selected')
                print('\nTile Size: ',params['window_size_th'][-1])
                print('\nAdaptive Function Power: ',params['equalizer_th'][-1])
                print('\nAdaptive Function Footprint: ',params['steepness_th'][-1])
                print('\nRemove Leftover debries: ',params['remove_debries_th'][-1])
            else:
                print('Adaptive threshold was not selected')

        except:
            print('\n### ERROR: It seems you have not set your treshold values!\nIf you continue, the analysis will fail!\nGo back and run the previous cells.')
            pass


# A function to simulate the circle in Nuc Adder
def simulate_circle(address, file, x,y,radius,filled):
    if filled:
        th = -1
    else:
        th = 2

    img = plt.imread(os.path.join(address,file))
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    cv2.circle(img, (x,y), radius, color=(0,0,255), thickness=th, lineType=8, shift=0)
    plt.imshow(img)
    return radius

# Set the number of cells in the image
def how_many_cells():
    print('\nTo how many cells do you want to add a neucleus in this image?')

    while True:
        how_many = input('Please enter a valid integer: ')

        try:
            how_many = int(how_many)
            if how_many <= 1000:
                return how_many
                break
            else:
                pass

        except:
            print("\nLet's forget about biology for a second and focus on math!")
            print(f'\n{how_many} is not an integer!')


def nuc_adder_make_folders(address):
    try:
        os.makedirs(f'{address}/transformed/')
    except:
        pass

def manual_seg_make_folders(address):
    try:
        os.makedirs(f'{address}/manually_segmented/')
    except:
        pass



# A function to chooce pen thickness for manual segmentation
def pen_size_select(address, file, x,y,tilt,radius):

    img = plt.imread(os.path.join(address,file))
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    cv2.line(img,(x+90,y+90),(x+180,y+2*(tilt+45)),(255,255,0),radius)
    plt.imshow(img)
    return radius
