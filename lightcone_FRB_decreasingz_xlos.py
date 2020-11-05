#!/usr/bin/env python
# coding: utf-8

#This code takes in an arbitrary set of boxes, whose directory and redshift ranges you must provide, and makes a lightcone
import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import astropy
from astropy.cosmology import Planck15 as p15
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo


def lightcone(**kwargs ):

    #set defaults:
    mode = 'xH'
    marker = 'xH_'
    N = 500
    DIM = 200
    Box_length = 300
    z_start = 10
    z_end = 6
    nboxes = 21
    directory = '/Users/michael/Documents/MSI-C/21cmFAST/Boxes/z6-10/OriginalTemperatureBoxesNoGaussianities/'
    halo_location_x = 0
    halo_location_y = 0
    slice = DIM - 1

    #sort arguments
    if 'marker' in kwargs:
        marker = kwargs.get('marker')
    if 'DIM' in kwargs:
        DIM = kwargs.get('DIM')
    if 'z_range_of_boxes' in kwargs:
        z_range_of_boxes  = kwargs.get('z_range_of_boxes')
        print(z_range_of_boxes)
        nboxes = len(z_range_of_boxes)
        z_start = np.max(z_range_of_boxes)
        z_end = np.min(z_range_of_boxes)
        print(z_start,z_end, nboxes)
    if 'N' in kwargs:
        N = kwargs.get('N')
    else:
        N = 50*nboxes
    if 'Box_length' in kwargs:
        Box_length = kwargs.get('Box_length')
    if 'directory' in kwargs:
        directory = kwargs.get('directory')
    if 'box_slice' in kwargs:
        slice = kwargs.get('box_slice')
    if 'return_redshifts' in kwargs:
        return_redshifts = kwargs.get('return_redshifts')
    else:
        return_redshifts = False
    if 'sharp_cutoff' in kwargs:
        sharp_cutoff = kwargs.get('sharp_cutoff')
    else:
        sharp_cutoff = np.inf

    #21cmFAST boxes have different naming tags, if it is an ionization box the redshifts info will be found
    #at different parts of the filename as compared to a density box
    if 'smoothed' in marker:
        #this is a density box box
        s,e=25,31
    else:
        if 'xH' in marker:
            s,e = 10, 20
        else:
            if 'halos' in marker:
                s , e = 5, 10
            else:
                print('We can not identify what box this is')
                return -1
    
    #the total range of redshifts that this lightcone will span
    z_range_of_boxes = np.linspace(z_start,z_end,nboxes)
    print(z_range_of_boxes)




    ####################################################
    # useful functions
    ####################################################

    def box_maker(name):       #reads in the boxes
        data = np.fromfile(directory + name ,dtype=np.float32)
        box = data.reshape((DIM,DIM,DIM))
        return box

    box_path = np.zeros((len(z_range_of_boxes)), dtype = 'object')
    box_path_redshifts = np.zeros((len(box_path)))
    for fn in os.listdir(directory):   #store the filename and directories of the boxes
        for z in range(len(z_range_of_boxes)):
            if marker in fn and str(np.round(z_range_of_boxes[z],2)) in fn[s:e]:
                box_path[z] = fn
                #this next part searches the filename for the redshift, however we now realize that this doesn't need to be done
                #index = box_path[z].find('_z0')
                #start = index + len('_z0')
                #box_path_redshifts[z] = float(box_path[z][start:start + 4])


    #this function determines which boxes a given redshift lies between
    def find_sandwiched_bins(z_range, z):
        z_floor = np.max(z_range)
        z_ceil = z_range[1]
        binn = 1
        while(z_ceil >= np.min(z_range)):
            if ((z <= z_floor) and (z > z_ceil)):
                return ( z_ceil, z_floor)
            z_floor = z_ceil
            if z_ceil == np.max(z_range):
                print('looking for ' , z_range, z)
                break
            z_ceil = z_range[binn+1]
            binn += 1
            #safety net
            if binn > 1000:
                print('breaking')
                break

    #function which converts a comoving distance to a pixel location within the box
    def comoving2pixel(DIM, Box_length, comoving_distance):
        return int(float(comoving_distance * DIM)/float(Box_length))

    #function which determines whether we have exceeded the maximum allowable redshift for a box
    def didweswitchbox(historyofzminus, z_plus, ctr):
        if z_plus < historyofzminus[ctr - 1 ]:
            return True
        else:
            return False

    ####################################################
    # initialize all relevant arrays
    ####################################################

    lightcone = np.zeros((N, DIM, DIM))
    lightcone_halo = np.zeros((N))
    z_range = np.linspace(z_start,z_end,N)
    zs = []
    z = z_range[0]
    ctr = 0
    comoving_distance_z0_zstart = cosmo.comoving_distance(z_range[0]).value
    prev_pix_loc = 0
    pixel_addition = 0
    pixel_origin = 0
    pixel_location_relative_to_origin = 0
    historyofzminus = []

    ####################################################
    # loop through redshifts
    ####################################################

    box_path_redshifts = z_range_of_boxes

    #scroll through all the redshifts and pick out the slice of the box that corresponds to that z
    while(z > np.min(z_range)):
        
        #this redshift is sandwiched between the following z
        z_sandwhich = find_sandwiched_bins(box_path_redshifts, z)
        z_minus = z_sandwhich[0]
        z_plus = z_sandwhich[1]
        
        historyofzminus.append(z_plus)
        
        #these are the boxes that z is sandwiched between
        xH_minus = box_maker(box_path[list(box_path_redshifts).index(z_minus)])
        xH_plus = box_maker(box_path[list(box_path_redshifts).index(z_plus)])
        
        #convert that redshift to a comoving distance
        comoving_distance_z = cosmo.comoving_distance(z).value
        comoving_distance_z0_to_z = comoving_distance_z0_zstart - comoving_distance_z
        comoving_distance_from_last_switch = cosmo.comoving_distance(z_plus).value
        
        
        if ctr == 0:
            pixel_addition = comoving2pixel(DIM,Box_length, comoving_distance_z0_to_z)
            prev_pix_loc = -pixel_addition + slice
            pixel_origin = slice
            #save this redshift
            zs.append(z)
            lightcone[ctr,:,:] =  (xH_plus[:,:,slice] - xH_minus[:,:,slice])*((z - z_minus)/(z_plus - z_minus)) + xH_minus[:,:,slice]
            #increment counter and redshift
            ctr += 1
            z = z_range[ctr]
            #skip to the next step
            continue

        else:
            if didweswitchbox(historyofzminus, z_plus, ctr):
                pixel_origin = prev_pix_loc
            pixel_location_relative_to_origin = -comoving2pixel(DIM,Box_length, comoving_distance_from_last_switch - comoving_distance_z)
            pixel_addition = (pixel_location_relative_to_origin + pixel_origin)%DIM
            prev_pix_loc = pixel_addition

        #save this redshift
        zs.append(z)
        #save the box information for this particular lightcone slice
        lightcone[ctr,:,:] =  (xH_plus[pixel_addition,:,:] - xH_minus[pixel_addition,:,:])*((z - z_minus)/(z_plus - z_minus)) + xH_minus[pixel_addition,:,:]


        ctr += 1
        z = z_range[ctr]
        #pl.savefig(str(ctr)+'.png')
        #safety net
        if ctr > N:
            break
        #does the user want us to stop the z scroll after a particular value?
        if ctr >= sharp_cutoff:
            if return_redshifts:
                return lightcone[0:sharp_cutoff,:,] , np.array(zs[0:])
            else:
                return lightcone[0:sharp_cutoff,:,]

    #return the lightcone history as the redshift log (should the user specify that)
    if return_redshifts:
        return lightcone[0:int(N-1),:,] , np.array(zs)
    else:
        return lightcone[0:int(N-1),:,]


#lightconepng = lightcone(N = 500 )
#directory = '/Users/michael/Research/LAE_Clustering/Boxes_w_HaloFinder/'

#pl.imshow(np.swapaxes(lightconepng,0,2)[100])
#pl.savefig('Lightcone.png')
#pl.ylabel('Box slice at x = 0')
#pl.xlabel('Redshift')
#pl.show()
#pl.close()


