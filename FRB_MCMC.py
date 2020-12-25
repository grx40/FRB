import numpy as np
import sys
from subprocess import *
import os
import emcee
from emcee.utils import MPIPool
import astropy
import xi_2D
import time
import lightcone_FRB_decreasingz_xlos as lc
import DM
import h5py
import matplotlib.pyplot as pl
import misc_functions as misc
import mpi4py
import lightcone_FRB_decreasingz_xlos_forHaloFinder as lcH
from scipy.stats import skewnorm
import misc_functions as misc
from scipy import stats
import lightcone_FRB_decreasingz_xlos_forHaloFinder as lcH

#let's name this run 
RUN= int(np.abs(np.random.uniform(0, 2000)))
os.system("echo RUN IS : " + str(RUN))



def load_data(path, HII_DIM):
    data = np.fromfile(path ,dtype=np.float32)
    return data.reshape((HII_DIM,HII_DIM,HII_DIM))
    
def beta2sigma(beta):
    if beta >= 0:
        sign = np.sign(-np.pi*(beta) + np.pi)
        sigma = np.abs(-np.pi*(beta) + np.pi)
        return (sign*sigma)
    else:
        sign = np.sign(-np.pi*(beta) - np.pi)
        sigma = np.abs(-np.pi*(beta) - np.pi)
        return (sign*sigma)

####################################################################
#          Cosmology and Astrophysical Parameters                 #
####################################################################

#EoR parameters (fixed in this version)
zeta = 500
Mturn = 10
Rmfp = 30
#Cosmology constants
nHI = 10
H0 = float(68)/float(3.086e19)
OMm = 0.25
OMl = 0.75
baryon2DMfrac = 0.05
#constants
pc = 3.08*10**16 # pc in terms of m
cm2m = 0.01 #cm to m conversion



####################################################################
#                    Emcee specific parameters                    #
####################################################################

#os.chdir("/home/grx40/projects/def-acliu/grx40/soft/21cmFASTM/Programs/")
#dimensions and walkers of EnsembleSampler
ndim = 1
nwalkers = 2


####################################################################
#                     FRB script loader                #
####################################################################
#Constants for the script

Box_length = 300
HII_DIM = 200
DIM = 800

####################################################################
#                     Lightcone stuff               #
####################################################################
halo_directory = '../Boxes/Default_Res/'
lightcone_sharpcutoff = False
z_end = 0.0
z_start = 10.0
delta_z = 0.5
box_slice = 199

nboxes = int(np.round(float(z_start - z_end)/float(delta_z),1)) + 1
z_range_of_halo_boxes = np.linspace(z_start, z_end, nboxes)
#confirm that the z_range is correct (debug)
os.system("echo " + str(z_range_of_halo_boxes))
#directory to get the base density boxes AFTER reionization
density_boxes ='../Boxes/Fiducial_1_5000_30_5e8_allZ/'

#make the base density lightcone once (use it for all subsequent times
densitylightcone = lc.lightcone(DIM = HII_DIM, z_range_of_boxes = z_range_of_halo_boxes, box_slice = int(box_slice), N = 500, directory = density_boxes, marker = 'updated_smoothed_deltax')

#make the halolightcone
#what is the redshift range spanned by the halo files?
z_range_of_halo_boxes = np.linspace(10, 0.0, np.round(float(10)/float(0.5),2) + 1)
print(z_range_of_halo_boxes)

#load halos (i.e. FRBs)
halo_directory = '../Boxes/Halos/'

#load all the halopos for all the redshifts and store them into a single array
Halopos_z = np.zeros((len(z_range_of_halo_boxes)), dtype = object)
for z in range(Halopos_z.shape[0]):
    if z_range_of_halo_boxes[z] < 6.0:
        print('switching to the same box')
        #switch to the same box over and over (because those boxes aren't made yet)
        box = 'halos_z6.00_800_300Mpc_5015241'
        Halopos_z[z] = np.genfromtxt(halo_directory + box, dtype=None)
    else:
        box = 'halos_z'+str(np.round(z_range_of_halo_boxes[z],1))+'0_800_300Mpc_5015241'
        Halopos_z[z] = np.genfromtxt(halo_directory + box, dtype=None)

    os.system('echo Done redshift' + str(np.round(z_range_of_halo_boxes[z],1)))

#save the lightcone should something go very very wrong
np.savez('Halopos_z'+str(np.round(z_range_of_halo_boxes[z],1))+'_FRB.npz', Halopos_z = Halopos_z[z])

#do the lightcone for the Halo field
Halo_Position_Box = np.zeros((len(z_range_of_halo_boxes), HII_DIM, HII_DIM, HII_DIM))
Halo_Mass_Box = np.zeros_like(Halo_Position_Box)
for z in range(len(z_range_of_halo_boxes)):
    Halo_Position_Box[z] , Halo_Mass_Box[z] = misc.map2box(Halopos_z[z], HII_DIM)
Halo_lightcone, halolightcone_redshifts = lcH.lightcone(DIM = HII_DIM, halo_boxes_z =  Halo_Position_Box, z_range_of_boxes = z_range_of_halo_boxes, box_slice = int(box_slice), return_redshifts = True)




#load Fiducial stuff
npzfile = np.load('Halo_lightcone.npz', allow_pickle = True)
Halo_lightcone = npzfile['Halo_lightcone']

npzfile = np.load('Density_and_xH_lightcones.npz')
densitylightcone_beta_Dictionary = npzfile['densitylightcone_beta_Dictionary']
lightcone_redshifts_fiducial = npzfile['lightcone_redshifts']



####################################################################
#                 Define Bayesian Probabilities                    #
####################################################################

#lets add the number density to the prior. If we constrain it using the likelihood then we may end up in the
#unfortunate situation of having the code get stuck with a gigantic ACF
def lnprior(x):
    beta = x[0]
    zeta = x[1]
    Mturn = x[2]
    Rmfp = x[3]

    if  -1  < beta < 1 and  200 < zeta < 1000 and 1e7 < (Mturn*5e7) < 9e9  and 5 < Rmfp < 60:
        os.system("echo RUN " + str(RUN) + " accepting the fuck out of beta " + str(beta) + " "  + str(zeta) + " " + str(Mturn) + " " + str(Rmfp)   )
        return 0.0
    os.system("echo RUN " + str(RUN) + " Rejecting the fuck out of beta " + str(beta) +  " " + str(zeta) + " " + str(Mturn) + " " + str(Rmfp ) )
    return -np.inf


def lnprob(x, fiducial_data ):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x, fiducial_data)

beta_list = []
zeta_list = []
Mturn_list = []
model_DM_z = []
Rmfp_list = []
chi2_model = []
def lnlike(x, fiducial_data):
    #draw a tag for this run
    OUTPUT_NUMBER = int(np.abs(np.random.uniform(1000000, 9990000)))
    
    #map emcee space to EoR parameters
    beta = x[0]
    zeta = x[1]
    Mturn = x[2]
    Rmfp = x[3]

    beta_list.append(beta)
    zeta_list.append(zeta)
    Mturn_list.append(Mturn)
    Rmfp_list.append(Rmfp)

    if beta >= 0:
        sign = np.sign(-np.pi*(beta) + np.pi)
        sigma = np.abs(-np.pi*(beta) + np.pi)
    else:
        sign = np.sign(-np.pi*(beta) - np.pi)
        sigma = np.abs(-np.pi*(beta) - np.pi)
    
    t21_i = time.time()
    #make the reionization scenario for these parameters
    os.system("echo choice of beta is "  + str(beta) + ' leading to a sigma of' + str(sigma) +' with sign' + str(sign) )
    os.system("./init " + str(sign) + ' ' + str(sigma) +' ' + str(OUTPUT_NUMBER) )
    os.system("./drive_zscroll_noTs " + str(10*zeta) +' ' + str(Rmfp) +' ' + str(Mturn*5*10**7)+ ' '  + str(OUTPUT_NUMBER))
    t21_f = time.time()
    os.system("echo RUN " + str(RUN) + " 21cmfast runtime is " + str(t21_f - t21_i))


    #make lightcone for this model data
    os.system("echo n boxes is " + str(nboxes))
    #make the lightcone for each quantity
    box_slice = 199
    xH_lightcone_model , lightcone_redshifts = lc.lightcone(DIM = HII_DIM, z_range_of_boxes = z_range_of_halo_boxes, box_slice = int(box_slice), directory =  '../Boxes/', tag = OUTPUT_NUMBER, return_redshifts = True )
    
    os.system('echo Done making lightcone!')
    
    time_DM_start = time.time()
    #number of redshifts to include in our FRB plot
    lc_z_subsample = 10
    DM_z_y_z_model = np.zeros((lc_z_subsample, HII_DIM, HII_DIM ))
    n_FRBs_z = np.zeros((lc_z_subsample))
    z_we_are_actually_using = np.zeros((lc_z_subsample))
    #start with z = 9.6 for now (redshift = 0 index)
    for red_idx in range(lc_z_subsample):
        
        red = red_idx*int(len(lightcone_redshifts)/lc_z_subsample)
        z_we_are_actually_using[red_idx] =  lightcone_redshifts[red]
        os.system("echo Doing z" + str( lightcone_redshifts[red]))
        for y in range(HII_DIM):
            for z in range(HII_DIM):
                #if Halo_lightcone[red][y][z] != 0:
                #    n_FRBs_z[red_idx] += 1
                DM_z_y_z_model[red_idx][y][z] = DM.compute_DM(y,z, xH_lightcone_model[red:,:,], densitylightcone[red:,:,],  lightcone_redshifts[red:], Halo_lightcone[red:])
    time_DM_end = time.time()
    os.system("echo RUN " + str(RUN) + "  DM runtime is " + str(time_DM_end-time_DM_start))
    
    #sum DMs over all sightlines
    DM_z = (0.01**3)*np.true_divide(np.sum(DM_z_y_z_model, axis = (1,2)), (HII_DIM*HII_DIM*pc))
    
    #compute chi squared for each redshift and then add them up
    chi_squared_total = 0
    diff = np.zeros_like(DM_z)
    diff = np.subtract(DM_z, fiducial_DM_z)
    diff = np.true_divide(diff, 1)
    chi_squared_total += np.dot(diff, diff)
    
    os.system("echo RUN " + str(RUN) + " chi squared total is " + str(chi_squared_total) + str("for params") + str(zeta) + " " + str(Mturn) + " " + str(beta) + str(Rmfp))
    
    #add this chi2 to the list
    chi2_model.append(chi_squared_total)
    
    #add DM to the list
    model_DM_z.append(DM_z)

    #cleanup boxes that are leftover
    os.system("rm ../Boxes/*" + str(OUTPUT_NUMBER))

    #save results to an npz file
    np.savez('MCMC_snapshot_FRB' + str(RUN)+ '.npz', betas = np.array(beta_list) , zetas = np.array(zeta_list) , Mturns = np.array(Mturn_list),Rmfps = np.array(Rmfp_list) , model_DM_z = np.array(model_DM_z), chi2_model = np.array(chi2_model) )

    return -chi_squared_total/2.


####################################################################
#                       Make Mock Data                            #
####################################################################
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit()

#parameters used for making fiducial data
#we are using a fiducial inside-out model
npzfile = np.load('Fiducials.npz')
fiducial_DM_z = npzfile['fiducial_DM_z']
fiducial_redshifts = npzfile['fiducial_redshifts']



####################################################################
#              Define Starting Point and run the MCMC              #
####################################################################
randomize = np.random.normal(1, 0.1, ndim * nwalkers).reshape((nwalkers, ndim))
for i in range(nwalkers):
    randomize[i][0] = np.random.uniform(0.25,0.95)
    randomize[i][1] = np.random.uniform(300, 700)
    randomize[i][2] = np.random.uniform(5, 15)
    randomize[i][3] = np.random.uniform(20, 38)

#starting_parameters = randomize

#npzfile = np.load('checkpoint_values_FRB_full.npz')
#starting_parameters = npzfile['position']


os.system('echo Our starting parameters have been saved ')
np.savez('starting_params_full_FRB.npz' , starting_parameters = starting_parameters)


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool = pool,  args = [fiducial_DM_z])
pos , prob , state = sampler.run_mcmc(starting_parameters, 30)


####################################################################
#        Save MCMC results and checkpoint the progress             #
####################################################################

#save final position in an npz file to be ready afterwards
np.savez('checkpoint_values_FRB_full.npz', position = pos, probability = prob, stateof = state, acceptance_frac= np.mean(sampler.acceptance_fraction))

#write out chain data to npz files
np.savez('flatchain_FRB_' +str(RUN)+ '_full.npz', betas = sampler.flatchain[:,0],, zeta = sampler.flatchain[:,1] ,  Mturn = sampler.flatchain[:,2],  Rmfp = sampler.flatchain[:,3] ,  acceptance_frac= np.mean(sampler.acceptance_fraction))


np.savez('chain_FRB_' + str(RUN) +'_full.npz', samples = sampler.chain)


pool.close()




