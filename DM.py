import numpy as np
import os


#cosmology constants
c = 3*10**8
H0 = float(68)/float(3.086e19)
OMm = 0.31
OMl = 0.75
rho_crit = 8.62*10**-27
mp = 1.672e-27
baryon2DMfrac = 0.1
Hy2baryonfrac = 0.75
HyIGMfrac = 1


#this code computes the dispersion measure of an FRB at location x,y in a box along the LoS from redshift
#z_i to z_end
def compute_DM(x,y, xH_lightcone, density_lightcone, lightcone_redshifts, z_end, **kwargs):
    
        #starting redshift
        #print(lightcone_redshifts.shape)
        z_start = lightcone_redshifts[0]
        
        #ending redshift
        z_end = z_end
        
        #initialize the DM
        DM = 0
        
        #the redshifts of each cell in the lightcone will tell us what the redshift spacing is
        #print(lightcone_redshifts[0], lightcone_redshifts[1])
        delta_z = lightcone_redshifts[1] - lightcone_redshifts[0]
        #print('delta z is ' , delta_z   )
        
        #as an approximation, we may not want to integrate through the entire lightcone
        #if not specified by the user then assume we are integrating through the entire lightcone
        if 'depth' in kwargs:
            depth = kwargs.get('depth')
        else:
            depth = lightcone_redshifts.shape[0]

        #loop through the entire box
        z_i = z_start
        i = 0
#print('Doing z_i and z_end', z_i, z_end)
        while(z_i < z_end):
            
            #the redshift at this location is
            z_i = z_start +delta_z*i
            #print(z_i, i)
            #print('the redshift is ' , z_i)
            
            #H(z_i) is
            H = H0*np.sqrt(OMm*(1 + z_i)**3 + OMl)
            
            #compute the dispersion
            #print(density_lightcone[i][x][y], xH_lightcone[i][x][y] )
            insta_DM  = float(((1+z_i)**3)*c*delta_z*np.abs(float(1) + float(density_lightcone[i][x][y]))*(float(1) - float(xH_lightcone[i][x][y]))*rho_crit*OMm*baryon2DMfrac*HyIGMfrac)/float(mp*H*(1 + z_i)**2)
            if insta_DM < 0 :
                print('lightcone:' , (float(1) - float(xH_lightcone[i][x][y])))
                print('redshit and delta_z', z_i, delta_z)
            
            #print('at i' + str(i) + ' we are at redshift ' + str(z_i) + ' and are adding ' + str(insta_DM) + ' to our DM')
            DM += ((1+z_i)**3)*c*delta_z*np.abs(float(1) + float(density_lightcone[i][x][y]))*(float(1) - float(xH_lightcone[i][x][y]))*rho_crit*OMm*baryon2DMfrac*HyIGMfrac/float(mp*H*(1 + z_i)**2)
            i +=1
#print('checked out at z_i', z_i)
        return DM

