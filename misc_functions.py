import numpy as np

rho_crit = 8.62*10**-27
mass_H = 1.672*10**-27
c = 3*10**8
H0 = float(68)/float(3.086e19)
OMm = 0.31
OMl = 0.75
baryon2DMfrac = 0.1
Hy2baryonfrac = 0.75
HyIGMfrac = 1
lya_min = 2.5e42
f12 = 0.4162
e = 1.602e-19
me = 9.11e-31
kb = 1.38064852e-23 #m^2 kg s^-2 K ^-1
mp = 1.672e-27 #kg
c = 3.0e8 #m/s
f_alpha_0 = 2.466e15 #1/s



    
def oops_we_are_at_the_edge(y,z, HII_DIM):
    if y == HII_DIM and z == HII_DIM:
        return int(HII_DIM - 1), int(HII_DIM - 1)
    else:
        if y == HII_DIM:
            return int(HII_DIM - 1) , z
        if z == HII_DIM:
            return y, int(HII_DIM - 1)

def map2box(list_to_map, HII_DIM):
    #HII_DIM is the target resolution of the output map
    Halo_Position_Box = np.zeros((HII_DIM, HII_DIM, HII_DIM))
    Halo_Mass_Box = np.zeros((HII_DIM, HII_DIM, HII_DIM))
    for i in range(list_to_map.shape[0]):
        x, y, z = np.round(HII_DIM*list_to_map[i][1],0), np.round(HII_DIM*list_to_map[i][2],0), np.round(HII_DIM*list_to_map[i][3],0)
        if y == HII_DIM or z == HII_DIM:
            y, z = oops_we_are_at_the_edge(int(y),int(z), HII_DIM   )
        if x == HII_DIM or z == HII_DIM:
            x, z = oops_we_are_at_the_edge(int(x),int(z), HII_DIM)
        Halo_Position_Box[int(x)][int(y)][int(z)] += 1
        Halo_Mass_Box[int(x)][int(y)][int(z)] += list_to_map[i][0]
    return Halo_Position_Box , Halo_Mass_Box


def sort_into_slices(list_to_sort, HII_DIM, slice):
    filtered_list = []
    for i in range(list_to_sort.shape[0]):
        if np.round(HII_DIM*list_to_sort[i][1],0) == slice:
            filtered_list.append(list_to_sort[i])
    return np.array(filtered_list)


def remove_los_from_list(list_of_halos):
    filtered_list = []
    for i in range(list_of_halos.shape[0]):
        filtered_list.append((list_of_halos[i][2], self.HII_DIM*list_of_halos[i][3]))
    return np.array(filtered_list)
