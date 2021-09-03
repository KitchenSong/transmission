import numpy as np
import re
import matplotlib.pyplot as plt
import time
import sys


def read_vec(fname):
    # read lattice vector and k grid 
    cell = np.zeros((3,3))
    f = open(fname,'r')
    lines = f.readlines()
    f.close()
    for i,line in enumerate(lines):
        if re.findall('cell',line):
            a = float(lines[i+1].split()[0])
            for j in range(3):
                l = lines[i+2+j].split()
                for k in range(3):
                    cell[j,k] = float(l[k])*a # in Angstrom
        elif re.findall('NKD',line):
            nkd = re.split('; | =',line)[1]
        elif re.findall('kpoint',line):
            nx = int(lines[i+2].split()[0])
            ny = int(lines[i+2].split()[1])
            nz = int(lines[i+2].split()[2])
            break
    recivec = np.transpose(2*np.pi*np.linalg.inv(cell))

    return cell,recivec,nx,ny,nz,nkd

def read_v_n_omega(fname,nbranch,nx,ny,nz):
    f = open(fname,'r')
    kmesh = np.array([nx,ny,nz],dtype=int)
    kpoint = np.zeros((nx,ny,nz,3))
    omega = np.zeros((nx,ny,nz,nbranch))
    vel = np.zeros((nx,ny,nz,nbranch,3))
    lines = f.readlines() 
    f.close()
    nl = len(lines)
    ik = 0
    nc = 2+nbranch
    for i in range(nl):
        if re.findall('# Irreducible k point  :',lines[i]):
            line = re.split('\( |\)|:|\n',lines[i])
            knum = int(line[2])
            ik += knum
            for j in range(knum):
                kvec = np.array(lines[i+1].split()[3:6],dtype=float)
                ix = np.array(np.remainder(kmesh*kvec,kmesh),dtype=int)
                kpoint[ix[0],ix[1],ix[2],0:3] = kvec
                for k in range(nbranch):
                    omega[ix[0],ix[1],ix[2],k] = float(lines[i+2+nc*j+k].split()[3])
                    vel[ix[0],ix[1],ix[2],k,0:3] = np.array(lines[i+2+nc*j+k].split()[4:7],dtype=float)

    print('Total number of k points: ' + str(ik))
            
            #break

    return kpoint,omega,vel


natm = 2 # num of atoms in primitive cell
nbranch = natm * 3
# get cell
cell,recivec,nx,ny,nz,nkd = read_vec('ge_group-vel_10x10x10.in')
# get volume of primitive cell
vol = np.dot(np.cross(cell[0,:],cell[1,:]),cell[2,:])
# get velocity
kpoint,omega,vel = read_v_n_omega('ge_10x10x10.phvel_all',nbranch,nx,ny,nz)
omega = omega * 0.029979245800e12*2*np.pi
# gaussian smearing width
sigma = 0.3e12*2*np.pi
# direction of transport (a unit vector)
direction = np.array([0,0,1])
# heat flux
nks = nx*ny*nz
nw = 100
wlist = np.linspace(0,10,nw)*1e12*2*np.pi
J = np.zeros(wlist.shape)

for w in range(nw):
    sys.stdout.write("\r%d%%" % int(w/(nw-1.0)*100))
    sys.stdout.flush()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in range(nbranch):
                    J[w] += abs(np.dot(vel[i,j,k,b,:],direction))*np.exp(-(wlist[w]-omega[i,j,k,b])**2/sigma**2)/vol*(2*np.pi)**2/nks

plt.plot(wlist,J)
plt.ylabel('Transmission (vxT)')
plt.xlabel('Frequency (Hz)')
plt.savefig('J.svg')
plt.show()




