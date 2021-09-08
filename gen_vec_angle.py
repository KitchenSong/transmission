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
                kvec = np.array(lines[i+1+nc*j].split()[3:6],dtype=float)
                ix = np.mod(np.array(kmesh*kvec,dtype=int),kmesh)
                kpoint[ix[0],ix[1],ix[2],0:3] = kvec
                for k in range(nbranch):
                    omega[ix[0],ix[1],ix[2],k] = float(lines[i+2+nc*j+k].split()[3])
                    vel[ix[0],ix[1],ix[2],k,0:3] = np.array(lines[i+2+nc*j+k].split()[5:8],dtype=float)
    print('Total number of k points: ' + str(ik))
            
    return kpoint,omega,vel


nt = 30
theta = np.linspace(0,np.pi/4,nt)
nw = 100
J2d = np.zeros((nw,nt))
for ita in range(nt):
# direction of transport (a unit vector)
    direction = np.array([0,np.cos(theta[ita]),np.sin(theta[ita])])
    direction = direction/np.linalg.norm(direction)

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
# heat flux
    nks = nx*ny*nz

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
    J2d[:,ita] = J[:]

#plt.plot(wlist,J)
#plt.ylabel('Transmission (vxT)')
#plt.xlabel('Frequency (Hz)')
#np.save('w.npy',wlist)
#np.save('J011.npy',J)
#plt.savefig('J.svg')
np.save('J2d.npy',J2d)
plt.imshow(J2d.T,origin='lower')
plt.show()




