import numpy as np
import matplotlib.pyplot as plt
import subprocess
import matplotlib.colors as colors
print("Please set BispecPy.exdir=\"path_to_C++_executables\"")
print("This is \"./src/\" from the default BispecPy.py location")

exdir="./src/"

def set_exdir(path):
    global exdir
    if path[-1]!="/":
        path+="/"
    exdir=path
    print("C++ binaries assumed to live in:",exdir)

def fullrun(nside=64,realspace=True,doPk=True,doBk=False
                        ,Pkfilename=""
                        ,filenamek="./src/data/delta_k.dat",filenamePk="./src/data/pk.dat",filenameBk="./src/data/bk.dat"
                        ,filenamer="./src/data/delta.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function creating a Gaussian density field
    
    """
    arg=[exdir+"fullrun","-nside",str(nside)]
    arg.append("-Pkfilename")
    arg.append(Pkfilename)
    arg.append("-filenamek")
    arg.append(filenamek)
    arg.append("-filenamer")
    arg.append(filenamer)
    arg.append("-filenamePk")
    arg.append(filenamePk)
    arg.append("-filenameBk")
    arg.append(filenameBk)
    if realspace:
        arg.append("-realspace")
    if quiet:
        arg.append("-quiet")
    if not doPk:
        arg.append("-doPk")
        arg.append("0")
    if doBk:
        arg.append("-doBk")
        arg.append("1")
    if n_thread is not None:
        arg.append("-n_thread")
        arg.append(str(int(n_thread)))
    if only_command:
        print(" ".join(arg))
        return
    print(subprocess.check_output(arg).decode("utf-8"))

def bispectrum(nside=64,start=0,step=1,filename="./src/data/delta_k.dat",fromrealspace=False
    ,filenameBk="./src/data/bk.dat",filenameBkind="./src/data/bkinds.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function creating a Gaussian density field
    
    """
    arg=[exdir+"bispectrum","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-start")
    arg.append(str(start))
    arg.append("-step")
    arg.append(str(step))
    arg.append("-filenameBk")
    arg.append(filenameBk)
    arg.append("-filenameBkind")
    arg.append(filenameBkind)
    if fromrealspace:
        arg.append("-fromrealspace")
    if quiet:
        arg.append("-quiet")
    if n_thread is not None:
        arg.append("-n_thread")
        arg.append(str(int(n_thread)))
    if only_command:
        print(" ".join(arg))
        return
    print(subprocess.check_output(arg).decode("utf-8"))

def powerspectrum(nside=64,filename="./src/data/delta_k.dat",fromrealspace=False
    ,filenamePk="./src/data/pk.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function creating a Gaussian density field
    
    """
    arg=[exdir+"powerspectrum","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-filenamePk")
    arg.append(filenamePk)
    if fromrealspace:
        arg.append("-fromrealspace")
    if quiet:
        arg.append("-quiet")
    if n_thread is not None:
        arg.append("-n_thread")
        arg.append(str(int(n_thread)))
    if only_command:
        print(" ".join(arg))
        return
    print(subprocess.check_output(arg).decode("utf-8"))

def fft3d(nside,inverse=False,filename="",filenamek="",n_thread=None,only_command=False):
    arg=[exdir+"fft3d","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-filenamek")
    arg.append(filenamek)
    if inverse:
        arg.append("-inverse")
    if n_thread is not None:
        arg.append("-n_thread")
        arg.append(str(int(n_thread)))
    if only_command:
        print(" ".join(arg))
        return
    print(subprocess.check_output(arg).decode("utf-8"))

def flatsize(ny):
    return int((ny+1)*(ny+2)*(ny+3)/6)
    
def memory_estimate(nside=256,start=0,step=1):
    print("These are rough estimates for nside="+str(nside))
    print()
    numks=(nside/2-start)//step+1
    fsize=flatsize(numks-1)
    print("fullrun")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8*(1+1))/1e9))
    print("  -realspace: {:.2f} GB".format((nside*nside*nside*8*(1+1+1+1/2))/1e9))
    print("gaussianfield")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8*(1+1))/1e9))
    print("  -realspace: {:.2f} GB".format((nside*nside*nside*8*(1+1+1))/1e9))
    print("powerspectrum")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8)/1e9))
    print("bispectrum for",numks,"ks, thus",flatsize(numks-1),"triplets.")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8+nside*nside*nside*(numks)*8*(1+1+1)+fsize*8*(1+1/2))/1e9))
    print("bispectrum_naive")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8*(1+1)+flatsize(nside/2)*8*(1+1/2))/1e9))

def plotrfield(filename="./src/data/delta.dat",nside=-1,z=0):
	if nside==-1:
		assert False, "nside must be specified"
	if nside%2!=0:
		assert False, "nside must be even"
	delta=np.fromfile(filename, dtype=float)
	delta=delta.reshape(nside,nside,nside)
	plt.figure(figsize=(8,8))
	plt.imshow(delta[:,:,z].T,origin="below")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Field delta at z="+str(z))
	plt.colorbar()
	plt.show()

def plotkfield(filename="./src/data/delta_k.dat",nside=-1,z=0):
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    delta_k=np.fromfile(filename, dtype=float)
    delta_k=delta_k.reshape(nside,nside,int(nside/2+1),2)
    plt.figure(figsize=(8,8))
    vmin=np.min(delta_k)
    vmax=np.max(delta_k)
    plt.subplot(221)
    plt.imshow(delta_k[:,:,z,0].T,origin="below"
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax, base=10))
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.title("Re[delta] at k_z="+str(z))
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(delta_k[:,:,z,1].T,origin="below"
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax, base=10))
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.title("Im[delta] at k_z="+str(z))
    plt.colorbar()
    plt.subplot(223)
    abs=(delta_k[:,:,z,0]**2+delta_k[:,:,z,1]**2)
    plt.imshow(abs.T,origin="below"
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(abs), vmax=np.max(abs), base=10))
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.title("ABS[delta]**2 at k_z="+str(z))
    plt.colorbar()
    plt.subplot(224)
    theta=np.arctan2(delta_k[:,:,z,1],delta_k[:,:,z,0])
    plt.imshow(theta.T,origin="below"
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(theta), vmax=np.max(theta), base=10))
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.title("theta[delta] at k_z="+str(z))
    plt.colorbar()
    plt.show()

def readPk(filename="./src/data/pk.dat",nside=-1,inputPK=None):
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    pk=np.fromfile(filename, dtype=float)
    assert len(pk)==int(nside/2+1),"Pk length not matching"
    return pk

def plotPk(filename="./src/data/pk.dat",nside=-1,inputPK=None):
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    pk=np.fromfile(filename, dtype=float)
    assert len(pk)==int(nside/2+1),"Pk length not matching"
    plt.plot(pk,marker="o",label="Estimate")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title("Power Spectrum")
    if inputPK=="default":
        ks=np.linspace(0,nside/2,1000)
        plt.plot(ks,(4*(np.sin(ks/4)**2/(ks+1)**2))**2,label="Default Input")
    plt.yscale("log")
    plt.legend()
    plt.show()

def plotBk(filenameBk="./src/data/bk.dat",filenameBkind="./src/data/bkinds.dat",nside=-1):
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"


    fs=flatsize(nside//2)
    bk=np.fromfile(filenameBk, dtype=np.float64)
    bkind=np.fromfile(filenameBkind, dtype=np.int32)
    bkind=bkind.reshape(fs,3)
    #print(bkind)

    im=np.zeros((nside//2+1,nside//2+1))
    count=np.zeros((nside//2+1,nside//2+1))
    for ind,onebk in zip(bkind,bk):
        if ind[0]==0 or ind[2]<(ind[0]-ind[1]):
            continue
        im[int(nside//2*ind[2]/ind[0]+0.5),int(nside//2*ind[1]/ind[0]+0.5)]+=onebk
        count[int(nside//2*ind[2]/ind[0]+0.5),int(nside//2*ind[1]/ind[0]+0.5)]+=1
    imshowfield=np.divide(im,count,out=np.zeros_like(im),where=count!=0)
    plt.imshow(imshowfield.T,origin="below"
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(imshowfield), vmax=np.max(imshowfield), base=10))
    plt.colorbar()
    plt.xlabel("k3/k1")
    plt.ylabel("k2/k1")
    plt.title("Bispectrum")
    plt.show()


import os
import shutil
import glob
import time
import datetime


fieldcount=0
datafolder="data"#CHANGING THIS IS VERY DANGEROUS
if not os.path.exists(datafolder):
    os.makedirs(datafolder)

class field():
    def __init__(self,data,nside,folname=None,copy=True):
        global datafolder,fieldcount
        if folname is None:
            self.folpath=datafolder+"/"+"dataset_"+str(fieldcount)+"/"
            fieldcount+=1
        else:
            self.folpath=datafolder+"/"+str(folname)+"/"
        if os.path.exists(self.folpath):
            files=glob.glob(self.folpath+"*")
            for f in files:
                os.remove(f)
        else:
            os.makedirs(self.folpath)
        if type(data)==str:
            if copy:
                shutil.copyfile(data,self.folpath+"delta.dat")
                self.deltapath=self.folpath+"delta.dat"
            else:
                self.deltapath=data
                file=open(self.folpath+"delta.dat","w")
                file.write("File at: "+data)
                file.close()
        elif type(data)==type(np.array([])):
            self.deltapath=self.folpath+"delta.dat"
            data.astype(np.float64).tofile(self.deltapath)
        self.nside=nside
        self.readmepath=self.folpath+"readme.md"
        self.writetoreadme("Created: "+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))+"\n")

    def writetoreadme(self,text):
        file=open(self.readmepath,"a+")
        file.write(text)
        file.close()

    def compute_fft3d(self):
        try:
            self.deltapath
        except:
            assert False, "No delta_k"
        self.deltakpath=self.folpath+"delta_k.dat"
        fft3d(self.nside,inverse=False,filename=self.deltapath,filenamek=self.deltakpath)
    def compute_ifft3d(self):
        try:
            self.deltakpath
        except:
            assert False, "No delta_k"
        self.deltapath=self.folpath+"delta.dat"
        fft3d(self.nside,inverse=True,filename=self.deltapath,filenamek=self.deltakpath)
    def compute_bispectrum(self,bkname=None,bkindname=None,start=0,step=1,quiet=False,n_thread=None,only_command=False):
        if bkname is None:
            bkname="bk.dat"
        self.bkpath=self.folpath+bkname
        if bkindname is None:
            bkindname="bkind.dat"
        self.bkindpath=self.folpath+bkindname
        try:
            self.deltakpath
        except:
            self.compute_fft3d()
        try:
            self.deltakpath
            res=bispectrum(nside=self.nside,start=start,step=step,filename=self.deltakpath,fromrealspace=False
                ,filenameBk=self.bkpath,filenameBkind=self.bkindpath
                ,quiet=quiet,n_thread=n_thread,only_command=only_command)
        except:
            assert False,"This should not happen"
            """
            res=bispectrum(nside=self.nside,start=start,step=step,filename=self.deltapath,fromrealspace=True
                ,filenameBk=self.bkpath,filenameBkind=self.bkindpath
                ,quiet=quiet,n_thread=n_thread,only_command=only_command)
            """
        return res

    def compute_powerspectrum(self,pkname=None,quiet=False,n_thread=None,only_command=False):
        if pkname is None:
            pkname="pk.dat"
        self.pkpath=self.folpath+pkname
        try:
            self.deltakpath
        except:
            self.compute_fft3d()
        try:
            self.deltakpath
            res=powerspectrum(nside=self.nside,filename=self.deltakpath,fromrealspace=False
                ,filenamePk=self.pkpath,quiet=quiet,n_thread=n_thread,only_command=only_command)
        except:
            assert False,"This should not happen"
            """
            res=powerspectrum(nside=self.nside,filename=self.deltapath,fromrealspace=True
                ,filenamePk=self.pkpath,quiet=quiet,n_thread=n_thread,only_command=only_command)
            """
        return res

    def plot(self,select="r",z=0):
        if select=="r":
            plotrfield(filename=self.deltapath,nside=self.nside,z=0)
        if select=="k":
            try:
                self.deltakpath
            except:
                self.compute_fft3d()
            plotkfield(filename=self.deltakpath,nside=self.nside,z=0)
        if select=="Pk":
            try:
                self.pkpath
            except:
                self.compute_powerspectrum()
            plotPk(filename=self.pkpath,nside=self.nside)
        if select=="Bk":
            try:
                self.bkpath
                self.bkindpath
            except:
                assert False, "Compute Bispectrum"
                #self.compute_bispectrum() #No due to memory
            plotBk(filenameBk=self.bkpath,filenameBkind=self.bkindpath,nside=self.nside)

    def data(self,select="r"):
        """
        Options:
          - r: realspace field
          - k: k-space field
          - Pk: Powerspectrum (ks,Pk)
          - Bk: Bispectrum (Bkinds,Bk)
        """
        if select=="r":
            return np.fromfile(self.deltapath,dtype=np.float64).reshape(self.nside,self.nside,self.nside)
        if select=="k":
            return np.fromfile(self.deltakpath,dtype=np.float64).reshape(self.nside,self.nside,int(self.nside/2+1),2)
        if select=="Pk":
            return (None,np.fromfile(self.pkpath, dtype=float))
        if select=="Bk":
            #change units
            return (np.fromfile(self.bkindpath, dtype=np.int32).reshape(fs,3)
                ,np.fromfile(self.bkpath, dtype=np.float64))
