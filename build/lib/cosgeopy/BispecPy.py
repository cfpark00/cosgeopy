import numpy as np
import matplotlib.pyplot as plt
import subprocess
import matplotlib.colors as colors
import pickle
print("Please set BispecPy.exdir=\"path_to_C++_executables\"")
print("This is \"./src/\" from the default BispecPy.py location")

exdir="./src/"

def set_exdir(path):
    global exdir
    if path[-1]!="/":
        path+="/"
    exdir=path
    print("C++ binaries assumed to live in:",exdir)

def bispectrum(nside=64,k_min=1,k_max=-1,step=1,filename="./src/data/delta_k.dat",fromrealspace=False
    ,filenameBk="./src/data/bk.dat",filenameBkind="./src/data/bkinds.dat",filenameBkcount="./src/data/bkcount.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function calculating Bispectrum
    
    """
    if k_max==-1:
        k_max=nside//2
    arg=[exdir+"bispectrum","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-k_min")
    arg.append(str(k_min))
    arg.append("-k_max")
    arg.append(str(k_max))
    arg.append("-step")
    arg.append(str(step))

    arg.append("-filenameBk")
    arg.append(filenameBk)
    arg.append("-filenameBkind")
    arg.append(filenameBkind)
    arg.append("-filenameBkcount")
    arg.append(filenameBkcount)
    
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

def IsoWc(nside=64,order=1,sigma=1,loadfilt=False,filename="./src/data/delta_k.dat",fromkspace=False
    ,filenameWC="./src/data/wc.dat",filenameWCind="./src/data/wcind.dat",filename_filt=""
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function for isotropic wavelet scattering
    
    """
    arg=[exdir+"isotropic_wavelet","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-order")
    arg.append(str(order))
    arg.append("-sigma")
    arg.append(str(sigma))

    arg.append("-filenameWC")
    arg.append(filenameWC)
    arg.append("-filename_filt")
    arg.append(filename_filt)
    arg.append("-filenameWCind")
    arg.append(filenameWCind)
    
    if fromkspace:
        arg.append("-fromkspace")
    if loadfilt:
        arg.append("-loadfilt")
    if quiet:
        arg.append("-quiet")
    if n_thread is not None:
        arg.append("-n_thread")
        arg.append(str(int(n_thread)))
    if only_command:
        print(" ".join(arg))
        return
    print(subprocess.check_output(arg).decode("utf-8"))

def bispectrum_custom_k(nside=64,numtrips=-1,filename="./src/data/delta_k.dat",fromrealspace=False
    ,filenameBk="./src/data/bk.dat",filenameBkind="./src/data/bkinds.dat",filenameBkcount="./src/data/bkcount.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function creating a Gaussian density field
    
    """
    arg=[exdir+"bispectrum_custom_k","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-numtrips")
    arg.append(str(numtrips))

    arg.append("-filenameBk")
    arg.append(filenameBk)
    arg.append("-filenameBkind")
    arg.append(filenameBkind)
    arg.append("-filenameBkcount")
    arg.append(filenameBkcount)
    
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

def powerspectrum(nside=64,filename="./src/data/delta_k.dat",fromrealspace=False,mas=0
    ,filenameks="./src/data/ks.dat",filenamePk="./src/data/pk.dat"
                        ,quiet=False,n_thread=None,only_command=False):
    """
    Function creating a Gaussian density field
    
    """
    arg=[exdir+"powerspectrum","-nside",str(nside)]
    arg.append("-filename")
    arg.append(filename)
    arg.append("-filenamePk")
    arg.append(filenamePk)
    arg.append("-filenameks")
    arg.append(filenameks)
    arg.append("-mas")
    arg.append(str(mas))
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

def flatsize(k_min,k_max):
    c=0
    for i in range(k_min,k_max+1):
        for j in range(k_min,i+1):
            for k in range(k_min,j+1):
                if (k<(i-j)):
                    continue
                c+=1
    return c

def WCsize(order,numJs):
    count=0
    for m in range(order+1):
        quo=1
        div=1
        for i in range(m):
            quo*=(numJs-i)
            div*=(i+1)
        count+=(quo/div)
    return count


def get_numJs(nside):
    numJs=0
    twototheJ=1
    while(twototheJ*2 <= nside):
        twototheJ*=2
        numJs+=1
    return numJs

def memory_estimate(nside=256,k_min=1,k_max=-1,step=1,numtrips=0,numunique=-1,order=1):
    print("These are rough estimates for nside="+str(nside))
    print()
    if k_max==-1:
        k_max=nside//2
    if numunique==-1:
        numunique=numtrips
    fsize=flatsize(k_min//step,k_max//step)
    numks=k_max//step-k_min//step

    csize=nside*nside*(nside//2+1)
    size=nside*nside*nside

    print("fullrun")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8*(1+1))/1e9))
    print("  -realspace: {:.2f} GB".format((nside*nside*nside*8*(1+1+1+1/2))/1e9))
    print("gaussianfield")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8*(1+1))/1e9))
    print("  -realspace: {:.2f} GB".format((nside*nside*nside*8*(1+1+1))/1e9))
    print("powerspectrum")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8)/1e9))
    print("bispectrum for",numks,"ks, thus",fsize,"triplets.")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8+nside*nside*nside*(numks+1)*8*(1+1+1)+fsize*8*(1+1/2))/1e9))
    print("bispectrum custom for",numtrips,"triplets.")
    print("  -default: {:.2f} GB".format((nside*nside*nside*8+nside*nside*nside*(numunique+1)*8*(1+1+1)+numtrips*8*(1+1/2))/1e9))

    numJs=get_numJs(nside)
    fsize=WCsize(order,numJs)
    isomemGB=((order+1+1)*size+(numJs+1)*csize*2)*8/1e9
    print("IsoWc for order",order,", thus",fsize,"coefficients.")
    print("  -default: {:.2f} GB".format(isomemGB))

def plotrfield(filename="./src/data/delta.dat",nside=-1,physicalsize=None,z=0):
    if physicalsize==None:
        physicalsize=nside
    else:
        physicalsize=physicalsize
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    delta=np.fromfile(filename, dtype=float)
    delta=delta.reshape(nside,nside,nside)
    plt.figure(figsize=(8,8))
    plt.imshow(delta[:,:,z].T,extent=[0,physicalsize,0,physicalsize],origin="below")
    plt.xlabel("x [Mpc]")
    plt.ylabel("y [Mpc]")
    plt.title("Field delta at z="+str(z)+" [Mpc]")
    plt.colorbar()
    plt.show()

def shift(field2dk):
    halfside=int(field2dk.shape[0]/2)
    return np.concatenate((np.concatenate((field2dk[halfside:,halfside:],field2dk[:halfside,halfside:]),axis=0)
        ,np.concatenate((field2dk[halfside:,:halfside],field2dk[:halfside,:halfside]),axis=0)),axis=1)

def plotkfield(filename="./src/data/delta_k.dat",nside=-1,physicalsize=None,k_z=0):
    if physicalsize==None:
        physicalsize=nside
    else:
        physicalsize=physicalsize
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    k_ny=2*np.pi*nside/physicalsize/2
    delta_k=np.fromfile(filename, dtype=float)
    delta_k=delta_k.reshape(nside,nside,int(nside/2+1),2)
    plt.figure(figsize=(8,8))
    vmin=np.min(delta_k)
    vmax=np.max(delta_k)
    z=k_z
    plt.subplot(221)
    plt.imshow(shift(delta_k[:,:,z,0]).T,origin="below",extent=[-k_ny,k_ny,-k_ny,k_ny]
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax, base=10))
    plt.xlabel("k_x [1/Mpc]")
    plt.ylabel("k_y [1/Mpc]")
    plt.title("Re[delta] at k_z="+str(k_z)+" [1/Mpc]")
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(shift(delta_k[:,:,z,1]).T,origin="below",extent=[-k_ny,k_ny,-k_ny,k_ny]
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax, base=10))
    plt.xlabel("k_x [1/Mpc]")
    plt.ylabel("k_y [1/Mpc]")
    plt.title("Im[delta] at k_z="+str(k_z)+" [1/Mpc]")
    plt.colorbar()
    plt.subplot(223)
    abs=(delta_k[:,:,z,0]**2+delta_k[:,:,z,1]**2)
    plt.imshow(shift(abs).T,origin="below",extent=[-k_ny,k_ny,-k_ny,k_ny]
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(abs), vmax=np.max(abs), base=10))
    plt.xlabel("k_x [1/Mpc]")
    plt.ylabel("k_y [1/Mpc]")
    plt.title("ABS[delta]**2 at k_z="+str(k_z)+" [1/Mpc]")
    plt.colorbar()
    plt.subplot(224)
    theta=np.arctan2(delta_k[:,:,z,1],delta_k[:,:,z,0])
    plt.imshow(shift(theta).T,origin="below",extent=[-k_ny,k_ny,-k_ny,k_ny]
        ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(theta), vmax=np.max(theta), base=10))
    plt.xlabel("k_x [1/Mpc]")
    plt.ylabel("k_y [1/Mpc]")
    plt.title("theta[delta] at k_z="+str(k_z)+" [1/Mpc]")
    plt.colorbar()
    plt.show()

def plotPk(filenameks="./src/data/ks.dat",filenamePk="./src/data/pk.dat",nside=-1,physicalsize=None,inputPK=None):
    if physicalsize==None:
        physicalsize=nside
    else:
        physicalsize=physicalsize
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    pk=np.fromfile(filenamePk, dtype=float)
    ks=np.fromfile(filenameks, dtype=float)
    ks,pk=handleunits((ks,pk),mode="Pk",nside=nside,physicalsize=physicalsize)
    plt.plot(ks,pk,marker="o",label="Estimate")
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("P(k)")
    plt.title("Power Spectrum")
    if inputPK=="default":
        plt.plot(ks,(4*(np.sin(ks/4)**2/(ks+1)**2))**2,label="Default Input")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()

def plotBk(filenameBk="./src/data/bk.dat",filenameBkind="./src/data/bkinds.dat",filenameBkcount="./src/data/bkcount.dat",nside=-1,physicalsize=None,args={},**kwargs):
    if physicalsize==None:
        physicalsize=nside
    else:
        physicalsize=physicalsize
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"

    bkind,bk,bkcount=handleunits((np.fromfile(filenameBkind, dtype=np.int32),np.fromfile(filenameBk, dtype=np.float64)
        ,np.fromfile(filenameBkcount, dtype=np.float64)),mode="Bk",nside=nside,physicalsize=physicalsize)
    bkind=bkind.reshape(-1,3)

    res=20
    if "res" in args:
        res=args["res"]
    im=np.zeros((res+1,res+1))
    counts=np.zeros((res+1,res+1))
    for ind,onebk,count in zip(bkind,bk,bkcount):
        im[int(res*ind[2]/ind[0]+0.5),int(res*ind[1]/ind[0]+0.5)]+=onebk*count
        counts[int(res*ind[2]/ind[0]+0.5),int(res*ind[1]/ind[0]+0.5)]+=count
    imshowfield=np.divide(im,counts,out=np.zeros_like(im),where=counts!=0)
    plt.imshow(imshowfield.T,origin="below"
    ,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.min(imshowfield), vmax=np.max(imshowfield), base=10),**kwargs)
    plt.colorbar()
    plt.xlabel("k3/k1")
    plt.ylabel("k2/k1")
    plt.title("Bispectrum")
    plt.show()

def handleunits(data,mode,nside,physicalsize):
    if mode=="Pk":
        ks=data[0]
        Pk=data[1]
        k_ny=(2*np.pi)*nside/physicalsize/2
        return ks/(nside//2)*k_ny,Pk*((physicalsize**3)**(2-1)/(nside**3)**2)
    if mode=="Bk":
        Bkind=data[0]
        Bk=data[1]
        Bkcount=data[2]
        k_ny=(2*np.pi)*nside/physicalsize/2
        return Bkind/(nside//2)*k_ny,Bk*((physicalsize**3)**(3-1)/(nside**3)**3),Bkcount
    if mode=="k":
        return data*(physicalsize/nside)**3

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
    def __init__(self,data,nside,physicalsize=None,folname=None,copy=True):
        global datafolder,fieldcount

        if physicalsize==None:
            self.physicalsize=nside
        else:
            self.physicalsize=physicalsize

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
        self.k_ny=(2*np.pi)*nside/physicalsize/2
        self.k_fund=2*np.pi/physicalsize
        self.readmepath=self.folpath+"readme.md"
        self.writetoreadme("Created: "+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))+"\n")
        self.writetoreadme("nside: "+str(self.nside)+"\n")
        self.writetoreadme("physicalsize: "+str(self.physicalsize)+"\n")
        self.writetoreadme("  [derived]k_ny: "+str(self.k_ny)+"\n")
        self.writetoreadme("  [derived]k_fund: "+str(self.k_fund)+"\n")
        self.writetoreadme("\n")
        self.writetoreadme("folpath: "+self.folpath+"\n")
        self.writetoreadme("deltapath: "+self.deltapath+"\n")
        self.writetoreadme("\n")

    def writetoreadme(self,text):
        file=open(self.readmepath,"a+")
        file.write(text)
        file.close()

    def get_memory_estimate(self,k_min=1,k_max=-1,step=1,numtrips=0,numunique=-1,order=1):
        memory_estimate(nside=self.nside,k_min=k_min,k_max=k_max,step=step,numtrips=numtrips,numunique=numunique,order=order)

    def compute_fft3d(self):
        try:
            self.deltapath
        except:
            assert False, "No delta_k"
        recomp=False
        try:
            self.deltakpath
            recomp=True
        except:
            pass

        self.deltakpath=self.folpath+"delta_k.dat"
        fft3d(self.nside,inverse=False,filename=self.deltapath,filenamek=self.deltakpath)
        if recomp:
            self.writetoreadme("Recomputed delta_k"+"\n")
            self.writetoreadme("  deltakpath: "+self.deltakpath+"\n")
            self.writetoreadme("\n")
        else:
            self.writetoreadme("Computed delta_k"+"\n")
            self.writetoreadme("  deltakpath: "+self.deltakpath+"\n")
            self.writetoreadme("\n")

    def compute_ifft3d(self):
        try:
            self.deltakpath
        except:
            assert False, "No delta_k"
        recomp=False
        try:
            self.deltapath
            recomp=True
        except:
            pass

        self.deltapath=self.folpath+"delta.dat"
        fft3d(self.nside,inverse=True,filename=self.deltapath,filenamek=self.deltakpath)
        if recomp:
            self.writetoreadme("Computed delta by IFT"+"\n")
            self.writetoreadme("  deltapath: "+self.deltapath+"\n")
            self.writetoreadme("\n")
        else:
            self.writetoreadme("Computed delta by IFT"+"\n")
            self.writetoreadme("  deltapath: "+self.deltapath+"\n")
            self.writetoreadme("\n")

    def compute_bispectrum(self,bkname=None,bkindname=None,bkcountname=None,k_min=1,k_max=-1,step=1,quiet=False,n_thread=None,only_command=False):
        if k_max==-1:
            k_max=nside//2
        if bkname is None:
            bkname="bk.dat"
        self.bkpath=self.folpath+bkname
        if bkindname is None:
            bkindname="bkind.dat"
        self.bkindpath=self.folpath+bkindname
        if bkcountname is None:
            bkcountname="bkcount.dat"
        self.bkcountpath=self.folpath+bkcountname
        try:
            self.deltakpath
        except:
            self.compute_fft3d()

        self.deltakpath
        res=bispectrum(nside=self.nside,k_min=k_min,k_max=k_max,step=step,filename=self.deltakpath,fromrealspace=False
            ,filenameBk=self.bkpath,filenameBkind=self.bkindpath,filenameBkcount=self.bkcountpath
            ,quiet=quiet,n_thread=n_thread,only_command=only_command)
        self.k_min=k_min
        self.k_max=k_max

        self.writetoreadme("Computed Bk"+"\n")
        self.writetoreadme("  bkpath: "+self.bkpath+"\n")
        self.writetoreadme("  bkindpath: "+self.bkindpath+"\n")
        self.writetoreadme("  bkcountpath: "+self.bkcountpath+"\n")
        self.writetoreadme("\n")
        return res

    def compute_bispectrum_custom_k(self,bkind=None,bkname=None,bkindname=None,bkcountname=None,quiet=False,n_thread=None,only_command=False):
        if bkname is None:
            bkname="bk.dat"
        self.bkpath=self.folpath+bkname
        if bkind is not None:
            bkindname="bkind.dat"
            self.bkindpath=self.folpath+bkindname
            bkind=bkind.astype(np.int32).tofile(self.bkindpath)
            self.numtrips=len(bkind)
        elif bkindname is None:
            bkindname="bkind.dat"
            self.bkindpath=self.folpath+bkindname
        else:
            self.bkindpath=self.folpath+bkindname
        if bkcountname is None:
            bkcountname="bkcount.dat"
        self.bkcountpath=self.folpath+bkcountname
        try:
            self.deltakpath
        except:
            self.compute_fft3d()

        res=bispectrum_custom_k(nside=self.nside,numtrips=self.numtrips,filename=self.deltakpath,fromrealspace=False
            ,filenameBk=self.bkpath,filenameBkind=self.bkindpath,filenameBkcount=self.bkcountpath
            ,quiet=quiet,n_thread=n_thread,only_command=only_command)

        self.writetoreadme("Computed Bk"+"\n")
        self.writetoreadme(" bkpath: "+self.bkpath+"\n")
        self.writetoreadme("  bkindpath: "+self.bkindpath+"\n")
        self.writetoreadme("  bkcountpath: "+self.bkcountpath+"\n")
        self.writetoreadme("\n")
        return res

    def compute_powerspectrum(self,mas=0,ksname=None,pkname=None,quiet=False,n_thread=None,only_command=False):
        if pkname is None:
            pkname="pk.dat"
        self.pkpath=self.folpath+pkname
        if ksname is None:
            ksname="ks.dat"
        self.kspath=self.folpath+ksname
        try:
            self.deltakpath
        except:
            self.compute_fft3d()

        res=powerspectrum(nside=self.nside,filename=self.deltakpath,fromrealspace=False,mas=mas
            ,filenameks=self.kspath,filenamePk=self.pkpath,quiet=quiet,n_thread=n_thread,only_command=only_command)

        self.writetoreadme("Computed Pk"+"\n")
        self.writetoreadme("  pkpath: "+self.pkpath+"\n")
        self.writetoreadme("  kspath: "+self.kspath+"\n")
        self.writetoreadme("\n")

        return res

    def compute_IsoWc(self,order=1,sigma=1,loadfilt=False,filename_filt="",wcindname=None,wcname=None,quiet=False,n_thread=None,only_command=False):
        if wcname is None:
            wcname="wc.dat"
        self.wcpath=self.folpath+wcname

        if wcindname is None:
            wcindname="wcind.dat"
        self.wcindpath=self.folpath+wcindname

        self.wc_order=order
        self.wc_sigma=sigma
        self.numJs=get_numJs(self.nside)
        res=IsoWc(nside=self.nside,order=self.wc_order,sigma=self.wc_sigma,loadfilt=loadfilt,filename_filt=filename_filt,filename=self.deltapath,fromkspace=False
            ,filenameWC=self.wcpath,filenameWCind=self.wcindpath,quiet=quiet,n_thread=n_thread,only_command=only_command)

        self.writetoreadme("Computed Isotropic Wavelet Coefficients"+"\n")
        self.writetoreadme("  wcpath: "+self.wcpath+"\n")
        self.writetoreadme("  wcindpath: "+self.wcindpath+"\n")
        self.writetoreadme("\n")
        return res


    def plot(self,select="r",z=0,args={},**kwargs):
        if select=="r":
            plotrfield(filename=self.deltapath,nside=self.nside,physicalsize=self.physicalsize,z=0)
        if select=="k":
            try:
                self.deltakpath
            except:
                assert False, "Compute FFT3D"
            plotkfield(filename=self.deltakpath,nside=self.nside,physicalsize=self.physicalsize,k_z=z)
        if select=="Pk":
            try:
                self.pkpath
                self.kspath
            except:
                assert False, "Compute Powerspectrum"
            plotPk(filenameks=self.kspath,filenamePk=self.pkpath,nside=self.nside,physicalsize=self.physicalsize)
        if select=="Bk":
            try:
                self.bkpath
                self.bkindpath
                self.bkcountpath
            except:
                assert False, "Compute Bispectrum"
                #self.compute_bispectrum() #No due to memory
            plotBk(filenameBk=self.bkpath,filenameBkind=self.bkindpath,filenameBkcount=self.bkcountpath,nside=self.nside,physicalsize=self.physicalsize,args=args,**kwargs)
        if select=="WC":
            try:
                self.wcpath
                self.wcindpath
            except:
                assert False, "Compute Wavelet Coefficients"
            assert False, "Not implemented"
            #plotPk(filenameks=self.kspath,filenamePk=self.pkpath,nside=self.nside,physicalsize=self.physicalsize)

    def data(self,select="r",units="Physical"):
        """
        Options:
          - r: realspace field
          - k: k-space field
          - Pk: Powerspectrum (ks,Pk)
          - Bk: Bispectrum (Bkinds,Bk)
        """
        if units=="Physical":
            if select=="r":
                return np.fromfile(self.deltapath,dtype=np.float64).reshape(self.nside,self.nside,self.nside)
            if select=="k":
                return handleunits(np.fromfile(self.deltakpath,dtype=np.float64).reshape(self.nside,self.nside,int(self.nside/2+1),2)
                    ,mode=select,nside=self.nside,physicalsize=self.physicalsize)
            if select=="Pk":
                return handleunits((np.fromfile(self.kspath, dtype=float),np.fromfile(self.pkpath, dtype=float))
                    ,mode=select,nside=self.nside,physicalsize=self.physicalsize)
            if select=="Bk":
                #change units
                return handleunits((np.fromfile(self.bkindpath, dtype=np.int32).reshape(-1,3)
                    ,np.fromfile(self.bkpath, dtype=np.float64),np.fromfile(self.bkcountpath, dtype=np.float64))
                ,mode=select,nside=self.nside,physicalsize=self.physicalsize)
            if select=="WC":
                #change units
                return np.fromfile(self.wcpath, dtype=np.float64)
                return handleunits((np.fromfile(self.bkindpath, dtype=np.int32).reshape(-1,3)
                    ,np.fromfile(self.bkpath, dtype=np.float64),np.fromfile(self.bkcountpath, dtype=np.float64))
                ,mode=select,nside=self.nside,physicalsize=self.physicalsize)
        elif units=="Raw":
            if select=="r":
                return np.fromfile(self.deltapath,dtype=np.float64).reshape(self.nside,self.nside,self.nside)
            if select=="k":
                return np.fromfile(self.deltakpath,dtype=np.float64).reshape(self.nside,self.nside,int(self.nside/2+1),2)
            if select=="Pk":
                return (np.linspace(0,self.nside/2,self.nside//2+1),np.fromfile(self.pkpath, dtype=float))
            if select=="Bk":
                #change units
                return (np.fromfile(self.bkindpath, dtype=np.int32).reshape(-1,3)
                    ,np.fromfile(self.bkpath, dtype=np.float64))

    def save(self,file):
        pickle.dump(self,open(file,"wb"))

    @classmethod
    def loader(field,file):
        return pickle.load(open(file,"rb"))

























"""
def readPk(filename="./src/data/pk.dat",nside=-1,physicalsize=None,inputPK=None):
    if physicalsize==None:
        self.physicalsize=nside
    else:
        self.physicalsize=physicalsize
    if nside==-1:
        assert False, "nside must be specified"
    if nside%2!=0:
        assert False, "nside must be even"
    pk=np.fromfile(filename, dtype=float)
    assert len(pk)==int(nside/2+1),"Pk length not matching"
    return pk
"""

"""
def fullrun(nside=64,realspace=True,doPk=True,doBk=False
                        ,Pkfilename=""
                        ,filenamek="./src/data/delta_k.dat",filenamePk="./src/data/pk.dat",filenameBk="./src/data/bk.dat"
                        ,filenamer="./src/data/delta.dat"
                        ,quiet=False,n_thread=None,only_command=False):

    #Function creating a Gaussian density field
    

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
"""
