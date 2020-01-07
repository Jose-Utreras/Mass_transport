import sys
import numpy as np
import yt
from scipy.optimize import curve_fit
from yt.fields.api import ValidateParameter
from yt.units import kpc,pc,km,second,yr,Myr,Msun,kilometer,G
from astropy.table import Table , Column ,vstack,hstack

def _Disk_Radius(field, data):
    center = data.get_field_parameter('center')
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    r = np.sqrt(x*x+y*y)
    return r
def _Disk_H(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return np.abs(z)
def _vc(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        xv = data["gas","velocity_x"] - bv[0]
        yv = data["gas","velocity_y"] - bv[1]
        center = data.get_field_parameter('center')
        x_hat = data["x"] - center[0]
        y_hat = data["y"] - center[1]
        r = np.sqrt(x_hat*x_hat+y_hat*y_hat)
        x_hat /= r
        y_hat /= r

        return (yv*x_hat-xv*y_hat)
def _Disk_Angle(field, data):
    center = data.get_field_parameter('center')
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    r = np.arctan2(y,x)
    return r
def _vertical_velocity(field,data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
    else:
        bv = data.ds.arr(np.zeros(3), "cm/s")
    v = data["gas","velocity_z"] - bv[2]
    return v
yt.add_field("Disk_Radius",
             function=_Disk_Radius,
             units="cm",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("Disk_H",
             function=_Disk_H,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("vc", function=_vc,
        take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])
yt.add_field("Disk_Angle",
             function=_Disk_Angle,
             units="dimensionless",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field("vertical_velocity",
        function=_vertical_velocity,take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])


def Smooth(x,y,n=1,b=1):
        NN=len(y[:])
        z=np.zeros(NN)
        d=np.zeros(NN)
        X=np.zeros(NN+2*n)
        Y=np.zeros(NN+2*n)


        for i in range(n):
                X[n-i-1]=x[0]-(i+1)*(x[1]-x[0])
                Y[n-i-1]=y[0]
        for i in np.arange(NN,NN+2*n,1):
                X[i]=x[NN-1]+(i+1-NN)*(x[1]-x[0])
                Y[i]=y[NN-1]
        count = n
        for xa,xb in zip(x,y):
                X[count]=xa
                Y[count]=xb
                count+=1
        for i in range(len(x)):
                for j in range(2*n+1):
                        z[i]=z[i]+np.exp( -(X[i+n]-X[i+j])**2 /(2*b**2) )*Y[i+j]
                        d[i]=d[i]+np.exp( -(X[i+n]-X[i+j])**2 /(2*b**2) )

        return z/d
def radial_map_N(N):
    X=np.array(list(np.reshape(range(N),(1,N)))*N)
    Y=X.T
    X=np.reshape(X,(N,N))-(N-1)/2.0
    Y=np.reshape(Y,(N,N))-(N-1)/2.0
    R=np.sqrt(X**2+Y**2)
    return R
def XY_map_N(N):
    X=np.array(list(np.reshape(range(N),(1,N)))*N)

    Y=X.T
    X=np.reshape(X,(N,N))-(N-1)/2.0
    Y=np.reshape(Y,(N,N))-(N-1)/2.0
    return X,Y
def add_extremes(X):
    X=np.insert(X,0,0)
    X=np.insert(X,len(X),2*X[-1]-X[-2])
    return X
def V_curve(R,Vo,R1):
    return Vo*np.arctan(R/R1)
def Velocity_curve(R,Vo,R1,R2):
    return Vo*np.arctan(R/R1)*np.exp(-R/R2)
def Vorticity_curve(R,Vo,R1,R2):
    result = np.piecewise(R,[R<1e-3,R>=1e-3],[lambda R:2*Vo/R1,
             lambda R:Vo*np.exp(-R/R2)*(R1/(R1**2+R**2) +
                np.arctan(R/R1)*(1.0/R-1.0/R2))])

    return result
def Kappa_curve(R,Vo,R1,R2):
    result = 2*Omega_curve(R,Vo,R1,R2)*Vorticity_curve(R,Vo,R1,R2)
    return np.sqrt(result)
def Omega_curve(R,Vo,R1,R2):
    result = np.piecewise(R,[R<1e-3,R>=1e-3],[lambda R:Vo/R1,
             lambda R:Vo*np.arctan(R/R1)*np.exp(-R/R2)/R])
    return result
def analytic_model(radius,vel_cir,rbin):
    Redges=np.arange(0*pc,35000*pc,rbin*pc)
    Rcen=0.5*(Redges[1:]+Redges[0:-1])
    N=len(Rcen)
    V_16 = np.zeros(N)
    V_50 = np.zeros(N)
    V_84 = np.zeros(N)

    for k in range(N):
        ring=(Redges[k]<radius)&(Redges[k+1]>radius)
        if len(ring[ring])<4:
            V_16[k]  = np.nan
            V_50[k]  = np.nan
            V_84[k]  = np.nan
        else:
            vc_ring=vel_cir[ring]
            V_16[k], V_50[k], V_84[k] = np.percentile(vc_ring,[16,50,84])

    Rcen=add_extremes(Rcen)
    V_16=add_extremes(V_16)
    V_50=add_extremes(V_50)
    V_84=add_extremes(V_84)

    radius  =   Rcen[Rcen<20000]
    V_16    =   V_16[Rcen<20000]
    V_50    =   V_50[Rcen<20000]
    V_84    =   V_84[Rcen<20000]

    V_err=0.5*(V_84-V_16)
    V_err[0]=V_err[1:].min()

    p_0, cov_0 = curve_fit(V_curve,radius,V_50,sigma=V_err,p0=[V_50.max(),np.mean(radius)])
    p_0, cov_0 = curve_fit(Velocity_curve,radius,V_50,sigma=V_err,p0=[p_0[0],p_0[1],p_0[1]])
    return p_0
def parameters_rotation_curve(directory,sim):
    ds = yt.load(directory+'/'+sim+'/G-'+sim[-4:])
    Disk = ds.disk('c', [0., 0., 1.],(40, 'kpc'), (1, 'kpc'))

    VC  = Disk['vc'].in_units("pc/Myr")
    R   = Disk['Disk_Radius'].in_units("pc")
    par = analytic_model(R,VC,300)
    np.save('Tables/'+sim+'_rotation',par)
def turb_2D(directory,sim):
    ds = yt.load(directory+'/'+sim+'/G-'+sim[-4:])
    L=4e4
    L=L*kpc/1000
    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids
    dd=ds.all_data()
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 5.0e2"])
    proj = ds.proj(['velocity_x','velocity_y'], 2,data_source=disk_dd,weight_field='density')
    width = (float(L), 'kpc')

    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')
    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    ivx=frb['velocity_x'].in_units('km/s')
    ivy=frb['velocity_y'].in_units('km/s')

    R_map=radial_map_N(NN)*L/NN
    par=np.load('Tables/'+sim+'_rotation.npy')
    vc_map=Velocity_curve(R_map.in_units('pc'),*par)
    vc_map=vc_map*pc/Myr
    vc_map.convert_to_units('km/s')

    #vc_map=f_vc(R_map.in_units('pc'))
    X,Y=XY_map_N(NN)*L/NN
    avx=-Y*vc_map/R_map
    avy=X*vc_map/R_map
    avx[np.isnan(avx)]=0
    avy[np.isnan(avy)]=0

    dvx=ivx-avx
    dvy=ivy-avy

    redges=np.linspace(0,15000,120)
    rmin=redges[:-1]
    rmax=redges[1:]

    #rmin=np.array([0   ,1500,3000,4500,6000,7500 ,9000 ,10500,12000])
    #rmax=np.array([3000,4500,6000,7500,9000,10500,12000,13500,15000])
    Ri=0.5*(rmin+rmax)
    N=len(Ri)
    Dvt=np.zeros(N)

    for k in range(N):
        ring=(rmin[k]<R_map.in_units('pc'))&(rmax[k]>R_map.in_units('pc'))
        if len(ring[ring])<4:

            Dvt[k]=np.nan
            Sv[k]=np.nan
            Ss[k]=np.nan
            Sw[k]=np.nan
        else:
            aux=dvx[ring]
            p16,p84=np.percentile(aux,[16,84])
            Dvx=0.5*(p84-p16)
            Dvx=np.std(aux)
            del aux
            aux=dvy[ring]
            p16,p84=np.percentile(aux,[16,84])
            Dvy=0.5*(p84-p16)
            Dvy=np.std(aux)
            del aux
            Dvt[k]=np.sqrt(Dvx**2+Dvy**2)
    Dvt=Smooth(Ri,Dvt,20,1500)


    tab=Table()
    tab['Radius']=Column(Ri)
    tab['dv']=Column(Dvt)
    tab.write('Tables/'+sim+'_2D_turb',format='hdf5',path='data',overwrite=True)
    return 0
def velocity_dispersion(directory,sim):
    ds = yt.load(directory+'/'+sim+'/G-'+sim[-4:])
    Disk = ds.disk('c', [0., 0., 1.],(25, 'kpc'), (1, 'kpc'))
    L=30*kpc
    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    RM=DX
    DR=10*DX/grids
    dr=DX/grids

    radius=Disk['Disk_Radius'].in_units("pc")
    rho=Disk['density'].in_units("Msun/pc**3")
    vz=Disk['vertical_velocity'].in_units('km/s')
    cs=Disk['sound_speed'].in_units('km/s')
    angle=Disk['Disk_Angle']
    temperature = Disk['temperature'].in_units('K')

    filter = temperature < 1.0e4

    radius = radius[filter]
    rho    = rho[filter]
    vz     = vz[filter]
    cs     = cs[filter]
    angle  = angle[filter]

    Redges=np.arange(0*pc,20000*pc,DR)
    Rcen=0.5*(Redges[1:]+Redges[0:-1])
    N=len(Rcen)

    mass_bin       = np.zeros(N)
    cs_bin         = np.zeros(N)
    vz_bin         = np.zeros(N)
    vz_16          = np.zeros(N)
    vz_84          = np.zeros(N)

    for k in range(N):
        ring=(Redges[k]<radius)&(Redges[k+1]>radius)

        if len(ring[ring])<4:
            mass_bin[k]       = np.nan
            cs_bin[k]         = np.nan
            vz_bin[k]         = np.nan
            sz_bin[k]         = np.nan

        else:

            weights   = rho[ring]                  # density weights
            vz_ring   = vz[ring]
            #dvz_ring  = dvz[ring]

            ##### sound speed #####

            cs_bin[k]=np.average(1.0/cs[ring]**2,weights=weights)
            cs_bin[k]=cs_bin[k]**(-0.5)

            Nring=len(weights)
            Nang=int(Nring/200)

            Aedges      = np.linspace(-np.pi,np.pi,Nang)
            Acen        = 0.5*(Aedges[1:]+Aedges[0:-1])

            ##### zeta velocity #####

            vmin,vmax = np.percentile(vz_ring,[2,98])
            sample    = (vz_ring>vmin)&(vz_ring<vmax)

            vz_bin[k]=np.average(vz_ring[sample],weights=weights[sample])
            vz_bin[k]=np.average((vz_bin[k]-np.array(vz_ring[sample]))**2,weights=weights[sample])
            vz_bin[k]=vz_bin[k]**(0.5)

            aring=angle[ring]

            h_aux=[]

            for jk in range(Nang-1):
                section = (Aedges[jk]<aring)&(Aedges[jk+1]>aring)
                if len(section[section])>20:
                    try:
                        zangle  = vz_ring[section]
                        wangle  = weights[section]

                        dummy   = np.average(zangle,weights=wangle)
                        dummy   = np.average((dummy-zangle)**2,weights=wangle)
                        dummy   = np.sqrt(dummy)
                        h_aux.append(dummy)
                    except:
                        pass

            h_aux=np.array(h_aux)
            vz_bin[k]=np.median(h_aux)
            vz_16[k],vz_84[k]=np.percentile(h_aux,[16,84])

            del h_aux

    incorrect=np.isnan(cs_bin)
    cs_bin[incorrect]           = 0.0
    vz_bin[incorrect]           = 0.0
    vz_bin           = Smooth(Rcen,vz_bin ,10,500)
    cs_bin           = Smooth(Rcen,cs_bin ,10,500)

    tabla=Table()
    tabla['radius']	 = Column(np.array(Rcen))
    tabla['vz']      = Column(np.array(vz_bin))
    tabla['vz_16']   = Column(np.array(vz_16))
    tabla['vz_84']   = Column(np.array(vz_84))
    tabla['cs']      = Column(np.array(cs_bin))

    tabla.write('Tables/'+sim+'_dispersion',path='data',format='hdf5',overwrite=True)
    return 0
def stability_velocity(directory,sim):
    ds = yt.load(directory+'/'+sim+'/G-'+sim[-4:])
    Disk = ds.disk('c', [0., 0., 1.],(25, 'kpc'), (1, 'kpc'))
    L=30*kpc
    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    RM=DX
    DR=10*DX/grids
    dr=DX/grids

    radius=Disk['Disk_Radius'].in_units("pc")
    mass=Disk['cell_mass'].in_units("Msun")
    rho=Disk['density'].in_units("Msun/pc**3")
    cs=Disk['sound_speed'].in_units('km/s')
    angle=Disk['Disk_Angle']

    Redges=np.arange(0*pc,20000*pc,DR)
    Rcen=0.5*(Redges[1:]+Redges[0:-1])
    N=len(Rcen)

    mass_bin       = np.zeros(N)
    cs_bin         = np.zeros(N)
    sigma_50       = np.zeros(N)
    sigma_16       = np.zeros(N)
    sigma_84       = np.zeros(N)

    for k in range(N):
        ring=(Redges[k]<radius)&(Redges[k+1]>radius)

        if len(ring[ring])<4:
            mass_bin[k]       = np.nan
            cs_bin[k]         = np.nan
            sigma_50[k]       = np.nan
            sigma_16[k]       = np.nan
            sigma_84[k]       = np.nan

        else:
            weights   = rho[ring]                  # density weights
            mass_bin[k] = mass[ring].sum()    # mass in radial bin
            mring     = mass[ring]

            Nring=len(weights)
            Nang=int(Nring/200)

            Aedges      = np.linspace(-np.pi,np.pi,Nang)
            Acen        = 0.5*(Aedges[1:]+Aedges[0:-1])

            ##### sound speed #####

            cs_bin[k]=np.average(1.0/cs[ring]**2,weights=weights)
            cs_bin[k]=cs_bin[k]**(-0.5)

            #######################

            Aedges      = np.linspace(-np.pi,np.pi,33)
            Acen        = 0.5*(Aedges[1:]+Aedges[0:-1])

            m_aux=[]
            aring=angle[ring]
            for jk in range(32):
                section = (Aedges[jk]<aring)&(Aedges[jk+1]>aring)
                if len(section[section])>1:
                    mangle  = mring[section].sum()
                    mangle  /= np.pi*(Redges[k+1]**2-Redges[k]**2)/32
                    m_aux.append(mangle)
            m_aux       = np.array(m_aux)
            sigma_50[k] = np.median(m_aux)
            sigma_16[k],sigma_84[k] = np.percentile(m_aux,[16,84])

    sigma_gas_bin=mass_bin/(2*np.pi*Rcen*DR)
    incorrect=np.isnan(sigma_gas_bin)

    sigma_gas_bin[incorrect]    = 0.0
    sigma_50[incorrect]         = 0.0
    sigma_16[incorrect]         = 0.0
    sigma_84[incorrect]         = 0.0
    cs_bin[incorrect]           = 0.0

    ### Smoothing fields ###

    sigma_gas_bin    = Smooth(Rcen,sigma_gas_bin ,N,DR)
    sigma_50         = Smooth(Rcen,sigma_50 ,N,DR)
    sigma_16         = Smooth(Rcen,sigma_16 ,N,DR)
    sigma_84         = Smooth(Rcen,sigma_84 ,N,DR)
    cs_bin           = Smooth(Rcen,cs_bin ,N,DR)

    par = np.load('Tables/'+sim+'_rotation.npy')

    kappa_bin        = Kappa_curve(Rcen,*par)*km/second/pc
    kappa_bin.convert_to_units('1/Myr')
    sigma_gas_bin    = sigma_gas_bin*Msun/pc**2
    sigma_50         = sigma_50*Msun/pc**2
    sigma_16         = sigma_16*Msun/pc**2
    sigma_84         = sigma_84*Msun/pc**2
    cs_bin           = cs_bin*km/second

    Vmean   = np.pi*G*sigma_gas_bin/kappa_bin
    Vmedian = np.pi*G*sigma_50/kappa_bin
    V84     = np.pi*G*sigma_84/kappa_bin
    V16     = np.pi*G*sigma_16/kappa_bin
    Cmean   = np.sqrt(Vmean**2-cs_bin**2)
    Cmedian = np.sqrt(Vmedian**2-cs_bin**2)

    Vmean.convert_to_units('km/s')
    Vmedian.convert_to_units('km/s')
    V84.convert_to_units('km/s')
    V16.convert_to_units('km/s')
    Cmean.convert_to_units('km/s')
    Cmedian.convert_to_units('km/s')

    tabla=Table()
    tabla['radius']	    = Column(np.array(Rcen))
    tabla['Vmean']      = Column(np.array(Vmean))
    tabla['Vmedian']    = Column(np.array(Vmedian))
    tabla['V84']        = Column(np.array(V84))
    tabla['V16']        = Column(np.array(V16))
    tabla['Cmean']      = Column(np.array(Cmean))
    tabla['Cmedian']    = Column(np.array(Cmedian))

    tabla.write('Tables/'+sim+'_fiducial_velocity',path='data',format='hdf5',overwrite=True)
    return 0

cmdarg=sys.argv

sim=cmdarg[-1]
directory=cmdarg[-2]

#parameters_rotation_curve(directory,sim)
#turb_2D(directory,sim)
#stability_velocity(directory,sim)
velocity_dispersion(directory,sim)
