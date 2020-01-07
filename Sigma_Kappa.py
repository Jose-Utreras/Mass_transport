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
def Velocity_curve(R,Vo,R1,R2):
    return Vo*np.arctan(R/R1)*np.exp(-R/R2)
def add_extremes(X):
    X=np.insert(X,0,0)
    X=np.insert(X,len(X),2*X[-1]-X[-2])
    return X
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
def V_curve(R,Vo,R1):
    return Vo*np.arctan(R/R1)
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
def sigma_kappa(directory,sim):
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
    angle=Disk['Disk_Angle']

    Redges=np.arange(0*pc,20000*pc,DR)
    Rcen=0.5*(Redges[1:]+Redges[0:-1])
    N=len(Rcen)

    mass_bin       = np.zeros(N)
    sigma_50       = np.zeros(N)
    sigma_16       = np.zeros(N)
    sigma_84       = np.zeros(N)

    for k in range(N):
        ring=(Redges[k]<radius)&(Redges[k+1]>radius)

        if len(ring[ring])<4:
            mass_bin[k]       = np.nan
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

    ### Smoothing fields ###

    sigma_gas_bin    = Smooth(Rcen,sigma_gas_bin ,N,DR)
    sigma_50         = Smooth(Rcen,sigma_50 ,N,DR)
    sigma_16         = Smooth(Rcen,sigma_16 ,N,DR)
    sigma_84         = Smooth(Rcen,sigma_84 ,N,DR)

    par = np.load('Tables/'+sim+'_rotation.npy')

    kappa_bin        = Kappa_curve(Rcen,*par)*km/second/pc
    kappa_bin.convert_to_units('1/Myr')
    sigma_gas_bin    = sigma_gas_bin*Msun/pc**2
    sigma_50         = sigma_50*Msun/pc**2
    sigma_16         = sigma_16*Msun/pc**2
    sigma_84         = sigma_84*Msun/pc**2


    tabla=Table()
    tabla['radius']	    = Column(np.array(Rcen))
    tabla['kappa']      = Column(np.array(kappa_bin))
    tabla['sigma_gas']  = Column(np.array(sigma_gas_bin))
    tabla['sigma_16']   = Column(np.array(sigma_16))
    tabla['sigma_50']   = Column(np.array(sigma_50))
    tabla['sigma_84']   = Column(np.array(sigma_84))


    tabla.write('Tables/'+sim+'_sigma_kappa',path='data',format='hdf5',overwrite=True)
    return 0

cmdarg=sys.argv

sim=cmdarg[-1]
directory=cmdarg[-2]

parameters_rotation_curve(directory,sim)
sigma_kappa(directory,sim)
