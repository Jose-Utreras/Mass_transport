import sys
import yt
from yt.funcs import mylog
mylog.setLevel(50)
import numpy as np
from yt import derived_field
import matplotlib.pyplot as plt
from yt.units import yr, Myr, pc,gram,second
from scipy.interpolate import interp1d
from yt.data_objects.particle_filters import add_particle_filter
from yt.fields.derived_field import \
    ValidateGridType, \
    ValidateParameter, \
    ValidateSpatial, \
    NeedsParameter

from astropy.table import Table , Column

def _Disk_H(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return np.abs(z)
yt.add_field("Disk_H",
             function=_Disk_H,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])

def _radial_velocity(field,data):
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
        return xv*x_hat+yv*y_hat
yt.add_field("radial_velocity", function=_radial_velocity,
    take_log=False, units=r"km/s",validators=[ValidateParameter('bulk_velocity')])

def _Disk_Radius(field, data):
    center = data.get_field_parameter('center')
    x = data["x"] - center[0]
    y = data["y"] - center[1]
    z = data["z"] - center[2]
    r = np.sqrt(x*x+y*y)
    return r
yt.add_field("Disk_Radius",
             function=_Disk_Radius,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])

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

cmdarg=sys.argv
name=cmdarg[-1]
ds=yt.load('../Sims/'+name+'/G-'+name[-4:])

##########  Creating disk and surface #############
"""
Is faster to create a new disk in each step
"""

"""
##########################################
### This is the surface flux on cylynders
### It also considers the flux on the z-xaxis
### A possible solution is replacing velocity_z by a zero field

Radius = np.linspace(0,16e3,40)[1:]
Mdot = np.zeros_like(Radius)

for i,r in enumerate(Radius):
    print(i)
    Disk = ds.disk('c', [0,0,1],(r,'pc'),(2e3,'pc'))
    surf = ds.surface(Disk,'Disk_Radius',r)
    md   = surf.calculate_flux(
        "velocity_x", "velocity_y", "velocity_z", "density")
    Mdot[i] = (float(md)*gram/second).in_units('Msun/yr')
"""



"""
This method projects the velocity and mass fields
and computes the flux of mass across the boundary
by finding the cells that intersect the circumference
"""

dd=ds.all_data()
disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 2.0e3"])
proj = ds.proj('radial_velocity', 2,data_source=disk_dd,weight_field='density')

grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]
DX=ds.arr(1, 'code_length')
DX.convert_to_units('pc')
DX/=grids
L    = 4e4*pc
NN   = int(np.round(L/DX))
L=NN*DX
width = (float(L), 'pc')
res = [NN, NN]
frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])
vr=frb['radial_velocity'].in_units('pc/Myr')

suma = ds.proj('density', 2,data_source=disk_dd)
frbs = suma.to_frb(width, res, center=[0.5,0.5,0.5])
sigma=frbs['density'].in_units('Msun/pc**2')

R_map=radial_map_N(NN)*L/NN
x,y = XY_map_N(NN)*L/NN

Radius = np.linspace(0,16e3,65)[1:]*pc
Inflow = np.zeros_like(Radius)

for ijk,radius in enumerate(Radius):
    ro=round(float(radius*NN/L))
    i0=int(float(radius*NN/L)+NN/2)
    j0=round(NN/2)

    i_list=[]
    j_list=[]

    while True:
        try:
            if (i0==i_list[0])and(j0==j_list[0]):
                break
        except:
            pass

        try:
            if (i0==i_list[-2])and(j0==j_list[-2]):
                print('ERROR')
                break
        except:
            pass
        i_list.append(i0)
        j_list.append(j0)
        i1=i0-int(np.sign(x[i0][j0]))
        if i1==i0:
            i1=i0-1
        j1=j0
        r1=np.sqrt((x[i1][j1]-np.sign(x[i1][j1])*DX/2)**2
            +(y[i1][j1]-np.sign(y[i1][j1])*DX/2)**2)
        r2=np.sqrt((x[i1][j1]+np.sign(x[i1][j1])*DX/2)**2
            +(y[i1][j1]+np.sign(y[i1][j1])*DX/2)**2)
        if (r1-radius)*(r2-radius)<0.0:
            i0=i1
            j0=j1
        else:
            i1=i0
            j1=j0+int(np.sign(y[i0][j0]))

            r1=np.sqrt((x[i1][j1]-np.sign(x[i1][j1])*DX/2)**2
                +(y[i1][j1]-np.sign(y[i1][j1])*DX/2)**2)
            r2=np.sqrt((x[i1][j1]+np.sign(x[i1][j1])*DX/2)**2
                +(y[i1][j1]+np.sign(y[i1][j1])*DX/2)**2)
            if (r1-radius)*(r2-radius)<0.0:
                i0=i1
                j0=j1
            else:
                i1=i0-int(np.sign(x[i0][j0]))
                j1=j0+int(np.sign(y[i0][j0]))

                r1=np.sqrt((x[i1][j1]-np.sign(x[i1][j1])*DX/2)**2
                    +(y[i1][j1]-np.sign(y[i1][j1])*DX/2)**2)
                r2=np.sqrt((x[i1][j1]+np.sign(x[i1][j1])*DX/2)**2
                    +(y[i1][j1]+np.sign(y[i1][j1])*DX/2)**2)
                if (r1-radius)*(r2-radius)<0.0:
                    i0=i1
                    j0=j1

    mask=np.zeros((NN,NN))
    for i,j in zip(i_list,j_list):
        x1 =x[int(i)][int(j)]
        x2 =x1+np.sign(x1+0.01*pc)*0.5*DX
        x1 =x1-np.sign(x1+0.01*pc)*0.5*DX

        y1 =y[int(i)][int(j)]
        y2 =y1+np.sign(y1+0.01*pc)*0.5*DX
        y1 =y1-np.sign(y1+0.01*pc)*0.5*DX

        xcoord=[]
        ycoord=[]

        ya=np.sqrt(radius**2-x1**2)*np.sign(y1)
        yb=np.sqrt(radius**2-x2**2)*np.sign(y1)

        if(ya>min(y1,y2))and(ya<max(y1,y2)):
            xcoord.append(x1)
            ycoord.append(ya)
        if(yb>min(y1,y2))and(yb<max(y1,y2)):
            xcoord.append(x2)
            ycoord.append(yb)

        xa=np.sqrt(radius**2-y1**2)*np.sign(x1)
        xb=np.sqrt(radius**2-y2**2)*np.sign(x1)

        if(xa>min(x1,x2))and(xa<max(x1,x2)):
            xcoord.append(xa)
            ycoord.append(y1)
        if(xb>min(x1,x2))and(xb<max(x1,x2)):
            xcoord.append(xb)
            ycoord.append(y2)

        dL=np.sqrt((xcoord[0]-xcoord[1])**2+(ycoord[0]-ycoord[1])**2)
        mask[int(i)][int(j)]=dL

    flow=mask*pc*vr*sigma
    flow.convert_to_units('Msun/yr')
    Inflow[ijk]=flow.sum()

tabla = Table()
tabla['Radius']=Column(Radius,unit='pc')
tabla['Mdot']=Column(Inflow,unit='msun/yr')
tabla.write('Tables/'+name+'_mdot',path='data',format='hdf5',serialize_meta=True,overwrite=True)
