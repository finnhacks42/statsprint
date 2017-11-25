
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from matplotlib import cm
from stl import mesh as s_mesh
from numpy.linalg import inv

def write_stl(trimesh,filepath):
    """
    Writes a matplotlib Triangulation to stl.
    Assumes the trimesh object has an attribute z that specifies the height
    of each vertex. 
    """
    data = np.zeros(len(trimesh.triangles), dtype=s_mesh.Mesh.dtype)
    stl_mesh = s_mesh.Mesh(data, remove_empty_areas=False)
    stl_mesh.x[:] = trimesh.x[trimesh.triangles]
    stl_mesh.y[:] = trimesh.y[trimesh.triangles]
    stl_mesh.z[:] = trimesh.z[trimesh.triangles]
    stl_mesh.save(filepath)
    
def plot_surface(trimesh):
    """ 
    Plots a surface over a matplotlib Triangulation. 
    Assumes that the height of the surface has been stored in an attribute 'z'
    on the trimesh object.
    """
    fig,ax = plt.subplots()
    ax = fig.gca(projection='3d')
    polycollection = ax.plot_trisurf(trimesh, trimesh.z, cmap=cm.jet, linewidth=0.1)
    return fig,ax,polycollection

def inner(ids,edges,outer_ids=None):
    """Takes a closed loop of verticies and finds the next loop inside"""
    # vertices connected to current_ids
    connected = np.unique(edges[np.isin(edges[:,0],ids)|np.isin(edges[:,1],ids)].ravel()) 
    if outer_ids is None:
        outer_ids = ids.copy()
    else:
        outer_ids = np.concatenate([ids,outer_ids])
    inner_ids = connected[np.isin(connected,outer_ids,invert=True)]
    return inner_ids,outer_ids 

class MinMaxScaler1D(object):
    """ Transforms 1d arrays to a specified range. """
    def __init__(self,a = 0,b = 1):
        self.a = a
        self.b = b
    
    def fit(self,x):
        self.mult = (self.b - self.a)/(x.max() - x.min())
        self.offset =  self.a - self.mult*x.min()
        return self
        
    def transform(self,x):
        return (self.mult*x + self.offset)
    
    def fit_transform(self,x):
        return self.fit(x).transform(x)

class Grid(object): 
    
    def plot_mesh(self, ax = None):
        m = self.mesh(fold_under_base=False)
        if ax is None:
            fig,ax = plt.subplots(figsize=(4,4))
        ax.triplot(m,alpha=0.5)
        ax.scatter(m.x[self._outer_base],m.y[self._outer_base],color="red")
        ax.scatter(m.x[self._inner_base],m.y[self._inner_base],color="green")
        return ax

class CircleGrid(Grid):
    """ Constructs a trimesh surface over circular domain r <= 1, 0 <= theta < 2pi """ 
    def __init__(self,div_r=200,div_theta=300):
        r_points = np.concatenate([np.linspace(0,1,div_r),[2,3]])
        n_theta =np.maximum(1,(div_theta*r_points).astype(int)) # number of theta points for each value of the radius
        n_theta[-2:]=np.array([n_theta[-3],8])
        self._r = np.repeat(r_points,n_theta)
        self._t = np.hstack([np.linspace(0,2*np.pi,num=n,endpoint=False) for n in n_theta])
        self._x,self._y = self._r*np.cos(self._t),self._r*np.sin(self._t)
        self._surface_points = (self._r <= 1)
        self._outer_base,self._inner_base = (self._r == 2),(self._r == 3)
        
    def mesh(self, fold_under_base = True):
        tri = mtri.Triangulation(self._x.copy(),self._y.copy())
        if fold_under_base:
            tri.x[self._outer_base] = np.cos(self._t[self._outer_base])
            tri.y[self._outer_base] = np.sin(self._t[self._outer_base])
            tri.x[self._inner_base] = 0
            tri.y[self._inner_base] = 0
        return tri
    
    def surface(self,func,coords = "xy",scale=(40,40,40),base=3,plot=True):
        if coords == "xy":
            X = np.vstack([self._x[self._surface_points],self._y[self._surface_points]]).T 
        elif coords == "polar":
            X = np.vstack([self._r[self._surface_points],self._t[self._surface_points]]).T
        else:
            raise ValueError("Invalid coords: {}, must be one of ['xy', 'polar']".format(coords))
        tri = self.mesh()
        z = np.zeros(len(tri.x))
        z[self._surface_points] = func(X)
        scale_x = MinMaxScaler1D(0,scale[0])
        scale_y = MinMaxScaler1D(0,scale[1])
        scale_z = MinMaxScaler1D(base,scale[2]+base)
        z = scale_z.fit(z[self._surface_points]).transform(z)
        z[~self._surface_points] = 0
        tri.x = scale_x.fit_transform(tri.x)
        tri.y = scale_y.fit_transform(tri.y)
        tri.z = z
        return tri


class BarycentricTriangleGrid(Grid):
    """Constructs a Barycentric grid."""
    def __init__(self,subdiv=4):
        self.corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
        self.middle = np.array([.5,.5*(np.sqrt(.75))])
        self.triangle = mtri.Triangulation(self.corners[:,0],self.corners[:,1])
        self.Tinv = inv(np.asarray([self.corners[0,:]-self.corners[2,:],self.corners[1,:]-self.corners[2,:]]).T)
        self.r3 = self.corners[2,:]
        self.ext = self.corners 
        self.subdiv=subdiv
             
    def xy2bc(self,xy):
        """
        Converts cartesian coordinates to barycentric 
        (see https://en.wikipedia.org/wiki/Barycentric_coordinate_system)
        xy - an np.array of x,y cordinates with shape (n,2) where n is the number of points to convert
        returns bc_coords - an np.array of barycentric coordinates, with shape (n,3)
        """
        v = (xy - self.r3).T
        first2_componets = np.maximum(self.Tinv.dot(v),1e-20)
        third_component = np.maximum(1.0-first2_componets.sum(axis=0),1e-20)
        bc_coords = np.vstack([first2_componets,third_component]).T
        return bc_coords
        
    def mesh(self, fold_under_base = True):
        mesh = mtri.UniformTriRefiner(self.triangle).refine_triangulation(subdiv=self.subdiv)
        verticies, counts = np.unique(mesh.edges,return_counts=True)
        l1 = verticies[np.where(counts <=4)] # the outer layer of verticies
        l2, outer = inner(l1,mesh.edges) # one layer in from the edge
        l3, outer = inner(l2,mesh.edges,outer) # two layers in from the edge
        scale_x = MinMaxScaler1D()
        scale_x.fit(mesh.x[l3])
        scale_y = MinMaxScaler1D(0,0.75**0.5)
        scale_y.fit(mesh.y[l3])
        mesh.x = scale_x.transform(mesh.x)
        mesh.y = scale_y.transform(mesh.y)
        
        if fold_under_base:
            mesh.x[l2] = scale_x.fit(mesh.x[l2]).transform(mesh.x[l2])
            mesh.y[l2] = scale_y.fit(mesh.y[l2]).transform(mesh.y[l2])
            mesh.x[l1] = .5
            mesh.y[l1] = .5 
            
        self._surface_verts = np.isin(verticies,outer,invert=True)
        self._outer_base = l2
        self._inner_base = l1
        return mesh

    def surface(self,func,scale=(40,40,40),base=3):
        mesh = self.mesh()
        z = np.zeros(len(mesh.x))
        xy = np.vstack([mesh.x[self._surface_verts],mesh.y[self._surface_verts]]).T
        bc = self.xy2bc(xy).T
        z[self._surface_verts] = func(bc) # want the surface verticies to be between a,b
        scale_x = MinMaxScaler1D(0,scale[0])
        scale_y = MinMaxScaler1D(0,scale[1])
        scale_z = MinMaxScaler1D(base,scale[2]+base)
        mesh.x = scale_x.fit(mesh.x).transform(mesh.x)
        mesh.y = scale_y.fit(mesh.y).transform(mesh.y)
        z = scale_z.fit(z[self._surface_verts]).transform(z)
        z[~self._surface_verts] = 0
        mesh.z = z
        return mesh



        



