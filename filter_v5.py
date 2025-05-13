import math
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
import numpy as np
import os
import seaborn as sns
import sys

############################################
##################Class#####################
############################################
class Parameters:
    def __init__(self):
        self.nt                     = 50002
        self.dt                     = 0.01
        self.nx                     = 32#number of points including the edges
        #self.nx                     = 128#number of points including the edges
        self.number_interior_points = self.nx - 2
        self.domain                 = [0, 1.0]
        #self.domain                 = [0, 10.0]
        self.center                 = (self.domain[1] - self.domain[0])/2
        self.length                 = self.domain[1] - self.domain[0]

    @property
    def dx(self):
        return (self.domain[1] - self.domain[0])/(self.nx - 1)

class BioParameters:
    def __init__(self):
        self.diffusion_coefficient1 = 0.1
        self.diffusion_coefficient2 = 0.1
        self.growth_rate_interior1  = 0.5
        self.growth_rate_interior2  = 0.5
        self.growth_rate_boundary1  = 1.0
        self.growth_rate_boundary2  = 1.0
        self.inflow_rate1           = 1.0
        self.inflow_rate2           = 1.0
        self.feeding                = 1.0
        self.speed_ratio            = 0.1
        self.filtering_scale        = 2.0
        #self.filtering_scale        = 0.0#for no clogging case
        #self.filtering_scale        = 10e10#for clogging case

class MyLatticeWithEndpoints:
    def __init__(self, interior=[0]*16, left=0.0, right=0.0):
        self.interior = [iinterior for iinterior in interior]
        self.left     = left
        self.right    = right

    @property
    def total_length(self):
        return len(self.interior) + 2

    @property
    def interior_length(self):
        return len(self.interior)

    @property
    def list_of_lattice_with_endpoints(self):
        v = []
        v.append(self.left)
        for iinterior in self.interior:
            v.append(iinterior)
        v.append(self.right)
        return v
    
    def __add__(self, other):
        return MyLatticeWithEndpoints(
            interior=[iself + iother for iself, iother in zip(self.interior, other.interior)],
            left=self.left+other.left,
            right=self.right+other.right)

    def __sub__(self, other):
        return MyLatticeWithEndpoints(
            interior=[iself - iother for iself, iother in zip(self.interior, other.interior)],
            left=self.left-other.left,
            right=self.right-other.right)

    def __mul__(self, scalar):
        return MyLatticeWithEndpoints(
            interior=[scalar*iself for iself in self.interior],
            left=scalar*self.left,
            right=scalar*self.right)

    def __rmul__(self, scalar):
        return self*scalar
    
    def plugin_vector(self, vec):
        if (len(vec) != (len(self.interior)+2)):
            print("In plugin_vector of Class:MyLatticeWithEndpoints, the length of the vector is invalid")
            sys.exit()
        for i, ivec in enumerate(vec):
            self.interior[i] = ivec

    def plugin_endpoints(self, left, right):
        self.left  = left
        self.right = right

    def copy(self):
        v_new_interior = [self.interior[i] for i in range(len(self.interior))]
        l = self.left
        r = self.right
        return MyLatticeWithEndpoints(
            interior=v_new_interior,
            left=l,
            right=r)

    def show(self, c=''):
        print(f"{c} interior = {self.interior}")
        print(f"{c} left    = {self.left}")
        print(f"{c} right    = {self.right}")

class MyData:
    def __init__(self):
        pass

    #-----------------------initial data-----------------------#
    def get_two_initial_data_interior(self, domain, nx, dx, predator_capacity_ratio_interior):
        u0_list = self.get_function(
            level=2.0,
            left=domain[0],
            right=domain[-1],
            nx=nx,
            dx=dx,
            key='cos')
        u0 = MyLatticeWithEndpoints(
            interior=u0_list[1: -1],
            left=u0_list[0],
            right=u0_list[-1])
        v0_list = self.get_function(
            level=1.0*predator_capacity_ratio_interior,
            left=domain[0],
            right=domain[-1],
            nx=nx,
            dx=dx,
            key='cos')
        v0 = MyLatticeWithEndpoints(
            interior=v0_list[1: -1],
            left=v0_list[0],
            right=v0_list[-1])
        return u0, v0

    def get_two_initial_data_boundary(self, predator_capacity_ratio_boundary):
        return 0.0, predator_capacity_ratio_boundary*0.0

    def cos_function(self, level, left, right, nx, dx):
        level_top    = level
        level_bottom = 0.0
        cut_off      = 0.1#some small number
        center_point = (right - left)/2
        f = []
        for ix in range(nx):
            iy = (math.cos(math.pi*(dx*ix - center_point)))
            if (iy < cut_off):#if too small
                f.append(level_bottom*iy)
            else:
                f.append(level_top*iy)
        return f

    def get_function(self, level, left, right, nx, dx, key):
        if (key=='cos'):
            return self.cos_function(
                level=level,
                left=left,
                right=right,
                nx=nx,
                dx=dx)
        else:
            print("In function get_function in class MyData.\n"
            "Define initial data type.")
            sys.exit()

#    @classmethod
#    def save_fig_interior_as_png(cls, interiors):
#        try:
#            length_x = interiors[0].total_length
#            x = np.array(range(length_x))
#            graph_names = []
#            for i, iinitial_data in enumerate(interiors):
#                y = np.array(iinitial_data.list_of_lattice_with_endpoints)
#                plt.plot(x, y)
#                graph_names.append('u{}'.format(i))
#            plt.legend(graph_names)
#            plt.title('Plot of initial data')
#            plt.savefig('initial_data_interior.png')
#        except Exception as e:
#            print('Error at save_fig_interior in the class MyData')
#            print("Error: could not execute function with argument", interiors)
#            print("Error message:", str(e))
#            exit()

    #-----------------------forcing terms-----------------------#
    def feed_function(self, point_feeding, number_interior_points):
        constant_feeding = MyLatticeWithEndpoints(
            interior=[point_feeding]*number_interior_points,
            left=point_feeding,
            right=point_feeding)
        return constant_feeding

    def nonlinear_terms_interior(self, interiors, growth_rates):
        nonlinear_term_interior1 = []
        nonlinear_term_interior2 = []
        for interior1, interior2 in zip(interiors[0].interior, interiors[1].interior):
            igrowth1\
                = - growth_rates[0]*interior1/(1 + interior1)\
                *interior2
            igrowth2\
                = (\
                    growth_rates[1]*interior1/(1 + interior1)\
                    - interior2\
                )*interior2
            nonlinear_term_interior1.append(igrowth1)
            nonlinear_term_interior2.append(igrowth2)
        nonlinear_term_left1\
            = - growth_rates[0]*interiors[0].left/(1 + interiors[0].left)\
            *interiors[1].left
        nonlinear_term_left2\
            = (\
                growth_rates[1]*interiors[0].left/(1 + interiors[0].left)\
                - interiors[1].left\
            )*interiors[1].left
        nonlinear_term_right1\
            = - growth_rates[0]*interiors[0].right/(1 + interiors[0].right)\
            *interiors[1].right
        nonlinear_term_right2\
            = (\
                growth_rates[1]*interiors[0].right/(1 + interiors[0].right)\
                - interiors[1].right\
            )*interiors[1].right
        return MyLatticeWithEndpoints(
                    interior=nonlinear_term_interior1,
                    left=nonlinear_term_left1,
                    right=nonlinear_term_right1),\
               MyLatticeWithEndpoints(
                    interior=nonlinear_term_interior2,
                    left=nonlinear_term_left2,
                    right=nonlinear_term_right2)

    def force_interior(self, interiors, parameters, bio_parameters):
        f_feed = self.feed_function(
            point_feeding=bio_parameters.feeding,
            number_interior_points=parameters.number_interior_points)
        f1, f2 = self.nonlinear_terms_interior(
            interiors=interiors,
            growth_rates=[
                bio_parameters.growth_rate_interior1,
                bio_parameters.growth_rate_interior2])
        return f1 + f_feed, f2

    #@classmethod
    @staticmethod
    def absorption_rate_into_filter(clogging_substance, scale):
        return 1/(1 + scale*clogging_substance)

    #@classmethod
    @staticmethod
    def velocity(speed_ratio, clogging_substance, scale):
        return speed_ratio*MyData.absorption_rate_into_filter(clogging_substance, scale)

    def inflow_terms(self, boundary_traces_of_interior, inflow_rates, speed_ratio, clogging_substance, filtering_scale):
        c = MyData.velocity(
            speed_ratio=speed_ratio,
            clogging_substance=clogging_substance,
            scale=filtering_scale)
        r = MyData.absorption_rate_into_filter(
            clogging_substance=clogging_substance,
            scale=filtering_scale)
        f1 = inflow_rates[0]*c*r*boundary_traces_of_interior[0]
        f2 = inflow_rates[1]*c*r*boundary_traces_of_interior[1]
        return f1, f2

    def nonlinear_terms_boundary(self, boundaries, growth_rates):
        nonlinear_term_boundary1\
            = - growth_rates[0]*boundaries[0]/(1 + boundaries[0])\
            *boundaries[1]
        nonlinear_term_boundary2\
            = (\
                growth_rates[1]*boundaries[0]/(1 + boundaries[0])\
                - boundaries[1]\
            )*boundaries[1]
        return nonlinear_term_boundary1, nonlinear_term_boundary2

    def force_boundary(self, boundary_traces_of_interior, boundaries, bio_parameters):
        inflow1, inflow2 = self.inflow_terms(
            boundary_traces_of_interior=boundary_traces_of_interior,
            inflow_rates=[
                bio_parameters.inflow_rate1,
                bio_parameters.inflow_rate2],
            speed_ratio=bio_parameters.speed_ratio,
            clogging_substance=boundaries[0],
            filtering_scale=bio_parameters.filtering_scale)
        nonlinear_term1, nonlinear_term2 = self.nonlinear_terms_boundary(
            boundaries=boundaries,
            growth_rates=[
                bio_parameters.growth_rate_boundary1,
                bio_parameters.growth_rate_boundary2])
        return nonlinear_term1 + inflow1, nonlinear_term2 + inflow2 
    
    def get_forces(self, interiors, boundaries, parameters, bio_parameters):
        force_interior1, force_interior2 = self.force_interior(
            interiors=interiors,
            parameters=parameters,
            bio_parameters=bio_parameters)
        force_boundary1, force_boundary2 = self.force_boundary(
            boundary_traces_of_interior=[
                interiors[0].right,
                interiors[1].right],
            boundaries=boundaries,
            bio_parameters=bio_parameters)
        return force_interior1, force_interior2,\
               force_boundary1, force_boundary2
    
    @staticmethod
    def mkdir(directory_path, overwrite=False):
        if overwrite:
            if os.path.exists(directory_path):
                for root, dirs, files in os.walk(directory_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
        if os.path.exists(directory_path):
            pass
        else:
            os.makedirs(directory_path)

class FivefoldDiagonalMatrix:
    def __init__(self, length):
        # default = identity
        self.length     = length
        self.diagonal   = [1]*(self.length)
        self.diagonal_u = [0]*(self.length - 1)
        self.diagonal_l = [0]*(self.length - 1)
        self.edge_u     = 0
        self.edge_l     = 0

    def __add__(self, other):
        result = FivefoldDiagonalMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   + other.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [self.diagonal_u[i] + other.diagonal_u[i] for i in range(self.length - 1)]
        result.diagonal_l = [self.diagonal_l[i] + other.diagonal_l[i] for i in range(self.length - 1)]
        result.edge_u     = self.edge_u + other.edge_u
        result.edge_l     = self.edge_l + other.edge_l
        return result
         
    def __sub__(self, other):
        result = FivefoldDiagonalMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   - other.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [self.diagonal_u[i] - other.diagonal_u[i] for i in range(self.length - 1)]
        result.diagonal_l = [self.diagonal_l[i] - other.diagonal_l[i] for i in range(self.length - 1)]
        result.edge_u     = self.edge_u - other.edge_u
        result.edge_l     = self.edge_l - other.edge_l
        return result
 
    def __mul__(self, scalar):
        result = FivefoldDiagonalMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [scalar*self.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [scalar*self.diagonal_u[i] for i in range(self.length - 1)]
        result.diagonal_l = [scalar*self.diagonal_l[i] for i in range(self.length - 1)]
        result.edge_u     = scalar*self.edge_u
        result.edge_l     = scalar*self.edge_l
        return result
    
    def __rmul__(self, scalar):
        return self*scalar
    
    def apply_to_list(self, vec_list):
        vec_new = [0]*self.length
        for i in range(self.length):
            if (i == 0):
                vec_new[i] = self.diagonal[i]*vec_list[0] + self.diagonal_u[i]*vec_list[1] + self.edge_u*vec_list[self.length - 2]
            elif (i == (self.length - 1)):
                vec_new[i] = self.diagonal[i]*vec_list[i] + self.diagonal_l[i-1]*vec_list[i-1] + self.edge_l*vec_list[1]
            else:
                vec_new[i] = self.diagonal[i]*vec_list[i] + self.diagonal_l[i-1]*vec_list[i-1] + self.diagonal_u[i]*vec_list[i+1]
        return vec_new

    def apply(self, lattice_or_list):
        if isinstance(lattice_or_list, MyLatticeWithEndpoints):
            return MyLatticeWithEndpoints(
                interior=self.apply_to_list(lattice_or_list.interior),
                left=lattice_or_list.left,
                right=lattice_or_list.right)
        elif isinstance(lattice_or_list, list):
            return self.apply_to_list(lattice_or_list)
        else:
            raise ValueError("vec is Unknown object")

    def LU(self):
        L = LowerTriangularMatrix(self.length)
        U = UpperTriangularMatrix(self.length)
        for i in range(self.length - 1):
            if (i == 0):
                U.diagonal[i]   = self.diagonal[i]
                U.diagonal_u[i] = self.diagonal_u[i]
                U.vertical[i]   = self.edge_u
                L.diagonal_l[i] = self.diagonal_l[i]/U.diagonal[i]
                L.horizontal[i] = 0
            elif (i == 1):
                U.diagonal[i]   = self.diagonal[i] - L.diagonal_l[i-1]*U.diagonal_u[i-1]
                U.diagonal_u[i] = self.diagonal_u[i]
                U.vertical[i]   = - L.diagonal_l[i-1]*U.vertical[i-1]
                L.diagonal_l[i] = self.diagonal_l[i]/U.diagonal[i]
                L.horizontal[i] = self.edge_l/U.diagonal[i]
            elif (i == (self.length - 3)):
                U.diagonal[i]   = self.diagonal[i] - L.diagonal_l[i-1]*U.diagonal_u[i-1]
                U.diagonal_u[i] = self.diagonal_u[i] - L.diagonal_l[i-1]*U.vertical[i-1]
                L.diagonal_l[i] = self.diagonal_l[i]/U.diagonal[i]
                L.horizontal[i] = - L.horizontal[i-1]*U.diagonal_u[i-1]/U.diagonal[i]
            elif (i == (self.length - 2)):
                U.diagonal[i]   = self.diagonal[i] - L.diagonal_l[i-1]*U.diagonal_u[i-1]
                U.diagonal_u[i] = self.diagonal_u[i]
                s = 0
                for j in range(self.length - 2):
                    s = s + U.vertical[j]*L.horizontal[j]
                L.diagonal_l[i] = (self.diagonal_l[i] - L.horizontal[i-1]*U.diagonal_u[i-1] - s)/U.diagonal[i]
            else:
                U.diagonal[i]   = self.diagonal[i] - L.diagonal_l[i-1]*U.diagonal_u[i-1]
                U.diagonal_u[i] = self.diagonal_u[i]
                U.vertical[i]   = - L.diagonal_l[i-1]*U.vertical[i-1]
                L.diagonal_l[i] = self.diagonal_l[i]/U.diagonal[i]
                L.horizontal[i] = - L.horizontal[i-1]*U.diagonal_u[i-1]/U.diagonal[i]
            U.diagonal[self.length - 1] = self.diagonal[self.length - 1] - L.diagonal_l[self.length - 2]*U.diagonal_u[self.length - 2]
        return L, U
    
    def solve_for_list(self, vec_list):
        sol = [0]*self.length
        L, U = self.LU()
        #Lx = vec --> Uy = x --> sol = y
        x   = L.solve(vec_list)
        sol = U.solve(x)
        return sol

    def solve(self, vec):
        if isinstance(vec, MyLatticeWithEndpoints):
            return MyLatticeWithEndpoints(
                interior=self.solve_for_list(vec.interior),
                left=0.0,
                right=0.0)
        elif isinstance(vec, list):
            return self.solve_for_list(vec)
        else:
            raise ValueError("Unknown Object")

    def show(self):
        print(f'diagonal   = {self.diagonal}')
        print(f'diagonal_u = {self.diagonal_u}')
        print(f'diagonal_l = {self.diagonal_l}')
        print(f'edge_u     = {self.edge_u}')
        print(f'edge_l     = {self.edge_l}')

class UpperTriangularMatrix:
    def __init__(self, length):
        # default = identity
        self.length     = length
        self.diagonal   = [1]*(self.length)
        self.diagonal_u = [0]*(self.length - 1)
        self.vertical   = [0]*(self.length - 2)

    def __add__(self, other):
        result = UpperTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   + other.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [self.diagonal_u[i] + other.diagonal_u[i] for i in range(self.length - 1)]
        result.vertical   = [self.vertical[i]   + other.vertical[i]   for i in range(self.length - 2)]
        return result
         
    def __sub__(self, other):
        result = UpperTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   - other.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [self.diagonal_u[i] - other.diagonal_u[i] for i in range(self.length - 1)]
        result.vertical   = [self.vertical[i]   - other.vertical[i]   for i in range(self.length - 2)]
        return result
 
    def __mul__(self, scaler):
        result = UpperTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [scaler*self.diagonal[i]   for i in range(self.length)]
        result.diagonal_u = [scaler*self.diagonal_u[i] for i in range(self.length - 1)]
        result.vertical   = [scaler*self.vertical[i]   for i in range(self.length - 2)]
        return result

    def __rmul__(self, scalar):
        return self*scalar

    def apply(self, vec):
        vec_new = [0]*self.length
        for i in range(self.length):
            if (i == (self.length - 2)):
                vec_new[i] = self.diagonal[i]*vec[i] + self.diagonal_u[i]*vec[i+1]
            elif (i == (self.length - 1)):
                vec_new[i] = self.diagonal[i]*vec[i]
            else:
                vec_new[i] = self.diagonal[i]*vec[i] + self.diagonal_u[i]*vec[i+1] + self.vertical[i]*vec[self.length - 2]
        return vec_new

    def solve(self, vec):
        sol = [0]*self.length
        for i in reversed(range(self.length)):
            if (i == (self.length - 1)):
                sol[i] = vec[i]/self.diagonal[i]
            elif (i == (self.length - 2)):
                sol[i] = (vec[i] - self.diagonal_u[i]*sol[i+1])/self.diagonal[i]
            else:
                sol[i] = (vec[i] - self.diagonal_u[i]*sol[i+1] - self.vertical[i]*sol[self.length - 2])/self.diagonal[i]
        return sol

    def show(self):
        print(f'diagonal   = {self.diagonal}')
        print(f'diagonal_u = {self.diagonal_u}')
        print(f'vertical   = {self.vertical}')

class LowerTriangularMatrix:
    def __init__(self, length):
        # default = identity
        self.length     = length
        self.diagonal   = [1]*(self.length)
        self.diagonal_l = [0]*(self.length - 1)
        self.horizontal = [0]*(self.length - 2)

    def __add__(self, other):
        result = LowerTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   + other.diagonal[i]   for i in range(self.length)]
        result.diagonal_l = [self.diagonal_l[i] + other.diagonal_l[i] for i in range(self.length - 1)]
        result.horizontal = [self.horizontal[i] + other.horizontal[i] for i in range(self.length - 2)]
        return result
         
    def __sub__(self, other):
        result = LowerTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [self.diagonal[i]   - other.diagonal[i]   for i in range(self.length)]
        result.diagonal_l = [self.diagonal_l[i] - other.diagonal_l[i] for i in range(self.length - 1)]
        result.horizontal = [self.horizontal[i] - other.horizontal[i] for i in range(self.length - 2)]
        return result

    def __mul__(self, scalar):
        result = LowerTriangularMatrix(self.length)
        result.length     = self.length
        result.diagonal   = [scalar*self.diagonal[i]   for i in range(self.length)]
        result.diagonal_l = [scalar*self.diagonal_l[i] for i in range(self.length - 1)]
        result.horizontal = [scalar*self.horizontal[i] for i in range(self.length - 2)]
        return result
    
    def __rmul__(self, scalar):
        return self*scalar

    def apply(self, vec):
        vec_new = [0]*self.length
        for i in range(self.length):
            if (i == 0):
                vec_new[i] = self.diagonal[i]*vec[i]
            elif (i == (self.length - 1)):
                s = 0
                for j in range(self.length - 2):
                    s = s + self.horizontal[j]*vec[j]
                vec_new[i] = s + self.diagonal[i]*vec[i] + self.diagonal_l[i-1]*vec[i-1]
            else:
                vec_new[i] = self.diagonal[i]*vec[i] + self.diagonal_l[i-1]*vec[i-1]
        return vec_new

    def solve(self, vec):
        sol = [0]*self.length
        for i in range(self.length):
            if (i == 0):
                sol[i] = vec[i]
            if (i == (self.length - 1)):
                s = 0
                for j in range(self.length - 2):
                    s = s + self.horizontal[j]*sol[j]
                sol[i] = vec[i] - s - self.diagonal_l[i-1]*sol[i-1]
            else:
                sol[i] = vec[i] - self.diagonal_l[i-1]*sol[i-1]
        return sol

    def show(self):
        print(f'diagonal   = {self.diagonal}')
        print(f'diagonal_l = {self.diagonal_l}')
        print(f'horizontal = {self.horizontal}')

class TimeSeries:
    def __init__(self, is_interior=True):
        self.solution = []
        self.is_interior = is_interior

    def get_length(self):
        return len(self.solution)

    def put_at_tail(self, vecs):
        if self.is_interior:
            vecs_new = []
            for ivec in vecs:
                vecs_new.append(
                    [iivec for iivec in ivec])
        else:
            vecs_new = [ivec for ivec in vecs]
        self.solution.append(vecs_new)

    def get_first(self):
        if self.is_interior:
            return np.array(self.solution)[:, 0, :]
        else:
            return np.array(self.solution)[:, 0]

    def get_second(self):
        if self.is_interior:
            return np.array(self.solution)[:, 1, :]
        else:
            return np.array(self.solution)[:, 1]

    def get_averages_first(self):
        averages = np.zeros(len(self.solution))
        if self.is_interior:
            for i, isolution in enumerate(np.array(self.solution)[:, 0, :]):
                averages[i] = isolution.mean()
            return averages
        else:
            print('function get_averages_first is defined for interiors.')

    def get_averages_second(self):
        averages = np.zeros(len(self.solution))
        if self.is_interior:
            for i, isolution in enumerate(np.array(self.solution)[:, 1, :]):
                averages[i] = isolution.mean()
            return averages
        else:
            print('function get_averages_second is defined for interiors.')

    def get_derivatives(self, vec, dx):
        derivative = np.zeros(len(vec))
        for i in range(len(vec)):
            if (i==0):
                derivative[i] = (vec[1] - vec[0])/dx
            elif (i==(len(vec)-1)):
                derivative[i] = (vec[i] - vec[i-1])/dx
            else:
                derivative[i] = (vec[i+1] - vec[i-1])/(2*dx)
        return derivative
    
    def get_derivatives_first(self, dx):
        derivative = np.zeros((
            len(self.solution),
            len(self.solution[0][0])))
        for i, isolution in enumerate(np.array(self.solution)[:, 0, :]):
            derivative[i, :] = self.get_derivatives(isolution, dx)
        return derivative

    def get_derivatives_second(self, dx):
        derivative = np.zeros((
            len(self.solution),
            len(self.solution[0][0])))
        for i, isolution in enumerate(np.array(self.solution)[:, 1, :]):
            derivative[i, :] = self.get_derivatives(isolution, dx)
        return derivative

    def show_heat_map_interior(self, dt, dx, name_title="domain-time heat map", name_to_save='domain_time_heatmap.png', save_dir='.'):
        if not self.is_interior:
            print(f"Interior solution must be vector")
            sys.exit()
        x_axis = np.array([dx*ix for ix in range(len(self.solution[0][0]))])
        y_axis = np.array([dt*it for it in range(len(self.solution))])
        X, Y = np.meshgrid(x_axis, y_axis)
        Z, W = self.get_first(), self.get_second()
    
        fig_u, ax = plt.subplots(constrained_layout=True)
        cs = ax.contourf(X, Y, Z, cmap='coolwarm')
        cb = fig_u.colorbar(cs)
        ax.set_xlabel("Space variable (x)", fontsize=16)
        ax.set_ylabel("Time variable (t)",  fontsize=16)
        cb.set_label("Value of v1",         fontsize=16)
        plt.title('{0} {1}'.format('v1', name_title), fontsize=18)
        #plt.savefig('./pngs/'+'{0}_{1}'.format('v1', name_to_save))
        plt.savefig('{0}/{1}_{2}'.format(save_dir, 'v1', name_to_save))
        plt.clf()
    
        fig_v, ax = plt.subplots(constrained_layout=True)
        cs = ax.contourf(X, Y, W, cmap='coolwarm')
        cb = fig_v.colorbar(cs)
        ax.set_xlabel("Space variable (x)", fontsize=16)
        ax.set_ylabel("Time variable (t)",  fontsize=16)
        cb.set_label("Value of v2",         fontsize=16)
        plt.title('{0} {1}'.format('v2', name_title), fontsize=18)
        #plt.savefig('./pngs/'+'{0}_{1}'.format('v2', name_to_save))
        plt.savefig('{0}/{1}_{2}'.format(save_dir, 'v2', name_to_save))
        plt.clf()

    def show_heat_map_derivative_interior(self, dt, dx, name_title="domain-time heat map (derivative)", name_to_save='domain_time_heatmap_derivative.png', save_dir='.'):
        if not self.is_interior:
            print(f"Interior solution must be vector")
            sys.exit()
        x_axis = np.array([dx*ix for ix in range(len(self.solution[0][0]))])
        y_axis = np.array([dt*it for it in range(len(self.solution))])
        X, Y = np.meshgrid(x_axis, y_axis)
        Z, W = self.get_derivatives_first(dx), self.get_derivatives_second(dx)
    
        fig_u, ax = plt.subplots(constrained_layout=True)
        cs = ax.contourf(X, Y, Z, cmap='hsv')
        cb = fig_u.colorbar(cs)
        ax.set_xlabel("Space variable (x)", fontsize=16)
        ax.set_ylabel("Time variable (t)",  fontsize=16)
        cb.set_label("Derivatives of v1",   fontsize=16)
        #ax.set_xticklabels(x, fontsize=12)
        #ax.set_yticklabels(y, fontsize=12)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.title('{0} {1}'.format('v1', name_title), fontsize=18)
        #plt.savefig('./pngs/'+'{0}_{1}'.format('v1', name_to_save))
        plt.savefig('{0}/{1}_{2}'.format(save_dir, 'v1', name_to_save))
        plt.clf()
    
        fig_v, ax = plt.subplots(constrained_layout=True)
        cs = ax.contourf(X, Y, W, cmap='hsv')
        cb = fig_v.colorbar(cs)
        ax.set_xlabel("Space variable (x)", fontsize=16)
        ax.set_ylabel("Time variable (t)",  fontsize=16)
        cb.set_label("Derivatives of v2",   fontsize=16)
        #ax.set_xticklabels(x, fontsize=12)
        #ax.set_yticklabels(y, fontsize=12)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.title('{0} {1}'.format('v2', name_title), fontsize=18)
        #plt.savefig('./pngs/'+'{0}_{1}'.format('v2', name_to_save))
        plt.savefig('{0}/{1}_{2}'.format(save_dir, 'v2', name_to_save))
        plt.clf()

    def show_averages_interior(self, dt, name_title="Time series of averages", name_to_save='time_series_averages.png', save_dir='.'):
        X = np.array([dt*it for it in range(len(self.solution))])
        Y0, Y1 = self.get_averages_first(), self.get_averages_second()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(X, Y0, label='Average of v1', color='blue')
        plt.plot(X, Y1, label='Average of v2', color='green')
        plt.grid()
        plt.legend(fontsize=16)
        plt.xlabel('Time (t)',                       fontsize=16)
        plt.ylabel('Averages of interior variables', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        plt.title(name_title, fontsize=18)
        #plt.savefig('./pngs/'+name_to_save)
        plt.savefig(save_dir + '/' + name_to_save)
        plt.clf()
    
    def show_graph_boundary(self, dt, name_title="Time series of sigma1 and sigma2", name_to_save='time_series_sigmas.png', save_dir='.'):
        if self.is_interior:
            print(f"Boundary solution must be scalar")
            sys.exit()
        X = np.array([dt*it for it in range(len(self.solution))])
        Y0, Y1 = np.array(self.get_first()), np.array(self.get_second())
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(X, Y0, label='sigma1', color="blue")
        plt.plot(X, Y1, label='sigma2', color="green")
        plt.grid()
        plt.legend(fontsize=16)
        plt.xlabel('Time (t)', fontsize=16)
        plt.ylabel("Value of boundary variables", fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        plt.title(name_title, fontsize=18)
        #plt.savefig('./pngs/'+name_to_save)
        plt.savefig(save_dir + '/'+ name_to_save)
        plt.clf()
    
    def show_animation(self, dt, dx, name_title="Graph of v1 and v2", name_to_save='graph_of_v1_v2.png', save_dir='.'):
        X = np.array([dx*ix for ix in range(len(self.solution[0][0]))])
        os.system('rm ./pngs_for_gif/*.png')
        interval = 10
        for it in range(len(self.solution)):
            if (it%interval == 0):
                hight = 10.0
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, hight)
                Y0, Y1 = np.array(self.solution)[it, 0, :], np.array(self.solution)[it, 1, :]
                plt.plot(X, Y0, label=f'v1[{it*dt:.2f}]', color="blue")
                plt.plot(X, Y1, label=f'v2[{it*dt:.2f}]', color="green")
                plt.grid()
                plt.legend(fontsize=16)
                plt.xlabel("x")
                plt.ylabel("Value of v")
                plt.title(name_title)
                plt.savefig(f'{save_dir}/pngs_for_gif/{it:05}_{name_to_save}.png')
                plt.clf()
        print('Making a movie')
        os.system('convert -delay 20 -loop 0 ./pngs_for_gif/*.png ./movie.gif')
        print('finished!')

class MyGraphics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def draw_discrete_heat_map(two_dimensional_data_as_dictionary, x, y, x_label, y_label, z_label, title, saving_path, z_limit=50):
        data = two_dimensional_data_as_dictionary
        Z    = np.array([[data[(ix, iy)] for ix in x] for iy in y])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            Z,
            origin='lower', cmap='coolwarm',
            vmin=0, vmax=z_limit)
        cb = fig.colorbar(im) 

        plt.xlabel(x_label,   fontsize=14)
        plt.ylabel(y_label,   fontsize=14)
        cb.set_label(z_label, fontsize=14)
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(y)))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        cb.ax.tick_params(       labelsize=12)

        plt.title(title, fontsize=16)
        plt.savefig(saving_path)
        plt.clf()

        #Save date as backup
        source_data = []
        source_data.append(['x', 'y', 'z'])
        for ix in x:
            for jy in y:
                source_data.append([ix, jy, data[(ix, jy)]])
        path, _ = os.path.splitext(saving_path)
        np.savetxt(f'{path}_as_backup.csv', source_data, delimiter=',', fmt='%s')

########################################################################################
#######################################Solver###########################################
########################################################################################
def save_figures(
        solutions_interior, solutions_boundary,
        parameters, bio_parameters,
        enable_save_heat_map=False,          enable_save_heat_map_derivative=False,
        enable_save_averages_interior=False, enable_save_graph_boundary=False,
        limit_level_heat_map=None,
        title='',
        save_name=None,
        save_dir='./'):
    if save_name is not None:
        pass
    else:
        save_name = ''
    if enable_save_heat_map:
        solutions_interior.show_heat_map_interior(
            dt=parameters.dt,
            dx=parameters.dx,
            name_title=(\
                "Time Series Heatmap: "\
                +title),
            name_to_save=(\
                'heatmap_'
                +save_name),
            save_dir=save_dir)
    if enable_save_heat_map_derivative:
        solutions_interior.show_heat_map_derivative_interior(
            dt=parameters.dt,
            dx=parameters.dx,
            name_title=(\
                "Time Series Heatmap to Derivatives: "\
                +title),
            name_to_save=(\
                'heatmap_derivative_'\
                +save_name),
            save_dir=save_dir)
    if enable_save_averages_interior:
        solutions_interior.show_averages_interior(
            dt=parameters.dt,
            name_title=(\
                "Time Series of Spatial Averages to v1 and v2: "\
                +title),
            name_to_save=(\
                'time_series_averages_'\
                +save_name),
            save_dir=save_dir)
    if enable_save_graph_boundary:
        solutions_boundary.show_graph_boundary(
            dt=parameters.dt,
            name_title=(\
                "Time Series of sigma1 and sigma2: "\
                +title),
            name_to_save=(\
                'time_series_sigmas_'\
                +save_name),
            save_dir=save_dir)

def get_evolution_matrix_dynamic(boundaries, data, dx, number_interior_points, diffusion_coefficient, speed_ratio, filtering_scale):
    matrix = FivefoldDiagonalMatrix(number_interior_points)
    velocity = data.velocity(
        speed_ratio=speed_ratio,
        clogging_substance=boundaries[0],
        scale=filtering_scale)
    a = diffusion_coefficient/(dx**2)
    b = velocity/(2*dx)
    absorption_rate   = data.absorption_rate_into_filter(
                            clogging_substance=boundaries[0],
                            scale=filtering_scale)
    remainder         = 1 - absorption_rate
    remainder_squared = remainder*remainder
    for i in range(number_interior_points):
        if (i == 0):
            left_left  = (1 + (-1 + remainder_squared)/(1 + remainder_squared))/2
            left_right = 2*remainder/(1 + remainder_squared)/2
            matrix.diagonal[i]     = -2*a + a*left_left + b*left_left
            matrix.diagonal_u[i]   = a - b
            matrix.edge_u          = a*left_right + b*left_right
        elif (i == (number_interior_points - 1)):
            right_left  = 2*remainder/(1 + remainder_squared)/2
            right_right = (1 + (1 - remainder_squared)/(1 + remainder_squared))/2
            matrix.diagonal[i]     = -2*a + a*right_right - b*right_right
            matrix.diagonal_l[i-1] = a + b #debug ?????
            matrix.edge_l          = a*right_left - b*right_left
        else:
            matrix.diagonal[i]     = -2*a
            matrix.diagonal_u[i]   = a - b
            matrix.diagonal_l[i-1] = a + b
    return matrix

def set_endpoints(lattice, clogging_substance, data, filtering_scale):
    absorption_rate   = data.absorption_rate_into_filter(
                            clogging_substance=clogging_substance,
                            scale=filtering_scale)
    remainder         = 1 - absorption_rate
    remainder_squared = remainder*remainder
    left_from_left_boundary  = (1 + (-1 + remainder_squared)/(1 + remainder_squared))
    right_from_left_boundary = 2*remainder/(1 + remainder_squared)
    left_value               = (left_from_left_boundary*lattice.interior[0]\
                             + right_from_left_boundary*lattice.interior[-1])/2
    left_from_right_boundary  = 2*remainder/(1 + remainder_squared)
    right_from_right_boundary = (1 + (1 - remainder_squared)/(1 + remainder_squared))
    right_value               = (left_from_right_boundary*lattice.interior[0]\
                              + right_from_right_boundary*lattice.interior[-1])/2
    lattice.plugin_endpoints(
        left=left_value,
        right=right_value)

def forward_by_Euler(interiors, boundaries, data, parameters, bio_parameters):
    #solve interiors
    force_interior1, force_interior2, _, _\
        = data.get_forces(
            interiors=interiors,
            boundaries=boundaries,
            parameters=parameters,
            bio_parameters=bio_parameters)
    matrix1 = get_evolution_matrix_dynamic(
        boundaries=boundaries,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient1,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    matrix2 = get_evolution_matrix_dynamic(
        boundaries=boundaries,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient2,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    interior_new1\
        = interiors[0]\
        + parameters.dt*matrix2.apply(interiors[0])\
        + parameters.dt*force_interior1
    interior_new2\
        = interiors[1]\
        + parameters.dt*matrix1.apply(interiors[1])\
        + parameters.dt*force_interior2
    set_endpoints(
        lattice=interior_new1,
        clogging_substance=boundaries[0],
        data=data,
        filtering_scale=bio_parameters.filtering_scale)
    set_endpoints(
        lattice=interior_new2,
        clogging_substance=boundaries[0],
        data=data,
        filtering_scale=bio_parameters.filtering_scale)
    #solve boundaries
    _, _, force_boundary1, force_boundary2 = data.get_forces(
        interiors=[
            interior_new1,
            interior_new2],
        boundaries=boundaries,
        parameters=parameters,
        bio_parameters=bio_parameters)
    boundaries_new1 = boundaries[0] + parameters.dt*force_boundary1
    boundaries_new2 = boundaries[1] + parameters.dt*force_boundary2
    return interior_new1, interior_new2, boundaries_new1, boundaries_new2

def forward_by_Crank_Nicolson(interiors_old, boundaries_old, interiors_new_tmp, boundaries_new_tmp, data, parameters, bio_parameters):
    I = FivefoldDiagonalMatrix(parameters.number_interior_points)
    ########################Solve Equations########################
    #solve u1
    force_interior_old1, _, _, _ = data.get_forces(
        interiors=interiors_old,
        boundaries=boundaries_old,
        parameters=parameters,
        bio_parameters=bio_parameters)
    force_interior_new_tmp1, _, _, _ = data.get_forces(
        interiors=interiors_new_tmp,
        boundaries=boundaries_new_tmp,
        parameters=parameters,
        bio_parameters=bio_parameters)
    matrix_old1 = get_evolution_matrix_dynamic(
        boundaries=boundaries_old,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient1,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    matrix1_tmp = get_evolution_matrix_dynamic(
        boundaries=boundaries_new_tmp,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient1,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    interior_new1 = (I - (parameters.dt/2)*matrix1_tmp).solve(
        interiors_old[0]\
        + (parameters.dt/2)*matrix_old1.apply(interiors_old[0])\
        + (parameters.dt/2)*(\
            force_interior_new_tmp1\
            + force_interior_old1))
    set_endpoints(
        lattice=interior_new1,
        clogging_substance=boundaries_new_tmp[0],
        data=data,
        filtering_scale=bio_parameters.filtering_scale)
    #solve u2
    _, force_interior_old2, _, _ = data.get_forces(
        interiors=[
            interior_new1,
            interiors_old[1]],
        boundaries=boundaries_old,
        parameters=parameters,
        bio_parameters=bio_parameters)
    _, force_interior_new_tmp2, _, _ = data.get_forces(
        interiors=[
            interior_new1,
            interiors_new_tmp[1]],
        boundaries=boundaries_new_tmp,
        parameters=parameters,
        bio_parameters=bio_parameters)
    matrix_old2 = get_evolution_matrix_dynamic(
        boundaries=boundaries_old,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient1,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    matrix2_tmp = get_evolution_matrix_dynamic(
        boundaries=boundaries_new_tmp,
        data=data,
        dx=parameters.dx,
        number_interior_points=parameters.number_interior_points,
        diffusion_coefficient=bio_parameters.diffusion_coefficient2,
        speed_ratio=bio_parameters.speed_ratio,
        filtering_scale=bio_parameters.filtering_scale)
    interior_new2 = (I - (parameters.dt/2)*matrix2_tmp).solve(
        interiors_old[1]\
        + (parameters.dt/2)*matrix_old2.apply(interiors_old[1])\
        + (parameters.dt/2)*(\
            force_interior_new_tmp2\
            + force_interior_old2))
    set_endpoints(
        lattice=interior_new2,
        clogging_substance=boundaries_new_tmp[0],
        data=data,
        filtering_scale=bio_parameters.filtering_scale)
    #solve rho1
    _, _, force_boundary_old1, _ = data.get_forces(
        interiors=[
            interior_new1,
            interior_new2],
        boundaries=boundaries_old,
        parameters=parameters,
        bio_parameters=bio_parameters)
    _, _, force_boundary_new_tmp1, _ = data.get_forces(
        interiors=[
            interior_new1,
            interior_new2],
        boundaries=boundaries_new_tmp,
        parameters=parameters,
        bio_parameters=bio_parameters)
    boundary_new1\
        = boundaries_old[0]\
        + (parameters.dt/2)*force_boundary_new_tmp1\
        + (parameters.dt/2)*force_boundary_old1
    #solve rho2
    _, _, _, force_boundary_old2 = data.get_forces(
        interiors=[
            interior_new1,
            interior_new2],
        boundaries=[
            boundary_new1,
            boundaries_old[1]],
        parameters=parameters,
        bio_parameters=bio_parameters)
    _, _, _, force_boundary_new_tmp2 = data.get_forces(
        interiors=[
            interior_new1,
            interior_new2],
        boundaries=[
            boundary_new1,
            boundaries_new_tmp[1]],
        parameters=parameters,
        bio_parameters=bio_parameters)
    boundary2_new\
        = boundaries_old[1]\
        + (parameters.dt/2)*force_boundary_new_tmp2\
        + (parameters.dt/2)*force_boundary_old2
    return interior_new1, interior_new2, boundary_new1, boundary2_new

def solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Euler(solutions_interior, solutions_boundary, data, parameters, bio_parameters, predator_capacity_ratio_interior=1.0, predator_capacity_ratio_boundary=1.0):
    initial_data_interior1, initial_data_interior2\
    = data.get_two_initial_data_interior(
        domain=parameters.domain,
        nx=parameters.nx,
        dx=parameters.dx,
        predator_capacity_ratio_interior=predator_capacity_ratio_interior)
    initial_data_boundary1, initial_data_boundary2\
    = data.get_two_initial_data_boundary(
        predator_capacity_ratio_boundary=predator_capacity_ratio_boundary)
    solutions_interior.put_at_tail([
        initial_data_interior1.list_of_lattice_with_endpoints,
        initial_data_interior2.list_of_lattice_with_endpoints])
    solutions_boundary.put_at_tail([
        initial_data_boundary1,
        initial_data_boundary2])
    interior_old1, interior_old2\
        = initial_data_interior1, initial_data_interior2
    boundary_old1, boundary_old2\
        = initial_data_boundary1, initial_data_boundary2
    for it in range(parameters.nt-1):
        if it==0:
            print(f'it = {it:05}')
            interior_new1, interior_new2,\
            boundary_new1, boundary_new2 = forward_by_Euler(
                interiors=[
                    interior_old1,
                    interior_old2],
                boundaries=[
                    boundary_old1,
                    boundary_old2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            solutions_interior.put_at_tail([
                interior_new1.list_of_lattice_with_endpoints,
                interior_new2.list_of_lattice_with_endpoints])
            solutions_boundary.put_at_tail([
                boundary_new1,
                boundary_new2])
        else:
            if (it%1000 == 0): print(f'it = {it:05}')
            interior_old1, interior_old2, boundary_old1, boundary_old2\
                = interior_new1.copy(), interior_new2.copy(),\
                  boundary_new1, boundary_new2
            interior_new1, interior_new2, boundary_new1, boundary_new2 = forward_by_Euler(
                interiors=[
                    interior_old1,
                    interior_old2],
                boundaries=[
                    boundary_old1,
                    boundary_old2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            solutions_interior.put_at_tail([
                interior_new1.list_of_lattice_with_endpoints,
                interior_new2.list_of_lattice_with_endpoints])
            solutions_boundary.put_at_tail([
                boundary_new1,
                boundary_new2])

def solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior, solutions_boundary,
        data,
        parameters, bio_parameters,
        predator_capacity_ratio_interior=1.0, predator_capacity_ratio_boundary=1.0,
        max_length=None,
        saving_frequency=None):
    initial_data_interior1, initial_data_interior2\
    = data.get_two_initial_data_interior(
        domain=parameters.domain,
        nx=parameters.nx,
        dx=parameters.dx,
        predator_capacity_ratio_interior=predator_capacity_ratio_interior)
    initial_data_boundary1, initial_data_boundary2\
    = data.get_two_initial_data_boundary(
        predator_capacity_ratio_boundary=predator_capacity_ratio_boundary)
    solutions_interior.put_at_tail([
        initial_data_interior1.list_of_lattice_with_endpoints,
        initial_data_interior2.list_of_lattice_with_endpoints])
    solutions_boundary.put_at_tail([
        initial_data_boundary1,
        initial_data_boundary2])
    interior_old1, interior_old2\
        = initial_data_interior1, initial_data_interior2
    boundary_old1, boundary_old2\
        = initial_data_boundary1, initial_data_boundary2
    for it in range(parameters.nt-1):
        if it==0:
            #print(f'it = {it:05}')
            interior_new_tmp1, interior_new_tmp2,\
            boundary_new_tmp1, boundary_new_tmp2 = forward_by_Euler(
                interiors=[
                    interior_old1,
                    interior_old2],
                boundaries=[
                    boundary_old1,
                    boundary_old2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            interior_new1, interior_new2, boundary_new1, boundary_new2 = forward_by_Crank_Nicolson(
                interiors_old=[
                    interior_old1,
                    interior_old2],
                boundaries_old=[
                    boundary_old1,
                    boundary_old2],
                interiors_new_tmp=[
                    interior_new_tmp1,
                    interior_new_tmp2],
                boundaries_new_tmp=[
                    boundary_new_tmp1,
                    boundary_new_tmp2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            solutions_interior.put_at_tail([
                interior_new1.list_of_lattice_with_endpoints,
                interior_new2.list_of_lattice_with_endpoints])
            solutions_boundary.put_at_tail([
                boundary_new1,
                boundary_new2])
        else:
            #if (it%1000 == 0): print(f'it = {it:05}')
            interior_old1, interior_old2, boundary_old1, boundary_old2\
                = interior_new1.copy(), interior_new2.copy(),\
                  boundary_new1, boundary_new2
            interior_new_tmp1, interior_new_tmp2, boundary_new_tmp1, boundary_new_tmp2 = forward_by_Euler(
                interiors=[
                    interior_old1,
                    interior_old2],
                boundaries=[
                    boundary_old1,
                    boundary_old2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            interior_new1, interior_new2, boundary_new1, boundary_new2 = forward_by_Crank_Nicolson(
                interiors_old=[
                    interior_old1,
                    interior_old2],
                boundaries_old=[
                    boundary_old1,
                    boundary_old2],
                interiors_new_tmp=[
                    interior_new_tmp1,
                    interior_new_tmp2],
                boundaries_new_tmp=[
                    boundary_new_tmp1,
                    boundary_new_tmp2],
                data=data,
                parameters=parameters,
                bio_parameters=bio_parameters)
            if saving_frequency is None:
                solutions_interior.put_at_tail([
                    interior_new1.list_of_lattice_with_endpoints,
                    interior_new2.list_of_lattice_with_endpoints])
                solutions_boundary.put_at_tail([
                    boundary_new1,
                    boundary_new2])
            else:
                if (it%saving_frequency == 0):
                    solutions_interior.put_at_tail([
                        interior_new1.list_of_lattice_with_endpoints,
                        interior_new2.list_of_lattice_with_endpoints])
                    solutions_boundary.put_at_tail([
                        boundary_new1,
                        boundary_new2])
            if max_length is None:
                continue
            else:
                if solutions_interior.get_length()>max_length:
                    #print(f"solutions_interior.get_length() = {solutions_interior.get_length()}")
                    solutions_interior.solution = solutions_interior.solution[max_length-10:]
                    solutions_boundary.solution = solutions_boundary.solution[max_length-10:]
                    #print(f"solutions_interior.get_length() = {solutions_interior.get_length()}")
                    #Remark: 10 is arbitrary.
                else:
                    continue

def in_test3_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson(growth_rate_boundary, feeding):
    parameters=Parameters()
    bio_parameters=BioParameters()
    bio_parameters.growth_rate_boundary1 = growth_rate_boundary
    bio_parameters.growth_rate_boundary2 = growth_rate_boundary
    bio_parameters.feeding               = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters)
    save_figures(
        solutions_interior=v,
        solutions_boundary=sigma,
        parameters=parameters,
        bio_parameters=bio_parameters,
        enable_save_heat_map=True, 
        enable_save_heat_map_derivative=False, 
        enable_save_averages_interior=True, 
        enable_save_graph_boundary=True)

def in_test4_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson(filter_predator_capacities, feeding):
    parameters=Parameters()
    bio_parameters=BioParameters()
    bio_parameters.growth_rate_boundary1\
        = bio_parameters.growth_rate_boundary1*filter_predator_capacities
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacities
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters)
    save_dir = './pngs_filter_predator_capacities_vs_feeding_by_Crank_Nicolson'
    save_figures(
        solutions_interior=v,
        solutions_boundary=sigma,
        parameters=parameters,
        bio_parameters=bio_parameters,
        enable_save_heat_map=True, 
        enable_save_heat_map_derivative=False, 
        enable_save_averages_interior=True, 
        enable_save_graph_boundary=True,
        save_dir=save_dir)

def in_test5_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson(filter_predator_capacities, feeding, averages_at_final_time_step, nt):
    parameters=Parameters(); parameters.nt = nt
    bio_parameters=BioParameters()
    bio_parameters.growth_rate_boundary1\
        = bio_parameters.growth_rate_boundary1*filter_predator_capacities
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacities
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters)
    averages_at_final_time_step[(filter_predator_capacities, feeding)] = v.get_averages_first()[-1]

def in_test6_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson_by_original_parameters(filter_predator_capacities, feeding, averages_at_final_time_step, nt):
    parameters    = Parameters()
    parameters.nt = nt
    bio_parameters\
        = BioParameters()
    bio_parameters.growth_rate_boundary2\
        = bio_parameters.growth_rate_boundary2*filter_predator_capacities
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacities
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters,
        predator_capacity_ratio_boundary=filter_predator_capacities)
    averages_at_final_time_step[(filter_predator_capacities, feeding)] = v.get_averages_first()[-1]
    #save_figures(
    #    solutions_interior=v,
    #    solutions_boundary=sigma,
    #    parameters=parameters,
    #    bio_parameters=bio_parameters,
    #    enable_save_heat_map=True, 
    #    enable_save_heat_map_derivative=False, 
    #    enable_save_averages_interior=False, 
    #    enable_save_graph_boundary=True,
    #    save_name='nt_{0}_capacity_magnification_{1}_feed_magnification_{2}.png'.format(nt, filter_predator_capacities, feeding))

def in_test7_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters(filter_predator_capacity, feeding, v1s, nt):
    parameters    = Parameters()
    parameters.nt = nt
    bio_parameters\
        = BioParameters()
    bio_parameters.growth_rate_boundary2\
        = bio_parameters.growth_rate_boundary2*filter_predator_capacity
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacity
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters,
        predator_capacity_ratio_boundary=filter_predator_capacity)
    v1s[(filter_predator_capacity, feeding)] = v.get_averages_first()
    #save_figures(
    #    solutions_interior=v,
    #    solutions_boundary=sigma,
    #    parameters=parameters,
    #    bio_parameters=bio_parameters,
    #    enable_save_heat_map=True, 
    #    enable_save_heat_map_derivative=False, 
    #    enable_save_averages_interior=False, 
    #    enable_save_graph_boundary=True,
    #    save_name='nt_{0}_capacity_magnification_{1}_feed_magnification_{2}.png'.format(nt, filter_predator_capacities, feeding))

def in_test8_find_from_time_derivative_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_original_parameters(filter_predator_capacity, feeding, averages_at_final_time_step, dt, nt):
    ##Debug
    #averages_at_final_time_step[(filter_predator_capacity, feeding)] = 1.0
    #return
    ##End debug
    parameters    = Parameters()
    parameters.dt = dt
    parameters.nt = nt
    bio_parameters\
        = BioParameters()
    bio_parameters.growth_rate_boundary2\
        = bio_parameters.growth_rate_boundary2*filter_predator_capacity
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacity
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters,
        predator_capacity_ratio_boundary=filter_predator_capacity,
        max_length=40000)
    averages_at_final_time_step[(filter_predator_capacity, feeding)] = (v.get_averages_first()[-1] - v.get_averages_first()[-2])/parameters.dt

def in_test9_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters_with_save_freq(filter_predator_capacity, feeding, v1s, nt, dt, saving_frequency):
    parameters    = Parameters()
    parameters.nt = nt
    parameters.dt = dt
    bio_parameters\
        = BioParameters()
    bio_parameters.growth_rate_boundary2\
        = bio_parameters.growth_rate_boundary2*filter_predator_capacity
    bio_parameters.inflow_rate2\
        = bio_parameters.inflow_rate2/filter_predator_capacity
    bio_parameters.feeding\
        = feeding
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters,
        predator_capacity_ratio_boundary=filter_predator_capacity,
        saving_frequency=saving_frequency)
    #debug
    #save_figures(
    #    solutions_interior=v,
    #    solutions_boundary=sigma,
    #    parameters=parameters,
    #    bio_parameters=bio_parameters,
    #    enable_save_heat_map=True, 
    #    enable_save_heat_map_derivative=False, 
    #    enable_save_averages_interior=True, 
    #    enable_save_graph_boundary=True)
    v1s[(filter_predator_capacity, feeding)] = v.get_averages_first()

#######################################################################################
#----------------------------------------TESTS----------------------------------------#
#######################################################################################
def test0_compare_parameters_growth_rate_boundary_vs_feeding_rate_by_by_Euler():
    parameters=Parameters()
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    print("computing")
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Euler(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=Parameters(),
        bio_parameters=BioParameters())
    print("computation was finished!")
    print("printing pictures")
    v.show_heat_map_interior(
        dt=parameters.dt,
        dx=parameters.dx)
    v.show_heat_map_derivative_interior(
        dt=parameters.dt,
        dx=parameters.dx)
    sigma.show_graph_boundary(
        dt=parameters.dt)
    v.show_averages_interior(
        dt=parameters.dt)
    print("printing was finished!")

def test1_single_process_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson():
    parameters     = Parameters()
    bio_parameters = BioParameters()
    v     = TimeSeries() #time series of interior sol.
    sigma = TimeSeries(is_interior=False) #time series of boundary sol.
    print("computing.")
    solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
        solutions_interior=v,
        solutions_boundary=sigma,
        data=MyData(),
        parameters=parameters,
        bio_parameters=bio_parameters)
    print('finished!')
    print('printing.')
    save_figures(
        solutions_interior=v,
        solutions_boundary=sigma,
        parameters=parameters,
        bio_parameters=bio_parameters,
        enable_save_heat_map=True, 
        enable_save_heat_map_derivative=False, 
        enable_save_averages_interior=True, 
        enable_save_graph_boundary=True)
    print("finished!")

def test3_multi_process_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson():
    length = Parameters().length
    growth_rates_boundary = [1.0, 2.0, 3.0, 4.0]
    feedings = [0.5/length, 1.0/length, 1.5/length, 2.0/length]
    print("computing")
    with multiprocessing.Pool() as pool:
        pool.starmap(in_test3_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson, [
            (gr, f)
            for gr in growth_rates_boundary
            for f  in feedings])
    print("finished!")

def test4_multi_process_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson():
    length = Parameters().length
    filter_predator_capacities = [1.0, 0.75, 0.5, 0.25]
    #feedings = [0.5/length, 1.0/length, 1.5/length, 2.0/length]
    feedings = [0.25/length, 0.5/length, 0.75/length, 1.0/length]
    print("computing")
    with multiprocessing.Pool() as pool:
        pool.starmap(in_test4_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson, [
            (fc, f)
            for fc in filter_predator_capacities
            for f  in feedings])
    print("finished!")

def test5_find_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_Crank_Nicolson():
    length = Parameters().length
    nts                          = [10000, 20000, 40000, 80000]
    #nts                          = [10, 200, 400, 800]
    averages_at_final_time_steps = []
    filter_predator_capacities = [
        1.5, 1.4, 1.3, 1.2, 1.1,
        1.0, 0.9, 0.8, 0.7, 0.6,
        0.5, 0.4, 0.3, 0.2, 0.1]
    feedings = [
        0.1/length, 0.2/length, 0.3/length, 0.4/length, 0.5/length,
        0.6/length, 0.7/length, 0.8/length, 0.9/length, 1.0/length,
        1.1/length, 1.2/length, 1.3/length, 1.4/length, 1.5/length]
    print("computing")
    for i, nt in enumerate(nts):
        print(f"nt = {nt}")
        manager = Manager()
        averages_at_final_time_step = manager.dict()
        with multiprocessing.Pool() as pool:
            pool.starmap(in_test5_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson, [
                (fc, f, averages_at_final_time_step, nt)
                for fc in filter_predator_capacities
                for f  in feedings])
        MyGraphics.draw_discrete_heat_map(
            two_dimensional_data_as_dictionary=averages_at_final_time_step,
            x=sorted(filter_predator_capacities),
            y=feedings,
            x_label='Filter capacity',
            y_label='Feeding',
            z_label='spacial average of prey',
            title=f'Filter capacity vs Feeding at step {nt}',
            saving_path='./clogging_borderline_{}.png'.format(nt))
        averages_at_final_time_steps.append({
            key: value
            for key, value in averages_at_final_time_step.items()})
    for i, nt in enumerate(nts):
        difference_2D_data_as_dictionary = {}
        for icapacity in filter_predator_capacities:
            for jfeeding in feedings:
                difference_2D_data_as_dictionary[(icapacity, jfeeding)]\
                = averages_at_final_time_steps[-1][(icapacity, jfeeding)]\
                - averages_at_final_time_steps[i][(icapacity, jfeeding)]
                #print(f"difference_2D_data_as_dictionary[({icapacity}, {jfeeding})] = {difference_2D_data_as_dictionary[(icapacity, jfeeding)]}")
        MyGraphics.draw_discrete_heat_map(
            two_dimensional_data_as_dictionary=difference_2D_data_as_dictionary,
            x=np.array(sorted(filter_predator_capacities)),
            y=np.array(feedings),
            x_label='Filter capacity',
            y_label='Feeding',
            z_label='Difference of spacial averages of prey',
            title=f'Filter capacity vs Feeding ({nts[-1]}-{nt})',
            saving_path='./clogging_borderline_from_the_difference_between_{0}_{1}.png'.format(nts[0], nt),
            z_limit=5)
    print("finished!")

def test6_find_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_Crank_Nicolson_by_original_parameters():
    print('hello test6')
    """
    Assume
    A         = 1,         B       = 1
    u1        = v1,        u2      = Cu*v2
    rho1      = sigma1,    rho2    = Crho*rho2
    Rtilde1   = R*Cu,      Stilde1 = S1*Crho
    Q1        = 1,         Q2      = Cu/Crho
    F(sigma1) = F(sigma1), ftilde  = f
    """
    length = Parameters().length
    #nts                          = [10000, 20000]
    nts                          = [10000, 20000, 40000, 80000]
    #nts                          = [10, 200, 400, 800]
    averages_at_final_time_steps = []
    filter_predator_capacities = [
        1.5, 1.4, 1.3, 1.2, 1.1,
        1.0, 0.9, 0.8, 0.7, 0.6,
        0.5, 0.4, 0.3, 0.2, 0.1]
    feedings = [
        0.1/length, 0.2/length, 0.3/length, 0.4/length, 0.5/length,
        0.6/length, 0.7/length, 0.8/length, 0.9/length, 1.0/length,
        1.1/length, 1.2/length, 1.3/length, 1.4/length, 1.5/length]
    print("computing")
    for i, nt in enumerate(nts):
        print(f"nt = {nt}")
        manager = Manager()
        averages_at_final_time_step = manager.dict()
        with multiprocessing.Pool() as pool:
            pool.starmap(in_test6_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson_by_original_parameters, [
                (fc, f, averages_at_final_time_step, nt)
                for fc in filter_predator_capacities
                for f  in feedings])
        MyGraphics.draw_discrete_heat_map(
            two_dimensional_data_as_dictionary=averages_at_final_time_step,
            x=sorted(filter_predator_capacities),
            y=feedings,
            x_label='Filter capacity',
            y_label='Feeding',
            z_label='Spacial average of prey',
            title=f'Filter capacity vs Feeding at step {nt}',
            saving_path='./test6_clogging_borderline_{}.png'.format(nt))
        averages_at_final_time_steps.append({
            key: value
            for key, value in averages_at_final_time_step.items()})
    for i, nt in enumerate(nts):
        difference_2D_data_as_dictionary = {}
        for icapacity in filter_predator_capacities:
            for jfeeding in feedings:
                difference_2D_data_as_dictionary[(icapacity, jfeeding)]\
                = averages_at_final_time_steps[-1][(icapacity, jfeeding)]\
                - averages_at_final_time_steps[i][(icapacity, jfeeding)]
                #print(f"difference_2D_data_as_dictionary[({icapacity}, {jfeeding})] = {difference_2D_data_as_dictionary[(icapacity, jfeeding)]}")
        MyGraphics.draw_discrete_heat_map(
            two_dimensional_data_as_dictionary=difference_2D_data_as_dictionary,
            x=np.array(sorted(filter_predator_capacities)),
            y=np.array(feedings),
            x_label='Filter capacity',
            y_label='Feeding',
            z_label='Difference of spacial averages of prey',
            title=f'Filter capacity vs Feeding ({nts[-1]}-{nt})',
            saving_path='./test6_clogging_borderline_from_the_difference_between_{0}_{1}.png'.format(nts[0], nt),
            z_limit=5)
    print("finished!")

def test7_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters():
    print('hello test7')
    """
    Assume
    A         = 1,         B       = 1
    u1        = v1,        u2      = Cu*v2
    rho1      = sigma1,    rho2    = Crho*rho2
    Rtilde1   = R*Cu,      Stilde1 = S1*Crho
    Q1        = 1,         Q2      = Cu/Crho
    F(sigma1) = F(sigma1), ftilde  = f
    """
    parameters = Parameters()
    length     = parameters.length
    dt         = parameters.dt
    #nt         = 500000
    #nt         = 1000000
    nt         = 1500000
    filter_predator_capacity = 0.5
    feedings = [
        0.25/length, 0.50/length, 0.75/length, 1.00/length, 1.250/length]
    print("computing")
    manager = Manager()
    averages_for_first_variable = manager.dict()
    with multiprocessing.Pool() as pool:
        pool.starmap(in_test7_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters, [
            (filter_predator_capacity, f, averages_for_first_variable, nt)
            for f in feedings])

    X = np.array([dt*it for it in range(nt)])
    for feeding in reversed(feedings):
        average_for_first_variable = averages_for_first_variable[(filter_predator_capacity, feeding)]
        time_derivative = (average_for_first_variable[-1] - average_for_first_variable[-2])/parameters.dt
        plt.plot(
            X,
            average_for_first_variable,
            label='(filter_capacity, feeding, slope) = ({0:.2f}, {1:.2f}, {2:.2f})'.format(
                filter_predator_capacity,
                feeding,
                time_derivative))

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Spatial Averages of Interior Prey')
    plt.title('Time Series Spacial Average: Predator Capacities at the Filter VS Feeding')
    plt.savefig('./pngs_test7/test7_nt_{}.png'.format(nt), bbox_inches='tight')
    plt.clf()
    print("finished!")

def test8_find_from_time_derivative_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_original_parameters():
    print('hello test8')
    """
    Assume
    A         = 1,         B       = 1
    u1        = v1,        u2      = Cu*v2
    rho1      = sigma1,    rho2    = Crho*rho2
    Rtilde1   = R*Cu,      Stilde1 = S1*Crho
    Q1        = 1,         Q2      = Cu/Crho
    F(sigma1) = F(sigma1), ftilde  = f
    """
    parameters = Parameters() 
    length = parameters.length
    dt     = 0.001
    #nts                          = [10000, 20000]
    #nts                          = [25000, 50000, 75000, 100000]
    #nts                          = [1000000, 2000000, 4000000, 8000000]
    #nts                          = [5000000, 10000000]
    nts                          = [5000000]
    #nts                          = [400000]
    #nts                          = [10, 200, 400, 800]
    #filter_predator_capacities = [
    #    1.5, 1.4, 1.3, 1.2, 1.1,
    #    1.0, 0.9, 0.8, 0.7, 0.6,
    #    0.5, 0.4, 0.3, 0.2, 0.1]
    #feedings = [
    #    0.1/length, 0.2/length, 0.3/length, 0.4/length, 0.5/length,
    #    0.6/length, 0.7/length, 0.8/length, 0.9/length, 1.0/length,
    #    1.1/length, 1.2/length, 1.3/length, 1.4/length, 1.5/length]
    #filter_predator_capacities = [
    #    2.0, 1.8, 1.6, 1.4, 1.2,
    #    1.0, 0.8, 0.6, 0.4, 0.2]
    #feedings = [
    #    0.2/length, 0.4/length, 0.6/length, 0.8/length, 1.0/length,
    #    1.2/length, 1.4/length, 1.6/length, 1.8/length, 2.0/length]
    filter_predator_capacities = [
        0.2, 0.4, 0.6, 0.8, 1.0,
        1.2, 1.4, 1.6, 1.8, 2.0]
    feedings = [
        0.20/length, 0.4/length, 0.6/length, 0.8/length, 1.0/length,
        1.2/length,  1.4/length, 1.6/length, 1.8/length, 2.0/length,
        2.2/length,  2.4/length, 2.6/length, 2.8/length, 3.0/length]
    print("computing")
    for i, nt in enumerate(nts):
        print(f"nt = {nt}")
        manager = Manager()
        time_derivatives_to_average_at_final_time_step = manager.dict()
        with multiprocessing.Pool() as pool:
            pool.starmap(in_test8_find_from_time_derivative_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_original_parameters, [
                (fc, f, time_derivatives_to_average_at_final_time_step, dt, nt)
                for fc in filter_predator_capacities
                for f  in feedings])
        MyGraphics.draw_discrete_heat_map(
            two_dimensional_data_as_dictionary=time_derivatives_to_average_at_final_time_step,
            x=sorted(filter_predator_capacities),
            y=feedings,
            x_label='Filter capacity (C_rho)',
            y_label='Feeding (f)',
            z_label='Time derivative to spacial average of prey',
            title=f'Filter Capacity vs Feeding at Time {dt*nt}',
            saving_path='./pngs_test8/test8_clogging_borderline_{}.png'.format(nt),
            z_limit=0.5)
    print("finished!")

def test9_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters_with_save_freq():
    print('Hello test9')
    """
    Assume
    A         = 1,         B       = 1
    u1        = v1,        u2      = Cu*v2
    rho1      = sigma1,    rho2    = Crho*rho2
    Rtilde1   = R*Cu,      Stilde1 = S1*Crho
    Q1        = 1,         Q2      = Cu/Crho
    F(sigma1) = F(sigma1), ftilde  = f
    """
    parameters       = Parameters()
    length           = parameters.length
    dt = 0.001
    nt = 10000000
    #nt = 30000000
    saving_frequency         = 1000
    filter_predator_capacity = 0.5
    feedings = [
        0.25/length, 0.50/length, 0.75/length, 1.00/length, 1.250/length]
    print("Computing...")
    manager = Manager()
    averages_for_first_variable = manager.dict()
    with multiprocessing.Pool() as pool:
        pool.starmap(in_test9_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters_with_save_freq, [
            (filter_predator_capacity, f, averages_for_first_variable, nt, dt, saving_frequency)
            for f in feedings])

    N = len(averages_for_first_variable[(filter_predator_capacity, filter_predator_capacity)])
    X = np.log10(np.ones(N) + np.array([dt*saving_frequency*it for it in range(N)]))
    #X = np.log10(X[- N: ])
    fig, ax = plt.subplots(figsize=(8, 6))
    for feeding in reversed(feedings):
        average_for_first_variable = averages_for_first_variable[(filter_predator_capacity, feeding)]
        time_derivative = (average_for_first_variable[-1] - average_for_first_variable[-2])/(dt*saving_frequency)
        average_for_first_variable = np.log10(average_for_first_variable)
        plt.plot(
            X,
            average_for_first_variable,
            label='(filter_capacity, feeding, time_derivative) = ({0:.2f}, {1:.2f}, {2:.2f})'.format(
                filter_predator_capacity,
                feeding,
                time_derivative))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
    plt.grid()
    ax.tick_params(axis='both', labelsize=14)
    plt.xlabel('Log_10 t',                                    fontsize=16)
    plt.ylabel('Log_10 to Spatial Averages of Interior Prey', fontsize=16)
    plt.title('Time Series Spacial Average: Predator Capacities at the Filter VS Feeding', fontsize=18)
    plt.savefig('./pngs_test9/test9_time_{}.png'.format(dt*nt), bbox_inches='tight')
    #plt.savefig('./pngs_test9/test9_nt_{}.png'.format(parameters.nt), bbox_inches='tight')
    #plt.title('Time Series Spacial Average: Predator Capacities at the Filter VS Feeding (No Filtering)')
    #plt.savefig('./pngs_test9/test9_nt_{}_no_filtering.png'.format(parameters.nt), bbox_inches='tight')
    plt.clf()
    print("Finished!")

def test10_draw_time_series():
    feedings = [0.5, 2.0]
    overwrite = True
    for feeding in feedings:
        parameters     = Parameters()
        bio_parameters = BioParameters()
        bio_parameters.feeding = feeding
        parameters.dt = 0.001
        parameters.nt = 500000
        #parameters.nt = 5000
        #assume capacity=1.0
        v     = TimeSeries() #time series of interior sol.
        sigma = TimeSeries(is_interior=False) #time series of boundary sol.
        print("Computing...")
        solve_1d_drift_diffusion_equations_dynamic_boundary_conditions_by_Crank_Nicolson(
            solutions_interior=v,
            solutions_boundary=sigma,
            data=MyData(),
            parameters=parameters,
            bio_parameters=bio_parameters)
        print('Finished!')
        print('Printing...')
        MyData.mkdir("./pngs_test10", overwrite=overwrite); overwrite = False
        title = "Feeding={}".format(bio_parameters.feeding)
        dir_name = "./pngs_test10/test10_feeding_{0}_time_{1}".format(feeding, parameters.dt*parameters.nt)
        MyData.mkdir(dir_name)
        saving_name = 'feeding_{0}_nt_{1}.png'.format(bio_parameters.feeding, parameters.dt*parameters.nt)
        save_figures(
            solutions_interior=v,
            solutions_boundary=sigma,
            parameters=parameters,
            bio_parameters=bio_parameters,
            enable_save_heat_map=True, 
            enable_save_heat_map_derivative=False, 
            enable_save_averages_interior=True, 
            enable_save_graph_boundary=True,
            title=title,
            save_name=saving_name,
            save_dir=dir_name)
        print("Finished!")

def main():
    #test0_compare_parameters_growth_rate_boundary_vs_feeding_rate_by_by_Euler()
    #test1_single_process_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson()
    #test3_multi_process_compare_parameters_growth_rate_boundary_vs_feeding_by_Crank_Nicolson()
    #test4_multi_process_compare_parameters_filter_predator_capacities_vs_feeding_by_Crank_Nicolson()
    #test5_find_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_Crank_Nicolson()
    #test6_find_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_Crank_Nicolson_by_original_parameters()
    #test7_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters()
    #test8_find_from_time_derivative_when_clogging_occurs_filter_predator_capacities_vs_feeding_by_original_parameters()
    #test9_make_time_series_filter_predator_capacities_vs_feeding_by_original_parameters_with_save_freq()
    test10_draw_time_series()



if __name__ == '__main__':
    main()