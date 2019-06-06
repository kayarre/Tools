import os
import numpy as np

import womersley.utils
import scipy.special as special
import matplotlib.pyplot as plt 

#import womersley.components

class waveform_info(object):
    def __init__(self, dir_path, name, radius, period, kind="profile", mu=0.0035, rho=1050.0):
        """create instance of converted coefficient information
        """
        self.dir_path = dir_path
        self.name = name
        self.whole_path = os.path.join(dir_path, name)
        self.read_pressure_coef()
        self.rho = rho # density
        self.mu = mu # viscosity
        self.radius = radius # radius
        self.period = period
        self.kind = kind
        self.Q = 0.0 # m^3/s
        self.set_nu()
        self.set_N_coeff(20)
        
    def read_pressure_coef(self):
        data = np.loadtxt(self.whole_path, delimiter="\t")
        shape = data.shape
        # take complex conjugate, PAT!
        self.coef = np.array([ complex(data[1][j], data[0][j]) for j in range(data.shape[-1])])

    def get_coef(self):
        return self.coef

    def get_period(self):
        test = self.periodiszero()
        #print(test)
        if(test):
            raise NameError("period is not set")
        return self.period
    
    def set_period(self, new_period):
        self.period = new_period
        self.set_womersley()
    
    def periodiszero(self, rel_tol=1e-09, abs_tol=0.0):
        return (abs(self.period-0.0) <= max(rel_tol * max(abs(self.period), abs(0.0)), abs_tol))
    
    def set_womersley(self):
        if ( self.period > 0.0):
            self.alpha = 2.0*np.pi / self.period # womersley number
        else:
            self.alpha = 0.0

    def get_womersley(self):
        return self.alpha
    
    def set_avg_Q(self, Q):
        self.Q_avg = Q
    def get_avg_Q(self):
        return self.Q_avg

    def set_peak_Q(self, peakflow):
        self.Q_peak_flow = peakflow
    def get_peak_Q(self):
        return self.Q_peak_flow

    def set_min_Q(self, minflow):
        self.Q_min_flow = minflow
    def get_min_Q(self):
        return self.Q_min_flow
    
    def set_avg_v(self, avg_v):
        self.avg_v = avg_v
    def get_avg_v(self):
        return self.avg_v
    
    def set_peak_v(self, peak_v):
        self.peak_v = peak_v
    def get_peak_v(self):
        return self.peak_v
    
    def set_min_v(self, min_v):
        self.min_v = min_v
    def get_min_v(self):
        return self.min_v
    
    def set_nu(self):
        self.nu = self.mu / self.rho
    def get_nu(self):
        return self.nu

    def set_N_coeff(self, N):
        self.N = N
    def get_N_coeff(self):
        return self.N




"""
  takes in a waveform and generates information about it
"""
class Waveform_Data(object):
    """ Womerley waveform data class based on input data
    
    Paramters: 
    dir_path (string): directory to read and write womersley files
    data (numpy nx2): array with time and flow rate information
    radius (float): the radius of the tube
    waveform_ctr_name (string): the fourier cofficient file name
    period (float): has a setter function, period of cardiac cycle in seconds
    output_path (path): path to store figure data
    scale (float): default convert millisecond data to seconds, and mm/second to m/second
    mu (float): the dynamic viscosity (Pa s)
    rho (float): the fluid density
    kind ( string): the kind of waveform information ('ctr', 'peak', etc.) currently not used I guess
     
    attributes:
    time: the time
    shape: shape of each waveform
    radius: the inlet profile radius, meters

    """

    def __init__(self, dir_path, data, radius, waveform_ctr_name,
                 period=0.0, output_path='', scale=1000.0, mu=0.0035, rho=1050.0, kind='ctr'):
        """Return and Image Stack Object
            extract data
        """
        self.dir_path = dir_path
        self.data = data
        self.time = self.data[:,0]/scale # seconds
        self.flowvelocity = self.data[:,1]/scale # m/s 
        
        self.shape = self.time.shape
        self.set_radius(radius)

        self.waveform_ctr_name = waveform_ctr_name
        self.Q = 0.0 # m^3/s
        
        self.period = period
        self.mu = mu #Pa s
        self.rho = rho  #kg/m**3
        self.set_nu()
        self.set_N_coeff(20)
        self.set_kind( kind)
        
        if (output_path == ''):
            self.output_path = self.dir_path
        else:
            self.output_path = output_path

    def get_kind(self):
        return self.kind

    def set_kind(self, k):
        self.kind = k

    def get_radius(self):
        return self.radius

    def set_radius(self, r):
        self.radius = r

    def set_output_path(self, output_path_str):
        self.output_path = output_path_str
    
    def get_period(self):
        test = self.periodiszero()
        #print(test)
        if(test):
            raise ValueError("period is not set")
        return self.period
    
    def set_period(self, new_period):
        self.period = new_period
        self.set_womersley()
    
    def periodiszero(self, rel_tol=1e-09, abs_tol=0.0):
        return (abs(self.period-0.0) <= max(rel_tol * max(abs(self.period), abs(0.0)), abs_tol))

    def set_womersley(self):
        if ( self.period > 0.0):
            self.alpha = 2.0*np.pi / self.period # womersley number
        else:
            self.alpha = 0.0

    def get_womersley(self):
        return self.alpha

    def write_coefficients(self, coef_new, name):
        fp = os.path.join(self.output_path,name)
        with open(fp, 'w') as fourier_f:
            c_s = [(c.real, c.imag) for c in coef_new] 
            imag_c = []
            real_c = []
            for c in c_s:
                # take complex conjugate, PAT!
                imag_c.append("{0:16.16f}".format(c[1]))
                real_c.append("{0:16.16f}".format(c[0]))
            fourier_f.write('\t'.join(imag_c) + '\n') #imaginary first and real second
            fourier_f.write('\t'.join(real_c) + '\n')
            
    def read_pressure_coef(self, name):
        fp = os.path.join(self.output_path, name)
        data = np.loadtxt(fp, delimiter="\t")
        shape = data.shape
        # take complex conjugate, PAT!
        coef = np.array([ complex(data[1][j], data[0][j]) for j in range(data.shape[-1])])
        return coef
    
    def save_fig(self, fig, fig_name="test.jpg"):
        fig.savefig(os.path.join(self.output_path, fig_name))

    def set_avg_Q(self, Q):
        self.Q_avg = Q
    def get_avg_Q(self):
        return self.Q_avg

    def set_peak_Q(self, peakflow):
        self.Q_peak_flow = peakflow
    def get_peak_Q(self):
        return self.Q_peak_flow

    def set_min_Q(self, minflow):
        self.Q_min_flow = minflow
    def get_min_Q(self):
        return self.Q_min_flow
    
    def set_mean_v(self, mean_v):
        self.mean_v = mean_v
    def get_mean_v(self):
        return self.mean_v
    
    def set_peak_v(self, peak_v):
        self.peak_v = peak_v
    def get_peak_v(self):
        return self.peak_v
    
    def set_nu(self):
        self.nu = self.mu / self.rho
    def get_nu(self):
        return self.nu

    def set_N_coeff(self, N):
        self.N = N
    def get_N_coeff(self):
        return self.N
    
    
"""
  reads normative data from paper values
"""
class norm_waveform_data(object):
    """ Womerley waveform data class based on normative input data
    
    Paramters: 
    dir_path (string): directory to read and write womersley files
    data (numpy nx2): array with time and flow rate information
    radius (float): the radius of the tube
    waveform_ctr_name (string): the fourier cofficient file name
    period (float): has a setter function, period of cardiac cycle in seconds
    output_path (path): path to store figure data
    scale (float): default convert millisecond data to seconds, and mm/second to m/second
    mu (float): the dynamic viscosity (Pa s)
    rho (float): the fluid density
    kind ( string): the kind of waveform information ('ctr', 'peak', etc.) currently not used I guess
     
    attributes:
    time: the time
    shape: shape of each waveform
    radius: the inlet profile radius, meters

    """

    def __init__(self, dir_path, txt_name, radius, waveform_ctr_name,
                 period=0.0, output_path='', scale=1000.0, mu=0.0035, rho=1050.0, kind='ctr'):
        """Return and Image Stack Object
            extract data
        """
        self.dir_path = dir_path
        self.txt_path = os.path.join(dir_path, txt_name)
        self.data = np.loadtxt(self.txt_path, delimiter=",")
        self.time = self.data[:,0]/scale # seconds
        self.flowrate = self.data[:,1]/(10.**6*60.) # m^3/s
        
        self.shape = self.time.shape
        self.set_radius(radius)

        self.waveform_ctr_name = waveform_ctr_name
        self.Q = 0.0 # m^3/s
        
        self.period = period
        self.mu = mu #Pa s
        self.rho = rho  #kg/m**3
        self.set_nu()
        self.set_N_coeff(20)
        self.set_kind(kind)
        
        if (output_path == ''):
            self.output_path = self.dir_path
        else:
            self.output_path = output_path

    def get_kind(self):
        return self.kind

    def set_kind(self, k):
        self.kind = k

    def get_radius(self):
        return self.radius

    def set_radius(self, r):
        self.radius = r

    def set_output_path(self, output_path_str):
        self.output_path = output_path_str
    
    def get_period(self):
        test = self.periodiszero()
        #print(test)
        if(test):
            raise ValueError("period is not set")
        return self.period
    
    def set_period(self, new_period):
        self.period = new_period
        self.set_womersley()
    
    def periodiszero(self, rel_tol=1e-09, abs_tol=0.0):
        return (abs(self.period-0.0) <= max(rel_tol * max(abs(self.period), abs(0.0)), abs_tol))

    def set_womersley(self):
        if ( self.period > 0.0):
            self.alpha = 2.0*np.pi / self.period # womersley number
        else:
            self.alpha = 0.0

    def get_womersley(self):
        return self.alpha

    def write_coefficients(self, coef_new, name):
        fp = os.path.join(self.output_path,name)
        with open(fp, 'w') as fourier_f:
            c_s = [(c.real, c.imag) for c in coef_new] 
            imag_c = []
            real_c = []
            for c in c_s:
                # take complex conjugate, PAT!
                imag_c.append("{0:16.16f}".format(c[1]))
                real_c.append("{0:16.16f}".format(c[0]))
            fourier_f.write('\t'.join(imag_c) + '\n') #imaginary first and real second
            fourier_f.write('\t'.join(real_c) + '\n')
            
    def read_pressure_coef(self, name):
        fp = os.path.join(self.output_path, name)
        data = np.loadtxt(fp, delimiter="\t")
        shape = data.shape
        # take complex conjugate, PAT!
        coef = np.array([ complex(data[1][j], data[0][j]) for j in range(data.shape[-1])])
        return coef
    
    def save_fig(self, fig, fig_name="test.jpg"):
        fig.savefig(os.path.join(self.output_path, fig_name))

    def set_avg_Q(self, Q):
        self.Q_avg = Q
    def get_avg_Q(self):
        return self.Q_avg

    def set_peak_Q(self, peakflow):
        self.Q_peak_flow = peakflow
    def get_peak_Q(self):
        return self.Q_peak_flow

    def set_min_Q(self, minflow):
        self.Q_min_flow = minflow
    def get_min_Q(self):
        return self.Q_min_flow
    
    def set_mean_v(self, mean_v):
        self.mean_v = mean_v
    def get_mean_v(self):
        return self.mean_v
    
    def set_peak_v(self, peak_v):
        self.peak_v = peak_v
    def get_peak_v(self):
        return self.peak_v
    
    def set_nu(self):
        self.nu = self.mu / self.rho
    def get_nu(self):
        return self.nu

    def set_N_coeff(self, N):
        self.N = N
    def get_N_coeff(self):
        return self.N


class coefficient_converter(object):
    
    coef_types = ['ctr', 'mean', 'press', 'shear', "profile"]
    
    def __init__(self, waveform_obj, coef, kind="ctr"):
        """create instance of converted coefficient information
        """

        self.rho = waveform_obj.rho # density
        self.mu = waveform_obj.mu # viscosity
        self.R = waveform_obj.radius # radius
        self.period = waveform_obj.period
        self.set_omega()
        self.set_alpha()# womersley number
        self.kind = kind
        self.N_coef = coef.shape[0] # assume numpy array
        self.set_womersley_params()
        self.input_coef = coef
        self.coef_dict = {}
        self.set_coef()


    def set_womersley_params(self):
        """Returns the womersley number alpha.
        @param R: pipe radius
        @param omega: oscillation frequency
        @param mu: viscosity
        @param rho: fluid density
        """
        self.omega_n = self.omega * np.arange(1.0, self.N_coef)
        alpha_n = self.alpha * np.sqrt(np.arange(1.0, self.N_coef))
        self.lambda_ = np.sqrt(1.0j**3) * alpha_n
        self.J1 = special.jn(1, self.lambda_)
        self.J0 = special.jn(0, self.lambda_)


    def coef_norm(self, kind):
        
        coef = np.zeros(self.input_coef.shape, dtype=np.complex)
        if (kind == "mean"):
            coef[0] = self.R**2 /(8.0*self.mu)
            coef[1:] = 1.0 - 2.0 / self.lambda_ * self.J1 / self.J0
        elif (kind == "ctr"):
            coef[0] = self.R**2 / (4.0*self.mu)
            coef[1:] = 1.0 - 1.0 / self.J0
        elif (kind == "shear"):
            coef[0] = self.R / 2.0
            coef[1:] = - self.mu * self.lambda_ / self.R * self.J1 / self.J0
        elif (kind == "press"):
            coef[0] = -1.0
            coef[1:] = 1.0j*self.rho*self.omega_n
        elif (kind == "profile"):
            coef[0] = self.R**2 / (4.0*self.mu)
            coef[1:] = np.ones(self.omega_n.shape)
        else:
            print("unknown kind given")
            raise Exception('Unknown kind: {0}'.format(kind))

        return coef
 
    def convert(self, coef_in_kind, coeff_set_list):
        coef_normed = self.coef_norm(coef_in_kind)
        
        for k in coeff_set_list:
            # gets the normed info based on kind
            self.coef_dict[k] = self.input_coef * self.coef_norm(k) / coef_normed       
            
    def set_coef(self):
        coef_set_list = []
        for k in self.coef_types:
            if(k == self.kind):
                self.coef_dict[self.kind] = self.input_coef
            else:
                coef_set_list.append(k)
        self.convert(self.kind, coef_set_list)

        
    def set_radius(self, radius):
        self.radius = radius
        self.set_alpha()
        self.set_coef()
        
    def set_viscosity(self, mu):
        self.mu = mu
        self.set_alpha()
        self.set_coef()
    
    def set_density(self, rho):
        self.rho = rho
        self.set_alpha()
        self.set_coef()
    
    def set_omega(self):
        self.omega = 2.0*np.pi / self.period

    def set_period(self, period):
        self.period = period
        self.set_omega()
        self.set_alpha()

    def set_alpha(self):
        self.alpha = np.sqrt(self.omega * self.rho /self.mu) * self.R
        
    def set_radius(self, radius):
        self.radius = radius
        self.set_alpha()
        self.set_coef()
        


class converter_from_norm(object):
    
    coef_types = ['ctr', 'mean', 'press', 'shear', "profile"]
    
    def __init__(self, waveform_obj):
        """create instance of converted coefficient information
        """

        self.rho = waveform_obj.rho # density
        self.mu = waveform_obj.mu # viscosity
        self.R = waveform_obj.radius # radius
        self.period = waveform_obj.period
        self.set_omega()
        self.set_alpha()# womersley number
        self.kind = waveform_obj.kind
        self.N_coef = waveform_obj.coef.shape[0] # assume numpy array
        self.set_womersley_params()
        self.input_coef = waveform_obj.coef
        self.coef_dict = {}
        self.set_coef()


    def set_womersley_params(self):
        """Returns the womersley number alpha.
        @param R: pipe radius
        @param omega: oscillation frequency
        @param mu: viscosity
        @param rho: fluid density
        """
        self.omega_n = self.omega * np.arange(1.0, self.N_coef)
        alpha_n = self.alpha * np.sqrt(np.arange(1.0, self.N_coef))
        self.lambda_ = np.sqrt(1.0j**3) * alpha_n
        self.J1 = special.jn(1, self.lambda_)
        self.J0 = special.jn(0, self.lambda_)


    def coef_norm(self, kind):
        
        coef = np.zeros(self.input_coef.shape, dtype=np.complex)
        if (kind == "mean"):
            coef[0] = self.R**2 /(8.0*self.mu)
            coef[1:] = 1.0 - 2.0 / self.lambda_ * self.J1 / self.J0
        elif (kind == "ctr"):
            coef[0] = self.R**2 / (4.0*self.mu)
            coef[1:] = 1.0 - 1.0 / self.J0
        elif (kind == "shear"):
            coef[0] = self.R / 2.0
            coef[1:] = - self.mu * self.lambda_ / self.R * self.J1 / self.J0
        elif (kind == "press"):
            coef[0] = 1.0
            coef[1:] = 1.0j*self.rho*self.omega_n
        elif (kind == "profile"):
            coef[0] = self.R**2 / (4.0*self.mu)
            coef[1:] = np.ones(self.omega_n.shape)
        else:
            print("unknown kind given")
            raise Exception('Unknown kind: {0}'.format(kind))

        return coef
 
    def convert(self, coef_in_kind, coeff_set_list):
        coef_normed = self.coef_norm(coef_in_kind)
        
        for k in coeff_set_list:
            # gets the normed info based on kind
            self.coef_dict[k] = self.input_coef * self.coef_norm(k) / coef_normed       
            
    def set_coef(self):
        coef_set_list = []
        for k in self.coef_types:
            if(k == self.kind):
                self.coef_dict[self.kind] = self.input_coef
            else:
                coef_set_list.append(k)
        self.convert(self.kind, coef_set_list)

        
    def set_radius(self, radius):
        self.radius = radius
        self.set_alpha()
        self.set_coef()
        
    def set_viscosity(self, mu):
        self.mu = mu
        self.set_alpha()
        self.set_coef()
    
    def set_density(self, rho):
        self.rho = rho
        self.set_alpha()
        self.set_coef()
    
    def set_omega(self):
        self.omega = 2.0*np.pi / self.period

    def set_period(self, period):
        self.period = period
        self.set_omega()
        self.set_alpha()

    def set_alpha(self):
        self.alpha = np.sqrt(self.omega * self.rho /self.mu) * self.R
        
    def set_radius(self, radius):
        self.radius = radius
        self.set_alpha()
        self.set_coef()
        


# this converts the coefficients from the centerline flow velocity using the above function
# beware it would be different from using a  the mean velocity
def plot_womersley(waveform_obj, t, v_mean, time=0.0, N_coefs=20, save_clk=False):
    """
    @param waveform obj, has all the information regarding the waveform
    @param t, the time values
    @param v, the centerline velocity coefficients
    @time  t, single point in time to plot the velocity profile
    """
    #from womersley.components import coefficient_converter
    
    T = waveform_obj.get_period()
    N_coef = N_coefs
    omega = (2.0*np.pi)/T
    R = waveform_obj.radius # meters
    mu = waveform_obj.mu #Pa s
    rho = waveform_obj.rho #kg/m**3
    
    coef_peak = womersley.utils.generate_coefficents(v_mean, 1024)
    coeff_inst_2 = coefficient_converter(waveform_obj, coef_peak, "ctr")
    
    flow_test, flow_time_test = womersley.utils.reconstruct(coeff_inst_2.coef_dict["mean"][:N_coef], T)
    
    # flow added from additional outlet mean flow for the
    # right posterior inferior cerebeller 
    
    flow_add = flow_test + 1.12647226e-07/(np.pi*R**2)/flow_time_test[-1]
    
    
    flow_rate = np.pi*R**2*flow_test
    flow_total = np.trapz(flow_rate, flow_time_test)
    print('Flow, Area under curve: {0} $ml/cycle$'.format(flow_total.real*10**6*60))
    print('Flow average: {0} $ml/min$'.format(flow_total.real/T*10**6*60))
    
    coef_mean_new = womersley.utils.generate_coefficents(flow_add, 1024)
    
    coeff_inst = coefficient_converter(waveform_obj, coef_mean_new, "mean")
    
    
    #stuff to get coefficients
    coef_mean = coeff_inst.coef_dict["mean"][0:N_coef]
    coef_v = coeff_inst.coef_dict["ctr"][0:N_coef] #peak_2_mean(coef_v, rho, omega, mu, R)
    coef_p = coeff_inst.coef_dict["press"][0:N_coef] #peak_2_press(coef_v, rho, omega, mu, R)
    coef_shear = coeff_inst.coef_dict["shear"][0:N_coef] #peak_2_shear(coef_v, rho, omega, mu, R)
    coef_flow = coeff_inst.coef_dict["profile"][0:N_coef] #peak_2_shear(coef_v, rho, omega, mu, R)
    
    #print(coef_new)
    # Lets create some data to visualize
    
    r = np.linspace(-R, R, 2000)
    # convert velocity coefficients to pressure coefficients
    
    #write fourier coefficients
        #write fourier coefficients
    #if ( save_clk == True):
        #print("got here")
    waveform_obj.write_coefficients(coef_flow, waveform_obj.waveform_ctr_name)
    print("saving coefficients {0:s}".format(waveform_obj.waveform_ctr_name))

    #print(coef)
    u = []
    us = []
    ss = []
    for r_pt in r:
        #print(womersley_velocity(coef_p, rho, omega, mu, R, r_pt, time))
        u.append(womersley.utils.womersley_velocity(coef_flow, rho, omega, mu, R, r_pt, time).real)
        sol = womersley.utils.womersley_parts(coef_flow, rho, omega, mu, R, r_pt, time)
        ss.append(sol[0].real)
        us.append(sol[1].real)
    #print(u)
    # Lets create a plotting grid with gridspec
    # Create a 2 x 2 plotting grid object with horizontal and vertical spacing wspace and hspace
    gs = plt.GridSpec(7, 2, wspace=0.2, hspace=0.2) 
    # Create a figure
    fig = plt.figure(figsize=(11, 19))

    # SUBFIGURE 1
    # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Womersley Velocity profile', fontsize=20)
    ax1.set_xlabel('velocity (m/s)', fontsize=20)
    ax1.set_ylabel('Radius, (m)', fontsize=20)
    #ax1.plot(np.abs(u)/100., r)
    ax1.plot(u, r)
    int_mid = len(u)//2
    ax1.plot(u[int_mid], r[int_mid], 'go',
             markersize=10, label="{0:2.3e}".format(u[int_mid]))
    #ax1.plot(np.abs(u)[::-1], -1.0*r[::-1])
    # Arrange axis and labels
    ax1.legend()
    
    ax6 = fig.add_subplot(gs[1, :])
    ax6.set_title('Womersley steady', fontsize=20)
    ax6.set_xlabel('velocity (m/s)', fontsize=20)
    ax6.set_ylabel('Radius, (m)', fontsize=20)
    #ax1.plot(np.abs(u)/100., r)
    ax6.plot(ss, r)
    int_mid = len(ss)//2
    ax6.plot(ss[int_mid], r[int_mid], 'go',
             markersize=10, label="{0:2.3e}".format(ss[int_mid]))
    #ax1.plot(np.abs(u)[::-1], -1.0*r[::-1])
    # Arrange axis and labels
    ax6.legend()
    
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_title('Womersley unsteady', fontsize=20)
    ax7.set_xlabel('velocity (m/s)', fontsize=20)
    ax7.set_ylabel('Radius, (m)', fontsize=20)
    #ax1.plot(np.abs(u)/100., r)
    ax7.plot(us, r)
    int_mid = len(us)//2
    ax7.plot(us[int_mid], r[int_mid], 'go',
             markersize=10, label="{0:2.3e}".format(us[int_mid]))
    #ax1.plot(np.abs(u)[::-1], -1.0*r[::-1])
    # Arrange axis and labels
    ax7.legend()
    
    
    ax2 = fig.add_subplot(gs[3, :])
    ax2.set_title('Internal Carotid Artery', fontsize=20)
    #ax2.set_xlabel(r'$t$', fontsize=20)
    ax2.set_ylabel(r'Peak, $V(t)$ $m/s$', fontsize=20)
    
    Q = womersley.utils.reconstruct_pt(coef_v, time, T)
    Q2, t2 = womersley.utils.reconstruct(coef_v, T)
    vel_total = np.trapz(Q2,t2)
    print('peak velocity, Area under curve: {0} $m/s/cycle$'.format(vel_total.real))
    print('peak velocity average: {0} $m^3/s$'.format(vel_total.real/T))
    print('velocity peak: {0} velocity min: {1} $m^3/s$'.format(np.max(Q2.real),
                                                         np.min(Q2.real)))
    
    ax2.plot(t2, np.real(Q2))
    ax2.plot(time, np.real(Q), 'ro', markersize=10, label = "{0:2.3e}".format(Q.real))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.legend()
    ax1.axis([-0.06, 1.0,-R,R])
    ax2.set_xlim((0, T))
    
    ax3 = fig.add_subplot(gs[4, :])
    #ax3.set_title('Common Carotid Artery')
    #ax3.set_xlabel(r'$t$', fontsize=20)
    ax3.set_ylabel(r'Flow Rate, $Q(t)$ $ml/min$', fontsize=20)
    
    mean = womersley.utils.reconstruct_pt(coef_mean, time, T)
    mean3, t3 = womersley.utils.reconstruct(coef_mean, T)
    flow_total = np.pi*R**2*np.trapz(mean3,t3)
    flow_test = np.pi*R**2*mean3
    print('Flow, Area under curve: {0} $ml/cycle$'.format(flow_total.real*10**6*60))
    print('Flow average: {0} $ml/min$'.format(flow_total.real/T*10**6*60))
    print('Flow peak: {0} Flow min: {1} $ml/min$'.format(np.max(flow_test.real)*10**6*60,
                                                         np.min(flow_test.real)*10**6*60))
    print('Pulsatility Index (Q_max - Q_min)/Q_mean: {0} '.format((np.max(flow_test.real)- np.min(flow_test.real))
                                                                 / flow_total.real/T))
    
    
    ax3.plot(t3, np.real(flow_test)*10**6*60)
    ax3.plot(time, np.real(mean)*np.pi*R**2*10**6*60, 'ro',
             markersize=10, label = "{0:2.3e}".format(mean.real*np.pi*R**2*10**6*60))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.legend()
    ax3.set_xlim((0, T))
    
    ax4 = fig.add_subplot(gs[5, :])
    #ax4.set_title('Common Carotid Artery')
    ax4.set_xlabel(r'$t$', fontsize=20)
    ax4.set_ylabel(r'Shear, $\tau_{wall}$ $Pa$', fontsize=20)
    
    shear = womersley.utils.reconstruct_pt(coef_shear, time, T)
    shear4, t4 = womersley.utils.reconstruct(coef_shear, T)
    shear_total = np.trapz(shear4,t4)
    print('shear, Area under curve: {0} $Pa/cycle$'.format(shear_total.real))
    print('shear average: {0} $Pa/s$'.format(shear_total.real/T))
    print('shear peak: {0} shear min: {1} $Pa/s$'.format(np.max(shear4.real),
                                                         np.min(shear4.real)))
    
    ax4.plot(t4, np.real(shear4))
    ax4.plot(time, np.real(shear), 'ro', markersize=10, label = "{0:2.3e}".format(shear.real))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.legend()
    ax4.set_xlim((0, T))
    
    ax5 = fig.add_subplot(gs[6, :])
    #ax4.set_title('Common Carotid Artery')
    ax5.set_xlabel(r'$t$', fontsize=20)
    ax5.set_ylabel(r'Pressure, $P$ $Pa$', fontsize=20)
    
    press = womersley.utils.reconstruct_pt(coef_p, time, T)
    press5, t5 = womersley.utils.reconstruct(coef_p, T)
    press_total = np.trapz(press5,t5)
    print('pressure, Area under curve: {0} $Pa/cycle$'.format(press_total.real))
    print('pressure average: {0} $Pa/s$'.format(press_total.real/T))
    print('pressure peak: {0} pressure min: {1} $Pa/s$'.format(np.max(press5.real),
                                                         np.min(press5.real)))
    
    ax5.plot(t5, np.real(press5))
    ax5.plot(time, np.real(press), 'ro', markersize=10, label = "{0:2.2e}".format(press.real))
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax5.legend()
    ax5.set_xlim((0, T))
    
    
    #ensure all subplots fit bounding box
    gs.tight_layout(fig)
    #plt.show()


# this converts the coefficients from the centerline flow velocity using the above function
# beware it would be different from using a  the mean velocity
def plot_womersley_test(waveform_obj, coef, time=0.0):
    
    coef_p = coef
    
    T = waveform_obj.get_period()
    omega = (2.0*np.pi)/T
    R = waveform_obj.radius # meters
    mu = waveform_obj.mu #Pa s
    rho = waveform_obj.rho #kg/m**3
    
    N_coef = waveform_obj.get_N_coeff()
    
    coeff_inst = coefficient_converter(waveform_obj, coef_p, "profile")
    
    #stuff to get coefficients
    coef_mean = coeff_inst.coef_dict["mean"][0:N_coef]
    coef_v = coeff_inst.coef_dict["ctr"][0:N_coef] #peak_2_mean(coef_v, rho, omega, mu, R)
    coef_p = coeff_inst.coef_dict["press"][0:N_coef] #peak_2_press(coef_v, rho, omega, mu, R)
    coef_shear = coeff_inst.coef_dict["shear"][0:N_coef] #peak_2_shear(coef_v, rho, omega, mu, R)
    coef_flow = coeff_inst.coef_dict["profile"][0:N_coef] #peak_2_shear(coef_v, rho, omega, mu, R)

    # Lets create some data to visualize
    
    r = np.linspace(-R, R, 200)
    t = np.linspace(0.0, T, 300)
    # convert velocity coefficients to pressure coefficients
    
    u = []
    for r_pt in r:
        #print(womersley_velocity(coef_p, rho, omega, mu, R, r_pt, time))
        u.append(womersley.utils.womersley_velocity(coef_flow, rho, omega, mu, R, r_pt, time).real)
    #print(u)
    # Lets create a plotting grid with gridspec
    # Create a 2 x 2 plotting grid object with horizontal and vertical spacing wspace and hspace
    gs = plt.GridSpec(5, 2, wspace=0.2, hspace=0.2) 
    # Create a figure
    fig = plt.figure(figsize=(11, 15))

    # SUBFIGURE 1
    # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Womersley Velocity profile', fontsize=20)
    ax1.set_xlabel('velocity (m/s)', fontsize=20)
    ax1.set_ylabel('Radius, (m)', fontsize=20)
    #ax1.plot(np.abs(u)/100., r)
    ax1.plot(u, r)
    ax1.plot(u[100], r[100], 'ro', markersize=10, label = "{0:2.3e}".format(u[100]))
    ax1.legend()
    #ax1.plot(np.abs(u)[::-1], -1.0*r[::-1])
    # Arrange axis and labels
    
    
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('Common Carotid Artery', fontsize=20)
    #ax2.set_xlabel(r'$t$', fontsize=20)
    ax2.set_ylabel(r'Peak, $V(t)$ $m/s$', fontsize=20)
    
    Q = womersley.utils.reconstruct_pt(coef_v, time, T)
    Q2, t2 = womersley.utils.reconstruct2(coef_v, T)
    ax2.plot(t2, np.real(Q2))
    ax2.plot(time, np.real(Q), 'ro', markersize=10, label = "{0:2.3e}".format(Q.real))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.legend()
    ax1.axis([-0.06, 1.0,-R,R])
    ax2.set_xlim((0, T))
    
    ax3 = fig.add_subplot(gs[2, :])
    #ax3.set_title('Common Carotid Artery')
    #ax3.set_xlabel(r'$t$', fontsize=20)
    ax3.set_ylabel(r'Mean, $\bar{V}(t)$ $m/s$', fontsize=20)
    
    mean = womersley.utils.reconstruct_pt(coef_mean, time, T)
    mean3, t3 = womersley.utils.reconstruct2(coef_mean, T)
    ax3.plot(t3, np.real(mean3))
    ax3.plot(time, np.real(mean), 'ro', markersize=10, label = "{0:2.3e}".format(mean.real))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.legend()
    ax3.set_xlim((0, T))
    
    ax4 = fig.add_subplot(gs[3, :])
    #ax4.set_title('Common Carotid Artery')
    ax4.set_xlabel(r'$t$', fontsize=20)
    ax4.set_ylabel(r'Shear, $\tau_{wall}$ $Pa$', fontsize=20)
    
    shear = womersley.utils.reconstruct_pt(coef_shear, time, T)
    shear4, t4 = womersley.utils.reconstruct2(coef_shear, T)
    ax4.plot(t4, np.real(shear4))
    ax4.plot(time, np.real(shear), 'ro', markersize=10, label = "{0:2.3e}".format(shear.real))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.legend()
    ax4.set_xlim((0, T))
    
    ax5 = fig.add_subplot(gs[4, :])
    #ax4.set_title('Common Carotid Artery')
    ax5.set_xlabel(r'$t$', fontsize=20)
    ax5.set_ylabel(r'Pressure, $P$ $Pa$', fontsize=20)
    
    press = womersley.utils.reconstruct_pt(coef_p, time, T)
    press5, t5 = womersley.utils.reconstruct2(coef_p, T)
    ax5.plot(t5, np.real(press5))
    ax5.plot(time, np.real(press), 'ro', markersize=10, label = "{0:2.3e}".format(press.real))
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax5.legend()
    ax5.set_xlim((0, T))
    
    #ensure all subplots fit bounding box
    gs.tight_layout(fig)
    
    
# this converts the coefficients from the centerline flow velocity using the above function
# beware it would be different from using a  the mean velocity
def plot_womersley_mean(waveform_obj, coef, time=0.0):
    
    coef_p = coef
    
    T = waveform_obj.get_period()
    omega = (2.0*np.pi)/T
    R = waveform_obj.radius # meters
    mu = waveform_obj.mu #Pa s
    rho = waveform_obj.rho #kg/m**3
    
    coeff_inst = coefficient_converter(waveform_obj, coef_p, "profile")
    
    coef_peak = coeff_inst.coef_dict["ctr"]
    coef_mean = coeff_inst.coef_dict["mean"] #peak_2_mean(coef_v, rho, omega, mu, R)
    coef_v = coeff_inst.coef_dict["ctr"] #peak_2_press(coef_v, rho, omega, mu, R)
    
    # Lets create some data to visualize
    r = np.linspace(-R, R, 200)
    t = np.linspace(0.0, T, 300)
    # convert velocity coefficients to pressure coefficients
    

    #print(u)
    # Lets create a plotting grid with gridspec
    # Create a 2 x 2 plotting grid object with horizontal and vertical spacing wspace and hspace
    gs = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2) 
    # Create a figure
    fig = plt.figure(figsize=(11, 13))

    
    ax3 = fig.add_subplot(gs[0, :])
    #ax3.set_title('Common Carotid Artery')
    #ax3.set_xlabel(r'$t$', fontsize=20)
    ax3.set_ylabel(r'Mean, $\bar{V}(t)$ $m/s$', fontsize=20)
    
    mean = womersley.utils.reconstruct_pt(coef_mean, time, T)
    mean3, t3 = womersley.utils.reconstruct2(coef_mean, T, 1000)
    Q_flow = np.pi*R**2*mean3
    Q_total = np.trapz(Q_flow,t3)
    
    waveform_obj.set_avg_Q(Q_total.real/T)
    waveform_obj.set_peak_Q(np.max(Q_flow.real))
    waveform_obj.set_min_Q(np.min(Q_flow.real))
    
    
    print('Q, Area under curve: {0} $m^3/cycle$'.format(Q_total.real))
    print('Q average: {0} $m^3/s$'.format(waveform_obj.get_avg_Q()))
    print('Q peak, {0} Q min, {1} $m^3/s$'.format(waveform_obj.get_peak_Q(), waveform_obj.get_min_Q()))

    q_pt = np.pi*R**2*mean
    ax3.plot(t3, np.real(mean3))
    ax3.plot(time, np.real(mean), 'ro', markersize=10, label = "{0:2.2e}".format(mean.real))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.legend()
    ax3.set_xlim((0, T))
    
    ax4 = fig.add_subplot(gs[1, :])
    #ax4.set_title('Common Carotid Artery')
    ax4.set_xlabel(r'$t$', fontsize=20)
    ax4.set_ylabel(r'Q, $Q(t)$ $ml/min$', fontsize=20)
    ax4.plot(t3, np.real(Q_flow)*10.**6*60.)
    ax4.plot(time, np.real(q_pt)*10.**6*60., 'ro', markersize=10,
             label = "{0:2.2e}".format(q_pt.real*10.**6*60.))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.legend()
    ax4.set_xlim((0, T))
    
    ax5 = fig.add_subplot(gs[2, :])
    #ax4.set_title('Common Carotid Artery')
    ax5.set_xlabel(r'$t$', fontsize=20)
    ax5.set_ylabel(r'Pressure, $P$ $Pa$', fontsize=20)
    
    press = womersley.utils.reconstruct_pt(coef_p, time, T)
    press5, t5 = womersley.utils.reconstruct2(coef_p, T)
    dP = np.max(press5.real) - np.min(press5.real)
    #print(np.max(press5.real), np.min(press5.real))
    print('dP: {0} $Pa/s$'.format(dP/T))
    ax5.plot(t5, np.real(press5))
    ax5.plot(time, np.real(press), 'ro', markersize=10, label = "{0:2.2e}".format(press.real))
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax5.legend()
    ax5.set_xlim((0, T))
    
    
    #ensure all subplots fit bounding box
    gs.tight_layout(fig)

# this converts the coefficients from the centerline flow velocity using the above function
# beware it would be different from using a  the mean velocity
def plot_womersley_mean_norm(waveform_obj, time=0.0):

    
    T = waveform_obj.get_period()
    omega = (2.0*np.pi)/T
    R = waveform_obj.radius # meters
    mu = waveform_obj.mu #Pa s
    rho = waveform_obj.rho #kg/m**3
    
    coeff_inst = converter_from_norm(waveform_obj)
    
    coef_peak = coeff_inst.coef_dict["ctr"]
    coef_mean = coeff_inst.coef_dict["mean"] #peak_2_mean(coef_v, rho, omega, mu, R)
    coef_v = coeff_inst.coef_dict["ctr"] #peak_2_press(coef_v, rho, omega, mu, R)
    coef_p = coeff_inst.coef_dict["press"]
    # Lets create some data to visualize
    r = np.linspace(-R, R, 200)
    t = np.linspace(0.0, T, 300)
    # convert velocity coefficients to pressure coefficients
    

    #print(u)
    # Lets create a plotting grid with gridspec
    # Create a 2 x 2 plotting grid object with horizontal and vertical spacing wspace and hspace
    gs = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2) 
    # Create a figure
    fig = plt.figure(figsize=(11, 13))

    
    ax3 = fig.add_subplot(gs[0, :])
    #ax3.set_title('Common Carotid Artery')
    #ax3.set_xlabel(r'$t$', fontsize=20)
    ax3.set_ylabel(r'Mean, $\bar{V}(t)$ $m/s$', fontsize=20)
    
    mean = womersley.utils.reconstruct_pt(coef_mean, time, T)
    mean3, t3 = womersley.utils.reconstruct2(coef_mean, T, 1000)
    Q_flow = np.pi*R**2*mean3
    Q_total = np.trapz(Q_flow,t3)
    
    waveform_obj.set_avg_Q(Q_total.real/T)
    waveform_obj.set_peak_Q(np.max(Q_flow.real))
    waveform_obj.set_min_Q(np.min(Q_flow.real))
    
    
    print('Q, Area under curve: {0} $ml/cycle$'.format(Q_total.real*10**6*60))
    print('Q average: {0} $ml/min$'.format(waveform_obj.get_avg_Q()*10**6*60))
    print('Q peak, {0} Q min, {1} $ml/min$'.format(waveform_obj.get_peak_Q()*10**6*60, waveform_obj.get_min_Q()*10**6*60))

    q_pt = np.pi*R**2*mean
    ax3.plot(t3, np.real(mean3))
    ax3.plot(time, np.real(mean), 'ro', markersize=10, label = "{0:2.2e}".format(mean.real))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.legend()
    ax3.set_xlim((0, T))
    
    ax4 = fig.add_subplot(gs[1, :])
    #ax4.set_title('Common Carotid Artery')
    ax4.set_xlabel(r'$t$', fontsize=20)
    ax4.set_ylabel(r'Q, $Q(t)$ $ml/min$', fontsize=20)
    ax4.plot(t3, np.real(Q_flow)*10.**6*60.)
    ax4.plot(time, np.real(q_pt)*10.**6*60., 'ro', markersize=10,
             label = "{0:2.2e}".format(q_pt.real*10.**6*60.))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.legend()
    ax4.set_xlim((0, T))
    
    ax5 = fig.add_subplot(gs[2, :])
    #ax4.set_title('Common Carotid Artery')
    ax5.set_xlabel(r'$t$', fontsize=20)
    ax5.set_ylabel(r'Pressure, $P$ $Pa$', fontsize=20)
    
    press = womersley.utils.reconstruct_pt(coef_p, time, T)
    press5, t5 = womersley.utils.reconstruct2(coef_p, T)
    dP = np.max(press5.real) - np.min(press5.real)
    #print(np.max(press5.real), np.min(press5.real))
    print('dP: {0} $Pa/s$'.format(dP/T))
    ax5.plot(t5, np.real(press5))
    ax5.plot(time, np.real(press), 'ro', markersize=10, label = "{0:2.2e}".format(press.real))
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax5.legend()
    ax5.set_xlim((0, T))
    
    
    #ensure all subplots fit bounding box
    gs.tight_layout(fig)
