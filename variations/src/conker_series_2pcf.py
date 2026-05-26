# =========================================================================== #
"""
Created in April 2025
@authors: Hugh Randall

Description: This file, conker_series_multipole.py is a modified version of conker_series.py created by Zachery Brown. The majority of the changes are comments which skip some of the loops in the legendre expansion. This version of the code is meant for computing the multipoles of the 2pcf where we do not need to compute the m != 0 terms. Nothing is changed in regards to how the algorithm does it's calculations, simply a restriction on how many things are calculated for the purpose of reducing computational time.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u

# Required scipy functions
from scipy.signal import fftconvolve

# Required python imports
import time
import json
import os

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# =================== CONKER CONVOLVE PER PARTITION ========================= #


class ConKerBox:
    # Convolves the selected kernels for one of the partitioned regions
    # Reqs data catalog, randoms catalog, cfg file, and the box number
    # Will only work if the DivideScheme file has been saved!
    # Designed to run in series
    # TODO -> Multiprocessing options in future versions
    
    
    def __init__(self, data_file: str, rand_file: str, cfg_file: bool,
                 store_rand: str, box_idx: int, verbose: bool, use_oqe=False,
                 data_file2=None, rand_file2=None):
        # Reads in a data, randoms, and cfg file
        # Sets up print statemetns if inital box
        # Accounts for timing
        
        # Determines if the randoms need to be saved
        self.rand_save = store_rand
        
        # Set the data and randoms file string 
        self.datafileString = u.getFileString(data_file, data_file2)
        self.randfileString = u.getFileString(rand_file, rand_file2)
        # if data_file2==None:
        #     self.datafileString = data_file.split('.fits')[0]
        #     if self.rand_save:
        #         self.randfileString = rand_file.split('.fits')[0]
        # else:
        #     # concatenate the filenames
        #     self.datafileString = data_file.split('.fits')[0].split('.')[0] +\
        #                             '_' + data_file2.split('.fits')[0]
        #     if self.rand_save:
        #         self.randfileString = rand_file.split('.fits')[0].split('.')[0] +\
        #                                 '_' + rand_file2.split('.fits')[0]
        
        # PS if initial box to mark the start of the routine
        if box_idx == 0:
            print('\n'+'===================== '+'CONKER CONVOLUTION STEP'+
                  ' ====================='+'\n')
            print('Convolving kernels with catalog '+self.datafileString)
            
            # PS verb to display the catalog filenames and cfg file
            if verbose:
                print('Preparing to measure correlation functions...')
                print('Data file: '+data_file.split('.fits')[0])
                print('Randoms file: '+rand_file.split('.fits')[0])
                print('Configuration: '+cfg_file.split('.txt')[0])
                
        # Grabs info from the divide plan
        # Not failable at this stage if it has already been
        #   cleared by the driver
        self.LOS,self.conv_box_lims,self.map_box_lims,self.total_boxes,\
            self.shift_condition=u.getLOSbox(rand_file, cfg_file, box_idx, rand_file2)
            
        # Reads in the data, randoms, and cfg names
        self.data_file = data_file
        self.rand_file = rand_file
        self.data_file2 = data_file2
        self.rand_file2 = rand_file2
        self.cfg_file = cfg_file
        
        # Sets box index
        self.box_idx = box_idx

        # Sets verbose
        self.verbose = verbose
        
        # Stores parameters from the cfg file
        with open('./params/'+self.cfg_file) as cfgDict:
            cfg_set = json.load(cfgDict)
        self.cosmo = (cfg_set['c'],cfg_set['H0'],
                      (cfg_set['OmM'],0.,1.-cfg_set['OmM']))
        
        # Sets the s-bins using the cfg file
        # Uses the s-bins to store the grid size g_S
        self.desired_s_bin_centers = np.linspace(
            cfg_set['sMin'],cfg_set['sMax'],cfg_set['sBinN'])
        self.desired_s_bin_edges = u.cen2edge(self.desired_s_bin_centers)
        self.g_s = self.desired_s_bin_centers[1]-self.desired_s_bin_centers[0]
        
        # Set the rounding precision (very important paramter for this step!)
        self.rounding_pr = cfg_set['pR']
        
        # # Determine if the density field will be weighted
        # self.wtd = cfg_set['wtd']
        
        # Starts the timing
        self.start_map_time = time.perf_counter()
        
        # Read in the data and randoms 
        with fits.open('./data/'+self.data_file) as f:
            data_full1 = f[1].data
        with fits.open('./data/'+self.rand_file) as f:
            rand_full1 = f[1].data
            rand_cols1 = f[1].columns

        if self.data_file2 == None:
            data_full2 = data_full1
            rand_full2 = rand_full1
        else:
            with fits.open('./data/'+self.data_file2) as f:
                data_full2 = f[1].data
            with fits.open('./data/'+self.rand_file2) as f:
                rand_full2 = f[1].data

            nrows1 = rand_full1.shape[0]
            nrows = nrows1 + rand_full2.shape[0]
            hdu = fits.BinTableHDU.from_columns(rand_cols1, nrows=nrows)
            for colname in rand_cols1.names:
                hdu.data[colname][nrows1:] = rand_full2[colname]
            rand_full_comb = hdu.data
            del hdu
            
        # Sets the readout values for the LOS (ignores numerical shift)
        self.LOS_ra_readout = np.asarray([self.LOS[0]])
        
        # if self.wtd == False:
        #     # If weights are not goint to be used
            
        #     # Create arrays of weight 1 for data and randoms
        #     data_full['wts'] = np.ones(len(data_full),dtype='float')
        #     rand_full['wts'] = np.ones(len(rand_full),dtype='float')
            
        if self.shift_condition == True:
            # If the catalog is shifted (SGC)
            
            # Shift the RA values to the other side of the sky
            data_full1['ra'] = u.shift_RA(data_full1['ra'])
            rand_full1['ra'] = u.shift_RA(rand_full1['ra'])

            if self.data_file2 is not None:
                data_full2['ra'] = u.shift_RA(data_full2['ra'])
                rand_full2['ra'] = u.shift_RA(rand_full2['ra'])
                rand_full_comb['ra'] = u.shift_RA(rand_full_comb['ra'])
            
            # Fix the readout value for the print statements
            self.LOS_ra_readout = u.shift_RA(np.asarray([self.LOS[0]]))
            
        # Map the data and randoms to four boxes
        # Two for convolution (inner) regions, two for mapping (outer) regions
        # Outer region first 
        self.data = u.coordBoxSlice(data_full2,self.map_box_lims)
        self.rand = u.coordBoxSlice(rand_full2,self.map_box_lims)
        
        # Inner region next
        # The "_0" flag marks the convolution (inner) region
        self.data_0 = u.coordBoxSlice(data_full1,self.conv_box_lims)
        self.rand_0 = u.coordBoxSlice(rand_full1,self.conv_box_lims)
        
        if self.data_file2 is not None:
            self.rand_comb = u.coordBoxSlice(rand_full_comb,self.map_box_lims)
            del rand_full_comb
        
        # Get the nD and nR normalization values for the full catalogs
        # NOTE This choice of normalization will likely be updated.
        if use_oqe:
            weights = ['wts', 'wtilde', 'w0', 'w2']
        else:
            weights = ['wts']
        self.nDnR_0 = {}
        self.nDnR = {}
        for weight in weights:
            self.nDnR_0[weight] = (np.sum(data_full1[weight]), np.sum(rand_full1[weight]))
            self.nDnR[weight] = (np.sum(data_full2[weight]), np.sum(rand_full2[weight]))
        
        # Delete the large fits catalogs
        del data_full1, rand_full1, data_full2, rand_full2
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb shows the limits of the mapping and convolution regions
            if self.verbose:
                print('Initial LOS at (RA: '+
                      str(np.round(self.LOS_ra_readout[0],decimals=2))+
                      ', DEC: '+str(np.round(self.LOS[1],decimals=2))+')')
                print('Initial mapping box is '+
                      str(np.round(self.map_box_lims[0][1]-
                                   self.map_box_lims[0][0],decimals=2))+
                      ' deg. by '+str(np.round(self.map_box_lims[1][1]-
                                               self.map_box_lims[1][0],
                                               decimals=2))+' deg.')
                print('Initial convolution box is '+
                      str(np.round(self.conv_box_lims[0][1]-
                                   self.conv_box_lims[0][0],decimals=2))+
                      ' deg. by '+str(np.round(self.conv_box_lims[1][1]-
                                               self.conv_box_lims[1][0],
                                               decimals=2))+' deg.')
     
        
    def radialConvolveLegendre(self, ell_max: int, ftype: str = 'fits', use_moments=False, use_oqe=False):
        # A function to convolve the density field with 
        #   Y_ell_max^_ell to Y_-ell_max^ell kernels

        if use_moments:
            ftype = 'fits'
            centers = np.load('./params/centers.npy')
            edges = np.load('./params/edges.npy')
            
            try:
                os.makedirs('./moments/'+ self.datafileString+'_'+self.cfg_file.split('.txt')[0])
            except:
                FileExistsError
        
        # Store ell_max and define the ell_steps
        self.ell_max = ell_max
        ell_step = np.linspace(0,ell_max,ell_max+1,dtype=int)
        self.ell_step = ell_step[ ell_step % 2 == 0 ]

        if use_oqe:
            wts1_dict = {0:'wtilde', 2:'wtilde', 4:'wts'}
            wts2_dict = {0:'w0', 2:'w2', 4:'wts'}
        else:
            # wts1 = {0:'wts', 2:'wts', 4:'wts'}
            # wts2 = {0:'wts', 2:'wts', 4:'wts'}
            wts1 = 'wts'
            wts2 = 'wts'
        
        # Map the LUT radii and redshifts, padding by 0.01
        if self.data_file2 == None:
            LUT_radii, LUT_redshifts = u.interpolate_r_z(
                self.rand['z'].min()-0.01, self.rand['z'].max()+0.01, self.cosmo)
        else:   
            LUT_radii, LUT_redshifts = u.interpolate_r_z(
                self.rand_comb['z'].min()-0.01, self.rand_comb['z'].max()+0.01, self.cosmo)
        
        # Define the data and randoms XYZ coordinates
        # O(N) operation 
        # Outer region first 
        # Coordinates are transformed to local cartesian wrt the LOS
        data_XYZ = np.array(u.sky2localCart((self.data['ra'],self.data['dec'],
                                             LUT_radii(self.data['z'])),
                                             self.LOS)).T
        rand_XYZ = np.array(u.sky2localCart((self.rand['ra'],self.rand['dec'],
                                             LUT_radii(self.rand['z'])),
                                             self.LOS)).T
        if self.data_file2 is not None:
            rand_comb_XYZ = np.array(u.sky2localCart((self.rand_comb['ra'],self.rand_comb['dec'],
                                                 LUT_radii(self.rand_comb['z'])),
                                                 self.LOS)).T
        
        # Inner region next
        # The "_0" flag marks the convolution (inner) region
        data_XYZ_0 = np.array(u.sky2localCart((self.data_0['ra'],self.data_0['dec'],
                                             LUT_radii(self.data_0['z'])),
                                             self.LOS)).T
        rand_XYZ_0 = np.array(u.sky2localCart((self.rand_0['ra'],self.rand_0['dec'],
                                             LUT_radii(self.rand_0['z'])),
                                             self.LOS)).T
        
        # Define the centers and edges of the mapping (outer) box cells
        # Bins will be used to map the inner region as well
        if self.data_file2 == None:
            self.grid_edges = [
                u.makeBinEdges((rand_XYZ.T[0].min()-self.g_s,
                                rand_XYZ.T[0].max()+self.g_s),self.g_s),
                u.makeBinEdges((rand_XYZ.T[1].min()-self.g_s,
                                rand_XYZ.T[1].max()+self.g_s),self.g_s),
                u.makeBinEdges((rand_XYZ.T[2].min()-self.g_s,
                                rand_XYZ.T[2].max()+self.g_s),self.g_s)]
        else:
            self.grid_edges = [
                u.makeBinEdges((rand_comb_XYZ.T[0].min()-self.g_s,
                                rand_comb_XYZ.T[0].max()+self.g_s),self.g_s),
                u.makeBinEdges((rand_comb_XYZ.T[1].min()-self.g_s,
                                rand_comb_XYZ.T[1].max()+self.g_s),self.g_s),
                u.makeBinEdges((rand_comb_XYZ.T[2].min()-self.g_s,
                                rand_comb_XYZ.T[2].max()+self.g_s),self.g_s)]
            del rand_comb_XYZ, self.rand_comb

        self.grid_centers = [
            u.edge2cen(np.asarray(self.grid_edges[0])),
            u.edge2cen(np.asarray(self.grid_edges[1])),
            u.edge2cen(np.asarray(self.grid_edges[2]))]
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark successful coordinate transformation
            if self.verbose:
                print('Successful coordinate transformation...')

        if not use_oqe:
            # Map the data galaxies to a grid (NGP method)
            self.D_g_0 = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,weights=self.data_0[wts1])[0]
            
            # PS if initial box
            if self.box_idx == 0:
                
                # PS verb to mark successful data histogram
                if self.verbose:
                    print('Successful data histogram...')
                    
            # Map the random galaxies to a grid 
            # Normalize this grid to the overall sum of data weights
            self.R_g_0 = (self.nDnR_0[wts1][0]/self.nDnR_0[wts1][1])*np.histogramdd(
                rand_XYZ_0,bins=self.grid_edges,weights=self.rand_0[wts1])[0]
            
            # PS if initial box
            if self.box_idx == 0:
                
                # PS verb to mark successful randoms histogram
                if self.verbose:
                    print('Successful randoms histogram...')
                
        # If initial box, create temporary directory for files
        if self.box_idx == 0:
            
            try:
                # Try to make an appropriate directory to store the grids
                # Tag it with the data and cfg names
                os.makedirs('./grids/'+self.datafileString+'_'
                            +self.cfg_file.split('.txt')[0])
                
            except FileExistsError:
                # If the directory already exists
                
                # Trigger a failure
                # This prevents overwriting grid data by mistake
                print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                      ' ==!==!==!==!==!==!==!==!==!=='+'\n')
                print('You already have a temporary directory for this '+
                      'catalog and cfg!')
                print("Check ./grids/ to make sure you don't have "+
                      "existing data")
                print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                      '!==!==!==!==!==!==!==!==!==!==!'+'\n')
                return
            
            if self.rand_save:
                try:
                    # Try to make an appropriate dir to store the randoms
                    # Tag it with the randoms and cfg names
                    os.makedirs('./grids/'+self.randfileString+'_'
                                +self.cfg_file.split('.txt')[0])
                    
                except FileExistsError:
                    # If the directory already exists
                    
                    # Trigger a failure
                    # This prevents overwriting grid data by mistake
                    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
                    print('You already have a temporary directory for this '+
                          'catalog and cfg!')
                    print("Check ./grids/ to make sure you don't have "+
                          "existing data")
                    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
                    return
        
        # Set the name for the newly created directory
        tempDir = './grids/'+self.datafileString+'_'+\
            self.cfg_file.split('.txt')[0]+'/'
            
        # Set the name of a newly created directory for randoms
        if self.rand_save:
            tempDirRand = './grids/'+self.randfileString+'_'+\
                self.cfg_file.split('.txt')[0]+'/'
        
        # End the mapping time stamp
        self.end_map_time = time.perf_counter()
        
        # Initialize a value for the file writing time
        self.file_time = 0.

        if not use_oqe:
            #Start file writing time
            file_start_time_B = time.perf_counter()
            
            # Make a mask to reduce future file sizes
            # Data mask
            data_mask = u.make_grid_mask(self.D_g_0)
            
            # Randoms mask
            rand_mask = u.make_grid_mask(self.R_g_0)
            
            # Combined mask
            self.mask = u.make_grid_mask(data_mask+rand_mask)
            
            # Delete the individual masks for data and randoms
            del data_mask, rand_mask
            
            if ftype == 'fits':
                # If the requested grid filetype is fits
                            
                # Write the W0 and B0 grids (inner)
                # These values correspond to the convolution regions
                u.grid_to_fits_wrapper(self.D_g_0-self.R_g_0,self.mask,
                                       tempDir+'W_p{}_of_{}.fits'.format(
                                           self.box_idx+1,self.total_boxes, wts1))
                if self.rand_save:
                    u.grid_to_fits_wrapper((self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0,self.mask,
                                           tempDirRand+'B_p{}_of_{}.fits'.format(
                                               self.box_idx+1,self.total_boxes, wts1))
            if (ftype == 'npy')|use_moments:
                # If the requested grid filetype is npy
                
                # Write the W0 and B0 grids (inner)
                # These values correspond to the convolution regions
                if ftype == 'npy':
                    np.save(tempDir+'W_p{}_of_{}.npy'.format(
                        self.box_idx+1,self.total_boxes, wts1),self.mask*(
                            self.D_g_0-self.R_g_0))
                    if self.rand_save:
                        np.save(tempDirRand+'B_p{}_of_{}.npy'.format(
                            self.box_idx+1,self.total_boxes, wts1),
                            (self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0*self.mask)
                if self.rand_save & use_moments:
                    np.save(tempDirRand+'B_p{}_of_{}.npy'.format(
                            self.box_idx+1,self.total_boxes, wts1),
                            (self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0*self.mask)
                
            # End file writing time and update
            file_end_time_B = time.perf_counter()
            self.file_time += file_end_time_B - file_start_time_B
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark W0 and B0 files written
            if self.verbose:
                print('Wrote initial region grid files...')
                
        # Initialize kernel construction time
        self.kernel_time = 0.
        
        # Initialize convolution time
        self.conv_time = 0.

        for ell_idx in range(len(self.ell_step)):
            # Loop through all ell steps
            
            if use_moments:
                norm = (2*self.ell_step[ell_idx]+1)*(
                        u.ylm_norm_m0(0)/u.ylm_norm_m0(self.ell_step[ell_idx]))
            
            if use_oqe:
                wts1 = wts1_dict[self.ell_step[ell_idx]]
                wts2 = wts2_dict[self.ell_step[ell_idx]]
                # Map the data galaxies to a grid (NGP method)
                self.D_g_0 = np.histogramdd(data_XYZ_0,bins=self.grid_edges,
                                                weights=self.data_0[wts1])[0]
                # Map the random galaxies to a grid 
                # Normalize this grid to the overall sum of data weights
                self.R_g_0 = (self.nDnR_0[wts1][0]/self.nDnR_0[wts1][1])*\
                                np.histogramdd(rand_XYZ_0,
                                               bins=self.grid_edges,
                                               weights=self.rand_0[wts1])[0]
                if ell_idx==0:
                    # the mask only cares about nonzero elements, 
                    # weighting won't affect the mask
                    # Make a mask to reduce future file sizes
                    data_mask = u.make_grid_mask(self.D_g_0)
                    
                    # Randoms mask
                    rand_mask = u.make_grid_mask(self.R_g_0)
                    
                    # Combined mask
                    self.mask = u.make_grid_mask(data_mask+rand_mask)
            
                    # Delete the individual masks for data and randoms
                    del data_mask, rand_mask
                # Start file writing time
                file_start_time_B = time.perf_counter()  
                
                if ftype == 'fits':
                # If the requested grid filetype is fits   
                    # Write the W0 and B0 grids (inner)
                    # These values correspond to the convolution regions
                    u.grid_to_fits_wrapper(self.D_g_0-self.R_g_0,self.mask,
                                           tempDir+'W_w{}_p{}_of_{}.fits'.format(
                                               self.ell_step[ell_idx], self.box_idx+1, self.total_boxes))
                    if self.rand_save:
                        u.grid_to_fits_wrapper((self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0,self.mask,
                                               tempDirRand+'B_w{}_p{}_of_{}.fits'.format(
                                                   self.ell_step[ell_idx], self.box_idx+1, self.total_boxes))
                if (ftype == 'npy')|use_moments:
                    # If the requested grid filetype is npy
                    
                    # Write the W0 and B0 grids (inner)
                    # These values correspond to the convolution regions
                    if ftype == 'npy':
                        np.save(tempDir+'W_w{}_p{}_of_{}.npy'.format(
                                    self.ell_step[ell_idx], self.box_idx+1, self.total_boxes),
                                self.mask*(self.D_g_0-self.R_g_0))
                        if self.rand_save:
                            np.save(tempDirRand+'B_w{}_p{}_of_{}.npy'.format(
                                        self.ell_step[ell_idx], self.box_idx+1, self.total_boxes),
                                    (self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0*self.mask)
                    if (self.rand_save) & use_moments:
                        np.save(tempDirRand+'B_w{}_p{}_of_{}.npy'.format(
                                    self.ell_step[ell_idx], self.box_idx+1, self.total_boxes),
                                (self.nDnR_0[wts1][1]/self.nDnR_0[wts1][0])*self.R_g_0*self.mask)
                # End file writing time and update
                file_end_time_B = time.perf_counter()
                self.file_time += file_end_time_B - file_start_time_B


                self.D_g = np.histogramdd(data_XYZ,bins=self.grid_edges,
                                              weights=self.data[wts2])[0]
                self.R_g = ((self.nDnR[wts2][0]/self.nDnR[wts2][1]))*\
                                np.histogramdd(rand_XYZ,
                                               bins=self.grid_edges,
                                               weights=self.rand[wts2])[0]
            else:
                self.D_g = np.histogramdd(data_XYZ,bins=self.grid_edges,
                                          weights=self.data[wts2])[0]
                self.R_g = ((self.nDnR[wts2][0]/self.nDnR[wts2][1]))*\
                                np.histogramdd(rand_XYZ,
                                               bins=self.grid_edges,
                                               weights=self.rand[wts2])[0]
            
            for s_idx in range(len(self.desired_s_bin_centers)):
                # Loop through all radial steps
                
                m = 0                
                # Make kernel if this is partition 0
                if self.box_idx == 0:
                    
                    # Starts the kernel timer
                    kernel_start_time_W = time.perf_counter()
                    
                    # Creates the kernel and defines the REAL and IMAG
                    #   grids corresponding to the ylm functions
                    kern_grid_RE, _ = u.ylmKernel(
                        self.desired_s_bin_centers[s_idx],self.g_s,
                        m_=m,n_=self.ell_step[ell_idx])

                    kernel_end_time_W = time.perf_counter()
                    
                    self.kernel_time += kernel_end_time_W -\
                        kernel_start_time_W
                        
                    # Start the file timer
                    file_start_time_W = time.perf_counter()
                    
                    # Write the kernel grid(s) to file
                    # REAL kernel first
                    np.save(tempDir+
                            'K_{}_{}_{}_RE.npy'.format(
                                s_idx,self.ell_step[ell_idx],
                                m),kern_grid_RE)
                        
                    # End file timer and update
                    file_end_time_W = time.perf_counter()
                    self.file_time += file_end_time_W - file_start_time_W
                    
                # Otherwise load kernel(s)
                elif self.box_idx != 0:
                    
                    # Start the file timer
                    file_start_time_W = time.perf_counter()
                    
                    # Load the REAL kernel
                    kern_grid_RE = np.load(
                        tempDir+'K_{}_{}_{}_RE.npy'.format(
                            s_idx,self.ell_step[ell_idx],
                            m))
                        
                    # End file timer and update
                    file_end_time_W = time.perf_counter()
                    self.file_time += file_end_time_W - file_start_time_W
                        
                # Start convolution timer
                conv_start_time = time.perf_counter()
                
                # Convolve with the density grid (outer region)
                # Mask for reduction of outside regions
                W_i_ell_m_RE = self.mask*np.round(fftconvolve(
                    self.D_g-self.R_g,kern_grid_RE,mode='same'),
                    decimals=self.rounding_pr)
                # if self.rand_save:
                #     B_i_0_0_RE_well = self.mask*np.round(fftconvolve(
                #         self.R_g,kern_grid_RE_ran,mode='same'),
                #         decimals=self.rounding_pr)
                    
                # End convolution timer and update
                conv_end_time = time.perf_counter()
                self.conv_time += conv_end_time - conv_start_time
                
                # Start file timer
                file_start_time_W = time.perf_counter()
                
                if ftype == 'fits':
                    # If the requested grid filetype is fits
                    
                    # Write the REAL W and B grids
                    u.grid_to_fits_wrapper(
                        W_i_ell_m_RE,self.mask,
                        tempDir+'W_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                            s_idx,self.ell_step[ell_idx],m,
                            self.box_idx+1,self.total_boxes))
                    # if self.rand_save:
                    #     u.grid_to_fits_wrapper(
                    #         ((self.nDnR[w2_temp][1]/self.nDnR[w2_temp][0]))*B_i_0_0_RE_well,
                    #         self.mask,tempDirRand+
                    #         'B_{}_0_0_RE_w{}_p{}_of_{}.fits'.format(
                    #             s_idx,self.ell_step[ell_idx],
                    #             self.box_idx+1,self.total_boxes))
                    
                elif ftype == 'npy':
                    # If the requested grid filetype is npy
                    
                    # Write the REAL W and B grids
                    np.save(tempDir+'W_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                        s_idx,self.ell_step[ell_idx],m,
                        self.box_idx+1,self.total_boxes),
                        W_i_ell_m_RE*self.mask)
                    # if self.rand_save:
                    #     np.save(tempDirRand+
                    #             'B_{}_0_0_RE_w{}_p{}_of_{}.fits'.format(
                    #             s_idx,self.ell_step[ell_idx],
                    #             self.box_idx+1,self.total_boxes),
                    #         ((self.nDnR[w2_temp][1]/self.nDnR[w2_temp][0]))*B_i_0_0_RE_well*self.mask)
                        
                # End file timer and update
                file_end_time_W = time.perf_counter()
                self.file_time += file_end_time_W - file_start_time_W

                if self.rand_save:
                    if not use_oqe:
                    # if just FKP then only need monopole with one set of weights
                        if (ell_idx == 0):
                            # Loading kernel
                            kern_grid_RE_ran = kern_grid_RE.copy()

                            # convolving with kernel
                            B_i_0_0_RE_well = self.mask*np.round(fftconvolve(
                                self.R_g,kern_grid_RE_ran,mode='same'),
                                decimals=self.rounding_pr)

                            # Start file timer
                            file_start_time_W = time.perf_counter()
                            
                            # saving grids
                            if (ftype == 'fits'):
                                # If the requested grid filetype is fits
                                # Write the REAL B grids
                                u.grid_to_fits_wrapper(
                                    ((self.nDnR[wts2][1]/self.nDnR[wts2][0]))*B_i_0_0_RE_well,
                                    self.mask,tempDirRand+
                                    'B_{}_0_0_RE_p{}_of_{}.fits'.format(
                                        s_idx, self.box_idx+1, self.total_boxes))
                    
                            if (ftype == 'npy')|use_moments:
                                # If the requested grid filetype is npy
                                # Write the REAL  B grids
                                np.save(tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                                            s_idx, self.box_idx+1,self.total_boxes),
                                        ((self.nDnR[wts2][1]/self.nDnR[wts2][0]))*B_i_0_0_RE_well*self.mask)
                            file_end_time_W = time.perf_counter()
                            self.file_time += file_end_time_W - file_start_time_W
                        else:
                            if use_moments:
                                # Start file timer
                                file_start_time_W = time.perf_counter()
                                # if using moments but ell!=0 we still need the grid to define gamma 
                                # no fits choice because ftype must be npy
                                B_i_0_0_RE_well = (self.nDnR[wts2][0]/self.nDnR[wts2][1])*np.load(
                                                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                                                s_idx,self.box_idx+1,self.total_boxes))
                                file_end_time_W = time.perf_counter()
                                self.file_time += file_end_time_W - file_start_time_W
                            else:
                                continue
                    else:
                    # then OQE and we need monopole B with all weights
                        # Loading kernel
                        if (ell_idx == 0):
                            kern_grid_RE_ran = kern_grid_RE.copy()
                        else:
                            file_start_time_W = time.perf_counter()
                            kern_grid_RE_ran = np.load(tempDir+'K_{}_{}_{}_RE.npy'.format(
                                                   s_idx, 0, 0))
                            file_end_time_W = time.perf_counter()
                            self.file_time += file_end_time_W - file_start_time_W

                        # convolving with kernel
                        B_i_0_0_RE_well = self.mask*np.round(fftconvolve(
                            self.R_g,kern_grid_RE_ran,mode='same'),
                            decimals=self.rounding_pr)

                        # Start file timer
                        file_start_time_W = time.perf_counter()
                        # saving grids
                        if (ftype == 'fits')|use_moments:
                        # If the requested grid filetype is fits
                            # Write the REAL B grids
                            u.grid_to_fits_wrapper(
                                ((self.nDnR[wts2][1]/self.nDnR[wts2][0]))*B_i_0_0_RE_well,
                                self.mask,tempDirRand+'B_{}_0_0_RE_w{}_p{}_of_{}.fits'.format(
                                    s_idx,self.ell_step[ell_idx], self.box_idx+1,self.total_boxes))
                
                        if (ftype == 'npy')|use_moments:
                        # If the requested grid filetype is npy
                            # Write the REAL  B grids
                            np.save(tempDirRand+'B_{}_0_0_RE_w{}_p{}_of_{}.npy'.format(
                                        s_idx,self.ell_step[ell_idx], self.box_idx+1,self.total_boxes),
                                    ((self.nDnR[wts2][1]/self.nDnR[wts2][0]))*B_i_0_0_RE_well*self.mask)

                        # End file timer and update
                        file_end_time_W = time.perf_counter()
                        self.file_time += file_end_time_W - file_start_time_W
                else:
                    # if you are not storing the randoms...
                    if use_moments:
                        # if not storing the randoms but still using moments, you need B grids

                        # Start file timer
                        file_start_time_W = time.perf_counter()
                        
                        B_i_0_0_RE_well = (self.nDnR[wts2][0]/self.nDnR[wts2][1])*np.load(
                                                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                                                s_idx,self.box_idx+1,self.total_boxes))
                        # End file timer and update
                        file_end_time_W = time.perf_counter()
                        self.file_time += file_end_time_W - file_start_time_W
                        

                if use_moments:
                    eta_data = norm* self.mask*(self.D_g_0-self.R_g_0) * W_i_ell_m_RE*self.mask # W0*W1
                    eta_rand = (self.R_g_0*self.mask) * B_i_0_0_RE_well*self.mask # B0*B1
                    res = np.zeros(eta_data.shape, dtype=float)
                    gamma_s_ell = np.divide(eta_data, eta_rand, out=res, where=eta_rand!=0)
                    cond = gamma_s_ell != 0
                    gamma_s_ell_masked = gamma_s_ell[cond]
                    counts, _ = np.histogram(gamma_s_ell_masked.flatten(), edges)

                    # Start file timer
                    file_start_time_W = time.perf_counter()
                    
                    np.save('./moments/'+self.datafileString+
                       '_'+self.cfg_file.split('.txt')[0]+'/'+'gamma_{}_{}_p{}_of_{}'.format(
                                        s_idx,self.ell_step[ell_idx],
                                        self.box_idx+1,self.total_boxes),
                            counts)
                    # End file timer and update
                    file_end_time_W = time.perf_counter()
                    self.file_time += file_end_time_W - file_start_time_W
              
            # # PS if initial box
            # if self.box_idx == 0:
                
            #     # PS verb to mark the end of a radial step
            #     if self.verbose:
            #         print('Finished writing files'+
            #               ' for radial step s1 = {} Mpc (or Mpc/h)'.format(
            #                   np.round(
            #                       self.desired_s_bin_centers[s_idx],
            #                       decimals=2)))
        
        # PS to mark the end of a partition region
        print('Finished partition {} of {}'.format(self.box_idx+1,
                                                   self.total_boxes))
        return
            
    
# =================== SERIES CONVOLUTION STEP DRIVER ======================== #


def ConKerConvolveCatalog(data_file: str, rand_file: str, cfg_file: bool,
                          store_rand: str, ell_max: int, verbose: bool,
                          ftype: str = 'fits', use_moments=False, use_oqe=False,
                          data_file2=None, rand_file2=None):
    # A function to run the convolution step of the algorithm in series
    # Wraps the ConKerBox radial convolution for every partition
    
    try:
        # Check to make sure the divide plan exists
        # Also grab the total number of boxes
        _, _, _, total_boxes, _ = u.getLOSbox(rand_file, cfg_file, 0, rand_file2)
        
    except FileNotFoundError:
        # If the file isn't found
        
        # Trip a failure message
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('ConKer cannot find the divide scheme!')
        print("Make sure you've run DivideCatalog() "+
              "with save_plan = True")
        print('Requires a partition corresponding to both randoms '+
              'and cfg files')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    # Create a dictionary of times and initialize each one to 0
    times = {}
    times['T_MAP'] = 0.
    times['T_CONV'] = 0.
    times['T_KERN'] = 0.
    times['T_FILE'] = 0.
    
    for boxID in range(total_boxes):
        # For each of the boxes in the divide plan
        
        # Run the convolution with verbose=verbose if this is partition 0
        if boxID == 0:
            cb = ConKerBox(data_file = data_file,rand_file = rand_file,
                           cfg_file = cfg_file, store_rand = store_rand,
                           box_idx = boxID,
                           verbose = verbose,
                           use_oqe = use_oqe,
                           data_file2=data_file2, 
                           rand_file2=rand_file2)
            cb.radialConvolveLegendre(ell_max = ell_max,ftype=ftype, 
                                      use_moments=use_moments, use_oqe=use_oqe)
            
        # Run the convolution with verbose=False if this is another partition
        else:
            cb = ConKerBox(data_file = data_file,rand_file = rand_file,
                           cfg_file = cfg_file, store_rand = store_rand,
                           box_idx = boxID,
                           verbose = False,
                           use_oqe = use_oqe,
                           data_file2=data_file2, 
                           rand_file2=rand_file2)
            cb.radialConvolveLegendre(ell_max = ell_max,ftype=ftype, 
                                      use_moments=use_moments, use_oqe=use_oqe)
            
        # Update the timing trackers with each partition
        try:
            # May fail if a previous failure message has been tripped
            times['T_MAP'] += cb.end_map_time - cb.start_map_time
            times['T_CONV'] += cb.conv_time
            times['T_FILE'] += cb.file_time
            
        except AttributeError:
            # Return if something else has failed in ConKerBox()
            return
        
        # Update the kernel time if this is partions 0
        if boxID == 0:
            times['T_KERN'] += cb.kernel_time
            
    # PS to return the total CPU runtime
    print('ConKer Convolution Step took'+
          ' {} s CPU time'.format(times['T_MAP']+times['T_CONV']+
                                  times['T_KERN']+times['T_FILE']))
    
    # PS verb for timing breakdown by process
    if verbose:
        print('   Mapping time = {} CPU s'.format(times['T_MAP']))
        print('   Convolution time = {} CPU s'.format(times['T_CONV']))
        print('   Kernel creation time = {} CPU s'.format(times['T_KERN']))
        print('   File writing time = {} CPU s'.format(times['T_FILE']))
        
    # Get the name of the dictionary to store timing results
    
    tempDir = './grids/'+u.getFileString(data_file, data_file2)+\
        '_'+cfg_file.split('.txt')[0]+'/'
           
    # Save the timing breakdown
    with open(tempDir+'timing_info.txt', "w") as file:
        json.dump(times, file)
    
    # PS for final section
    print('\n'+'================================'+
          '==================================='+'\n')
    return
    
    
                