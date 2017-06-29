import math
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def parse_commandline():
        '''Parses Command line arguments and initiates rest of program.'''
        command = None
        #Instantiate command line parser.
        command = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''
                                                            Plotting Utility v0.1.
                                                            Author: James Kent. 
                                                            Institution: University of Cambridge, 2016.
                                                            Email: jck42@cam.ac.uk 

                                                            Takes a CSV File and Plots and outputs to csv file.'''))

        
        #Add required arguments.
        command.add_argument('filepath',help=
                                 ('Visibility File Path'))
        command.add_argument('resolution', metavar='R', help=
                                        ('Number of points per grid point.'))
        command.add_argument('size',metavar='N',help='Size of DFTd grid')
        command.add_argument('wavelength',metavar='L',help='Central wavelength, for converting u/v coordinates.')
        #command.add_argument('Plot_Title',metavar='PT',help='Plot Title')
        #self.command.add_argument('Legend_Titles',metavar='LT',help='Legend titles. Seperate using [title1,title2].')
        #self.command.add_argument('output_file',metavar='O',help='Saves plot to this PNG filepath.')

        #Optional arguements.
        command.add_argument('--echocommands',dest='echocommands',action='store_true',help='Echo Commands Back. Stops profiler from running. For debugging.')
        #self.command.add_argument('--power=',dest='powerof',default='2',help='Specifies what factor to test powers of, e.g 3^N, 7^N, 11^N etc. If this argument is not stated the default is 2^N.')
    

        args = command.parse_args()


        if(args.echocommands is True):
            echo_args(args)

        return(args)

def echo_args(args):
        '''Mostly for debug purposes when changing CLI.'''

        print("Received arguments: \n")
        print(args) 


def importvisibilities(filename):

    data = np.genfromtxt(filename,dtype=complex,delimiter=',')

    return data

def DFT_reference(N,resolution,data):

    data_np = data
    resp = int(1/resolution)
    
    dft_real = np.zeros([N*2*resp,N*2*resp])
    dft_imag = np.zeros([N*2*resp,N*2*resp])


    for l in np.arange(-N,N,resolution):

        print("Calculating row: {}".format(l))
        for m in np.arange(-N,N,resolution):
            sumreal = 0
            sumimag = 0
            for idx_u,data in enumerate(data_np):
                

                subang1 = (m*data[2])/80000
                subang2 = (l*data[1])/80000
                angle = 2 * math.pi * (subang1 + subang2)
                sumreal += data[0].real * np.cos(angle) + data[0].imag * np.sin(angle)
                sumimag += -data[0].real * np.sin(angle) + data[0].imag * np.cos(angle)

            dft_real[int((l+N)*resp)][int((m+N)*resp)] = sumreal
            dft_imag[int((l+N)*resp)][int((m+N)*resp)] = sumimag


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('DFT(real)')
    plt.imshow(dft_real,interpolation='nearest')
    plt.colorbar(orientation='horizontal')
    ax = fig.add_subplot(122)
    ax.set_title('DFT(imag)')
    plt.imshow(dft_imag,interpolation='nearest')
    plt.colorbar(orientation='horizontal')
    plt.show()


def DFT_parallel(N,resolution,data):

    data_np = data
    resp = int(1/resolution)

    dft_real = np.zeros([N*2*resp,N*2*resp])
    dft_imag = np.zeros([N*2*resp,N*2*resp])

    for l in np.arange(-N, N, resolution):

        print("Calculating row: {}".format(l))
        for m in np.arange(-N, N, resolution):

            sumreal = 0
            sumimag = 0

            subang1 = m*data_np[:,2] / 80000
            subang2 = l*data_np[:,1] / 80000
            angle = 2* np.pi * (np.add(subang1,subang2))

            for idx,data in enumerate(data_np):
    

                sumreal += data[0].real * np.cos(angle[idx]) + data[0].imag * np.sin(angle[idx])
                sumimag += -data[0].real * np.sin(angle[idx]) + data[0].imag * np.cos(angle[idx])

            
            dft_real[int((l+N)*resp)][int((m+N)*resp)] = sumreal
            dft_imag[int((l+N)*resp)][int((m+N)*resp)] = sumimag


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('DFT(real)')
    plt.imshow(dft_real,interpolation='nearest')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label("Relative Intensity")
    ax = fig.add_subplot(122)
    ax.set_title('DFT(imag)')
    plt.imshow(dft_imag,interpolation='nearest')
    plt.suptitle('VLA Dirty Map. Five simulated gaussian point sources.')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label("Relative Intensity")
    plt.show()


def main():

    args = parse_commandline()

    data = importvisibilities(args.filepath)

    #Iterate over rows and create uv gridding.
    data_np = np.array(data)
    u = []
    v = []
    vis = []
    for row in data:
        u.append(row[1].real)
        v.append(row[2].real)
        vis.append(row[0])

    u_np = np.array(u)
    v_np = np.array(v)
    vis_np = np.array(vis)

    u_np_neg = np.negative(u_np)
    v_np_neg = np.negative(v_np)


    vis_np = np.append(vis_np,vis_np)
    u_np = np.append(u_np,u_np_neg)
    v_np = np.append(v_np,v_np_neg)
    

    #Convert u and v to wavelengths
    u_np = u_np / 0.15
    v_np = v_np / 0.15

    #Plot UV coverage.
    plt.scatter(u_np,v_np)
    plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')

    plt.tricontourf(u_np,v_np,vis_np.real)
    plt.show()


    DFT_parallel(int(args.size),float(args.resolution),data_np)

if __name__ == "__main__":
    main()
