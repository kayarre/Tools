import numpy as np
import tifffile as tiff
import SimpleITK as sitk

import matplotlib.pyplot as plt

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def get_additional_info(pd_row):
    pd_data = pd_row[1]
    ff_path = pd_data["crop_paths"]
    ff = tiff.TiffFile(ff_path)

    base_res_x = pd_data["mpp-x"]
    base_res_y = pd_data["mpp-y"]

    print(base_res_x, base_res_y)
    #meta_data_orig = parse_vips(ff.pages[0].description)

    base_shape = ff.pages[0].shape
    pages = []
    for idx, page in enumerate(ff.pages):
        x_size = page.imagewidth
        y_size = page.imagelength
        xscale = base_shape[0] // x_size
        yscale = base_shape[1] // y_size
        #print(x_size, yscale, yscale*base_res_x)
        pages.append(dict(index=idx, size_x=x_size, size_y=y_size,
                          scale_x = xscale, scale_y=yscale,
                          mmp_x = xscale * base_res_x, 
                          mmp_y = yscale * base_res_y, 
                          )
                    )

    return pages


# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[:,:],cmap=plt.cm.Greys_r)
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[:,:],cmap=plt.cm.Greys_r)
    plt.title('moving image')
    plt.axis('off')
    #plt.ion()
    plt.show()
    #plt.pause(0.0001)

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:] + alpha*moving[:,:] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r)
    plt.axis('off')
    #plt.ion()
    plt.show()
    plt.pause(0.0001)
    

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot(angle):
    global metric_values, multires_iterations
    # del metric_values
    # del multires_iterations
    plt.title("Registration Iterations starting at {0:2.2f} angle".format(angle*180/np.pi))
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    # Close figure, we don't want to get a duplicate of the plot latter on.
    #plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    #(wait=True)
    #plt.ion()
    #plt.pause(0.0001)
    # Plot the similarity metric values
    # plt.plot(metric_values, 'r')
    # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    # plt.xlabel('Iteration Number',fontsize=12)
    # plt.ylabel('Metric Value',fontsize=12)
    # plt.show()

# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
        
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    
    t = ax.imshow(nda,
            extent=extent,
            interpolation='hamming',
            cmap='gray') #, origin='lower')
    
    if(title):
        plt.title(title)
    plt.show()

def resample(image, transform, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def get_center_of_gravity(np_image, spacing):
    f_itk = itk.GetImageFromArray(np_image)
    f_itk.SetSpacing(spacing)
    f_moments = itk.ImageMomentsCalculator.New(f_itk)
    f_moments.Compute()
    return f_moments.GetCenterOfGravity()

def get_sitk_image(np_image, spacing=(1.0, 1.0), origin=(0.0,0.0)):
    f_sitk = sitk.GetImageFromArray(np_image)
    f_sitk.SetSpacing(spacing)
    f_sitk.SetOrigin(origin)
    return f_sitk
