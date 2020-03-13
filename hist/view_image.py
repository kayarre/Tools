import SimpleITK as sitk
import os
import utils
import tifffile as tiff
import utils
import itk
import numpy as np

# if len(sys.argv) < 10:
#     print("Usage: {0} <inputImage> <outputImage> <seedX> <seedY> <Sigma> <SigmoidAlpha> <SigmoidBeta> <TimeThreshold>".format(sys.argv[0]))
#     sys.exit(1)

top_dir = "/media/store/krs/caseFiles/vwi_proc/"
#top_dir = "/Volumes/SD/caseFiles/vwi_proc"
#top_dir = "/media/sansomk/510808DF6345C808/caseFiles/vwi_proc/"

# this works for rgb, but not
fixed = os.path.join(top_dir, "case01/case_1_im_0014.tiff")

im_f = tiff.imread(fixed, key=6)
spacing = (16.16, 16.16)

#f_sitk = utils.get_sitk_image(im_f[:,:,:3], spacing = spacing, vector=True)

utils.display_image(im_f)


# checkerboard = sitk.CheckerBoardImageFilter()
# check_im = checkerboard.Execute( sitk.Cast(rescaleOutput, sitk.sitkFloat64),
#                                 rescale_gray,
#                                 (8,8)
#                                 )

# utils.display_images(sitk.GetArrayFromImage(dilate_im),
#                      sitk.GetArrayFromImage(masksed_im),
#                      checkerboard=sitk.GetArrayFromImage(check_im))