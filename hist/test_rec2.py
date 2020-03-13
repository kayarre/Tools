from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#import large_image
import openslide
import math
import string
import pickle
import os

#proj_dir = "/Volumes/muffins/vwi_proj"
proj_dir = "/Volumes/SD/caseFiles/vwi_proj"
#csv_file ="/Users/sansomk/code/ipythonNotebooks/VWI_proj/SpectrumData_test.csv"
csv_file ="/Users/sansomk/code/ipythonNotebooks/VWI_proj/SpectrumData_test.csv"
sub_dir ="case"
file_ext=".svs"

df = pd.read_csv(csv_file)

completed_cases = [1]
im_id = df[["study", "Image_ID"]]

file_name = []
full_path = []

for i in df[["study","Image_ID"]].iterrows():
    #print(i[1]["study"])
    #print(i)
    path_test = os.path.join(proj_dir, "{0}{1:02d}".format(sub_dir,i[1]["study"]), 
                            "{0}{1}".format(i[1]["Image_ID"],file_ext))
    if (os.path.exists(path_test)):
        split_p = os.path.split(path_test)
        file_name.append(split_p[-1])
        full_path.append(path_test)
    else:
        print(" some kind of error for this file: {0}".format(path_test))


df["file_name"] = file_name
df["full_path"] = full_path

#df.to_pickle(os.path.join(proj_dir, "process_df.pkl"))
#quit()

df = df[~df["study"].isin(completed_cases)]

study_id = 1
path_list = list(df["full_path"])
study_list = list(df["study"])
case_info = dict(path=path_list, study_id=study_list)


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def next_power_of_2(n): 
    count = 0

    # First n in the below  
    # condition is for the  
    # case where n is 0 
    if (n and not(n & (n - 1))): 
        return n 
        
    while( n != 0): 
        n >>= 1
        count += 1
        
    return 1 << count

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

class define_rectangle(object):

    def __init__(self):
        self.left = 0.0
        self.top = 0.0
        self.width = 0.0
        self.height = 0.0
        self.new_mag = 0.0

    def callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.left = x1 * self.new_mag
        self.top = y1 * self.new_mag
        self.width = (x2-x1) * self.new_mag
        self.height = (y2-y1) * self.new_mag
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (self.left,
                self.top, self.width, self.height))
        #print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def update_mag(self, new_mag):
        self.new_mag = new_mag

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


class Index(object):

    def __init__(self, current_ax, case_info, fig, rectangle):
        self.index_images = 0
        self.new_mag = 0.0
        self.case_info = case_info
        self.path_list = case_info['path']
        self.study_id = case_info['study_id']
        self.cur_ax = current_ax
        self.base_label = ""
        self.image_path = self.path_list[self.index_images]
        self.image_id = self.study_id[self.index_images]
        self.set_base_label()
        self.fig = fig
        self.alphabet = list(string.ascii_lowercase)

        self.thickness = 0.0
        self.gap = 0.0
        self.label = self.base_label

        self.text_box = self.set_text_box()
        self.init_test = False
        self.rectangle = rectangle
        self.play()
    
    def set_thickness(self, thickness):
        self.thickness = thickness
    
    def set_gap(self, gap):
        self.gap = gap
    
    def set_cur_label(self,label):
        self.label = label

    def set_base_label(self):
        self.base_label = os.path.splitext(os.path.split(self.image_path)[-1])[0]

    def set_text_box(self):
        text_box = self.cur_ax.text(0.05, 0.95,
            "study id : {0} base_label: {1}".format(self.image_id, self.base_label),
            transform=self.fig.transFigure, fontsize=12,
            verticalalignment='top'
            )
        return text_box
    
    def update_text(self):
        self.text_box.set_text(
            "study id : {0} base_label: {1}".format(self.image_id, self.base_label)
        )

    def next(self, event):
        if (self.index_images < len(path_list)):
            self.index_images += 1
        else:
            print("end of images use the other button")
        self.play(event)

    def prev(self, event):
        if (self.index_images > 0):
            self.index_images -= 1
        else:
            print("end of images use the other button")
        self.play(event)

    def play(self, event=None):
        self.alphabet = list(string.ascii_lowercase)
        self.image_path = self.path_list[self.index_images]
        self.image_id = self.study_id[self.index_images]
        self.set_base_label()

        print(self.image_path)
        image  = openslide.open_slide(self.image_path)
        levels = image.level_dimensions
        meta_data = image.properties
        #res_mag =  float(meta_data['aperio.AppMag'])
        level_down = image.level_downsamples
        print(levels, level_down)
        
        # reduce the lowest resolution by an additional 4th 
        mag = level_down[0] / 4.0
        mag2 = level_down[-1]

        # the ratio of higher resolution with the lower resolution
        # divide the highest by it to get the thumbnail size 
        # multiple to get back the correct pixel range info

        self.new_mag = round(mag2 / mag)
        self.rectangle.update_mag(self.new_mag)
        size_x = int(round(levels[0][0] // self.new_mag))
        size_y = int(round(levels[0][1] // self.new_mag))

        print(size_x, size_y)

        # get out a numpy array b/c lazy
        im_low_res = np.array(image.get_thumbnail(size=(size_x, size_y)))

        print(self.new_mag)
        print(" mag at level {0}: is {1}".format(1, mag))
        print(" mag at level {0}: is {1}".format(len(level_down), mag2))

        if (self.init_test == False):
            self.cur_ax.imshow(im_low_res)
            self.init_test = True
        else:
            self.cur_ax.images[0].set_extent(extent=[0, im_low_res.shape[1],
                                                    im_low_res.shape[0], 0])
            self.cur_ax.images[0].set_data(im_low_res)
            self.cur_ax.relim()
            self.cur_ax.autoscale_view()
            self.update_text()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
    
    def test_text_box(self, text):
        print(text)
        self.label = text


class MyRectangleSelector(RectangleSelector):
    def release(self, event):
        super(MyRectangleSelector, self).release(event)
        self.to_draw.set_visible(True)
        self.canvas.draw()

def mycallback(event):
    if toggle_selector.RS.active:
        # print('mycallback')
        toggle_selector.RS.update()

class crop_box(object):
    def __init__(self, rectangle, image_index):
        self.n_crop = 0
        self.rectangle = rectangle
        self.image_index = image_index
        self.crop_dict = {}
        self.current_label = ""
        self.cur_thickness = 5.0
        self.cur_gap = 25.0
        

    def save_crop(self, event):
        
        label = self.image_index.base_label
        extents = [ a * self.image_index.new_mag for a in self.rectangle.extents] 

        left = int(round(extents[0]))
        width = int(round(extents[1] - left))
        top = int(round(extents[2]))
        height = int(round(extents[3] - top))
        new_region = dict(left=left,
                    top=top,
                    width=width,
                    height=height,
                    units='base_pixels'
                    )
        # first label always ends _a
        new_label = label + "_a"
        while (new_label in self.crop_dict.keys()):
            new_label = label + "_" + self.image_index.alphabet.pop(0)

        self.crop_dict[new_label] = dict(region=new_region,
                                         file = self.image_index.image_path,
                                         thickness = self.cur_thickness,
                                         gap = self.cur_gap,
                                         crop_id = self.n_crop
                                         )
        self.current_label = new_label
        print(new_label, self.crop_dict[new_label])
        #self.print_shit()
        self.n_crop += 1

    def get_cur_label(self):
        return self.current_label
    
    def get_dict(self):
        return self.crop_dict

    def print_shit(self):
        print( self.crop_dict)

    def make_square(self, event):
        # if toggle_selector.RS.active:
        #     # print('mycallback')
        #     toggle_selector.RS.update()
        # give left right top bottom
        #
        e = self.rectangle.extents
        x_sz = (e[1] - e[0]) / 2.0 #+ e[0]
        y_sz = (e[3] - e[2]) / 2.0 #+ e[2]
        #print(self.rectangle.center)
        sq_d = max([x_sz, y_sz])
        sq_2 = round_up_to_even(sq_d)
        #sq_2 = shift_bit_length(sq_d)
        #print(sq_2)
        #print(next_power_of_2(x_sz), next_power_of_2(y_sz))
        #sq_2 = next_power_of_2(sq_d)

        x_c = round(self.rectangle.center[0])
        y_c = round(self.rectangle.center[1])

        left   = x_c - sq_2
        right  = x_c + sq_2
        top    = y_c - sq_2
        bottom = y_c + sq_2
        #print(dir(self.rectangle))
        self.rectangle.extents = (left, right, top, bottom)
        print(left, top, right - left, bottom - top)
    
    def set_thickness(self, text):
        self.cur_thickness = float(text)
    
    def get_thickness(self):
        return str(self.cur_thickness)

    def set_gap(self, text):
        self.cur_gap = float(text)

    def get_gap(self):
        return str(self.cur_gap)

fig, current_ax = plt.subplots()                 # make a new plotting range
#current_ax.imshow(im_low_res)
current_ax.set_autoscaley_on(True)
current_ax.set_autoscalex_on(True)


line_select = define_rectangle()
callback = Index(current_ax, case_info, fig, line_select)
axprev   = fig.add_axes([0.65, 0.05, 0.07, 0.07])
axnext   = fig.add_axes([0.73, 0.05, 0.07, 0.07])
axsave   = fig.add_axes([0.81, 0.05, 0.07, 0.07])
axsquare = fig.add_axes([0.89, 0.05, 0.07, 0.07])

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Prev')
bprev.on_clicked(callback.prev)


#start_button = Button(button_axes, 'Next', image=im_low_res)
#current_ax.imshow(im_low_res)

print("\n      click  -->  release")

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = MyRectangleSelector(current_ax, line_select.callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=20, minspany=20,
                                       spancoords='pixels',
                                       interactive=True)


cropper = crop_box(toggle_selector.RS, callback)
bsave = Button(axsave, 'Save')
bsave.on_clicked(cropper.save_crop)

bsquare = Button(axsquare, 'Square')
bsquare.on_clicked(cropper.make_square)

#toggle_selector.RS.set_active(True)

plt.connect('key_press_event', toggle_selector)
plt.connect('draw_event', mycallback)

axtext = fig.add_axes([0.75, 0.92, 0.2, 0.050])
text_box = TextBox(axtext, 'thickness', initial=cropper.get_thickness())
text_box.on_submit(cropper.set_thickness)

axtext2 = fig.add_axes([0.75, 0.85, 0.2, 0.050])
text_box2 = TextBox(axtext2, 'gap', initial=cropper.get_gap())
text_box2.on_submit(cropper.set_gap)

#start_button.on_clicked(play)

plt.show()

all_crops = cropper.get_dict()
print(all_crops)
with open(os.path.join(proj_dir, 'crop_info.pkl'), 'wb') as f:
    pickle.dump(all_crops, f)

