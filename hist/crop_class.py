class single_crop:
    def __init__(self, row):
        assert len(row) >= 5
        
        self.region = dict(left=row[0],
                            top=row[1],
                            width=row[2],
                            height=row[3],
                            units='base_pixels'
                          )
        self.scale = dict(magnification=row[4])
        self.format = dict(format="numpy")

    def shift_region(self, row):
        assert (len(row)  > 1 and len(row) < 5)

        self.region.left += row[0]
        self.region.top += row[1]
        if ( len(row) == 4):
            self.region.width += row[2]
            self.region.height += row[3]

    def update_region(self, row):
        assert (len(row)  > 1 and len(row) < 5)

        self.region.left = row[0]
        self.region.top = row[1]
        if ( len(row) == 4):
            self.region.width = row[2]
            self.region.height = row[3]

    def update_units(self, new_units):
        self.region.units = new_units

    def shift_scale(self, shift):
        self.scale.magnification *= shift

    def update_scale(self, shift):
        self.scale.magnification = shift

    def update_format(self, new_format):
        self.format.format = new_format


class crop_data:
    def __init__(self):
        self.label = dict()
        self.orig_paths = dict()
        self.paths = dict()
    def add_row(self, key, row):

        if key in self.label.keys():
            print("key already exists overwriting data")
        self.label[key] = single_crop(row)

    def add_crop(self, key, single_crop_instance):
        if key in self.label.keys():
            print("key already exists overwriting data")
        self.label[key] = single_crop_instance

    def add_orig_path(self, key, file_path):
        if key in self.label.keys():
            return
        if key in self.orig_paths.keys():
            print("key already exists in path dict, overwriting")
        self.orig_paths[key] = file_path

    def add_path(self, key, file_path):
        if key not in self.label.keys():
            return
        if key in self.paths.keys():
            print("key already exists in path dict, overwriting")
        self.paths[key] = file_path

# format_to_dtype = {
#     'uchar' : np.uint8,
#     'char' : np.int8,
#     'ushort' : np.uint16,
#     'short' : np.int16,
#     'uint' : np.uint32,
#     'int' : np.int32,
#     'float' : np.float32,
#     'double' : np.float64,
#     'complex' : np.complex64,
#     'dpcomplex' : np.complex128,
# }