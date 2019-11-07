import pickle 

file_name = "crop_info_study_1.pkl"
crop_info = pickle.load( open(file_name , "rb" ) )

#print(crop_info)

delete_keys = ["131547_b", "131546_b", "131557_b", "131557_c", "131557_d", "131567_b"]

for d in delete_keys:
    try:
        del crop_info[d]
    except KeyError:
        print("Key '{0}' not found".format(d))

for key, value in crop_info.items():
    if (value["thickness"] <= 0.0):
        print(key, value)
with open(file_name, 'wb') as f:
     pickle.dump(crop_info, f)