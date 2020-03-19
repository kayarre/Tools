import os
import pooch
from gen_sphere import particle_generator


output_dir = "/media/store/krs/particles"

par_gen = particle_generator()
par_gen.setup_pooch()
par_gen.set_base_mesh()
pooch_obj = par_gen.get_pooch_obj()
reg_dict = pooch_obj.registry

for key, value in reg_dict.items():
    dir_name, file_name = os.path.split(key)
    # not ideal but I am lazy
    vtp_name = os.path.splitext(file_name)[0] + ".vtp"
    sub_dir = os.path.join(output_dir, dir_name)

    par_gen.generate_particle(key)
    print("generation success")

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")

    par_gen.write_particle(os.path.join(sub_dir, vtp_name))
    par_gen.clean_mesh()


# fname = pooch_particles.fetch("C109-sand/C109-20002.anm")
print("all done")
