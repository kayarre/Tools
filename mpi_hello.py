#run stuff
import os
from mpi4py import MPI

def run_script():
  rank = MPI.COMM_WORLD.Get_rank()
  size = MPI.COMM_WORLD.Get_size()
  name = MPI.Get_processor_name()

  # ******************************
  # actual (serial) work goes here
  # ******************************

  print("Hello, world! This is rank {0:d} of {1:d} running on {2:s}".format(rank, size, name))


if ( __name__ == '__main__' ):
  run_script()
