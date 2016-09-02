from fabric.api import run, env
import sys

env.user = "parallella"
env.password = "parallella"

def host_type():
    try:
      run('uname -s')
    except: 
      e = sys.exc_info()[0]
      print(e)
