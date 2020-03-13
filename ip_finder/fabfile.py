#from fabric.api import run, env
import sys

from fabric import Connection
from fabric import task

#env.user = "parallella"
#env.password = "parallella"

#env.user = "Admin"
#env.password = ""


def host_type():
    try:
      run('uname -s')
    except: 
      e = sys.exc_info()[0]
      print(e)

@task
def hello(ctx):
  print("Hello World")
