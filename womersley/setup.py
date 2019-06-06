from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='womersley',
    version='0.1',
    description='Tools to manipulate the Womserley boundary condition',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    keywords='womersley boundary condition',
    url='https://github.com/kayarre/Tools/womersley',
    author='Kurt Sansom',
    author_email='sansomk@uw.edu',
    license='MIT',
    packages=['womersley'],
    #install_requires=[
        #'markdown',
    #],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'])
