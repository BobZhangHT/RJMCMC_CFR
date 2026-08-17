from setuptools import Extension, setup

import numpy


setup(
    name="rjmcmc-cfr-backend",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_rjmcmc_c",
            sources=["rjmcmc_c.c"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
