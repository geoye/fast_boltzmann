from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='fast_boltzmann',
    version='1.0.0',
    url='https://github.com/geoye/fast_boltzmann',
    license='MIT',
    author='Yuxuan YE<yuxuanye145@gmail.com>, Xinghua Cheng<cxh9791156936@gmail.com>',
    author_email='yuxuanye145@gmail.com, cxh9791156936@gmail.com',
    description='`fast_boltzmann` is a Python package developed for the fast computation of the Boltzmann Entropy '
                '(also known as configurational entropy) for a two-dimensional numerical or nominal data array.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        'Source': 'https://github.com/geoye/fast_boltzmann',
        "Bug Tracker": "https://github.com/geoye/fast_boltzmann/issues",
    },
    install_requires=["numpy>=1.20.0"],
    python_requires='>=3.6',
    package_dir={'fast_boltzmann': 'src/fast_boltzmann'},
    packages=['fast_boltzmann'],
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=['python', 'entropy', 'quality assessment', 'thermodynamic consistency', 'landscape ecology']

)
