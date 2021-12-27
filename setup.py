import setuptools
import os

version = '1.0'

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    LongDescription = f.read()

setuptools.setup(
    name='Theia',
    zip_safe=True,
    version=version,
    description='Team Peryton Image Recognition',
    long_description_content_type="text/markdown",
    long_description=LongDescription,
    url='',
    author='ogent & pmsem',
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    author_email='',
    classifiers=[
    ],
    license='',
    packages=setuptools.find_packages()
)
