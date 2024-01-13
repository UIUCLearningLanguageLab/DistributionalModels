from setuptools import setup, find_packages

setup(
    name='distributional_models',
    version='0.1.0',
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research'],
    python_requires='>=3.6.8',
    install_requires=['setuptools>=68.2.2',
                      'torch~=>=2.1.0.dev20230330',
                      'numpy>=1.24.1',
                      'pandas>=1.5.3',
                      'scipy>=1.10.1',
                      'plotly>=5.14.0',
                      'seaborn>=0.12.2',
                      'matplotlib>=3.7.0',
                      'scikit-learn>=1.2.2',
                      ],
    url='https://github.com/phueb/Ludwig',
    license='MIT',
    author='Jon Willits and Jingfeng Zhang',
    author_email='jwillits@illinois.edu',
    description='Build, test, explore, and understand distributional models',
)
