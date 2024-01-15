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
    install_requires=['setuptools',
                      'torch==1.6.0',
                      'numpy==1.17.5',
                      'pandas',
                      'scipy',
                      'plotly',
                      'seaborn',
                      'matplotlib',
                      'scikit-learn',
                      ],
    url='https://github.com/phueb/Ludwig',
    license='MIT',
    author='Jon Willits and Jingfeng Zhang',
    author_email='jwillits@illinois.edu',
    description='Build, test, explore, and understand distributional models',
)
