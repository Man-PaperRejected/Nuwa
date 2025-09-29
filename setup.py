from setuptools import setup, find_packages

setup(
    name="nuwa",
    version="0.1.0",
    packages=find_packages(where='nuwa'),
    author="hyh",
    author_email="huangyihong0303@gmail.com",
    description="nuwa; Where, Not Just What: Mending the Spatial Fabric Torn by LVLM Acceleration",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dvlab-research/nuwa",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
