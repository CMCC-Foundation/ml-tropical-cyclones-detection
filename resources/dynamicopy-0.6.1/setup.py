import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamicopy",
    version="0.6.1",
    author="Stella Bourdin",
    author_email="stella.bourdin@lsce.ipsl.fr",
    description="A set of tool to use and analyse netCDF _data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # TODO : Upload to Github when done
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "scipy", "seaborn", "matplotlib", "netCDF4", "xarray", "shapely", "sklearn"],
    include_package_data=True,
    package_data={"":['_data/*.csv']}
)

# Black formatting:
# `python -m black <directory or file(s)>`

# To generate the distribution:
# 1 / Check that wheel is up to date with `pip install --user --upgrade setuptools wheel`
# 2 / Run `python setup.py sdist bdist_wheel`

# To upload to PyPI:
# 1 / Check that twine is up-to-date `pip install --user --upgrade twine`
# 2 / Upload with `python -m twine upload --repository pypi dist/*`