import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

setuptools.setup(
    name="ghost",
    version="0.1",
    author="Emil Ljungberg",
    author_email="ljungberg.emil@gmail.com",
    description="Python code to UNITY phantom data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./"),
    python_requires=">=3.6",
    py_modules=['ghost'],
    install_requires=requirements,
    # entry_points={
    #     'console_scripts': [
    #         'rrdf2bart=rrdftools.main_rrdf2bart:main',
    #         'rrdf2riesling=rrdftools.main_rrdf2riesling:main',
    #         'bart2nii=rrdftools.main_bart2nii:main',
    #     ]
    # }
)
