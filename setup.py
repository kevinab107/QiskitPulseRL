from setuptools import setup

with open("version.txt", "r") as fv:
    VERSION = fv.read()
fv.close()


NAME = "pulseRL"
DESCRIPTION = "RL algorithm for quantum control"
URL = "https://github.com/kevinab107/QiskitPulseRL"
REQUIRES_PYTHON = ">=3.7"

AUTHOR = "Kevin Arbaham"
AUTHOR_EMAIL = "kevinab107@gmail.com"

DEPENDENCIES = [
    "tensorflow==2.6.0",
    "numpy==1.19.5",
    "tf-agents==0.10.0",
    "qiskit==0.31.0",
    "protobuf==3.18.1",
]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    python_requires=REQUIRES_PYTHON,
    install_requires=DEPENDENCIES,
    entry_points={"console_scripts": ["pulseRL=bin.__main__:main"]},
    packages = ['pulseRL']
)
