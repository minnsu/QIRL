from setuptools import setup, find_packages

setup(
    name='qirl',
    version='0.0.1',
    description='Quantitive Investment with Reinforcement Learning',
    author='minnsu',
    author_email='kms48491000@gmail.com',
    url='https://github.com/minnsu/QIRL.git',
    install_requires=['torch'],
    packages=find_packages(exclude=[]),
    keywords=['Quantitive Investment', 'Reinforcement Learning', 'RL', 'QI', 'QIRL', 'qirl'],
    python_requires='>=3.8',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)