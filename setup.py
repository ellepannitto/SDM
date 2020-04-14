from setuptools import setup

setup(
    name='sdm',
    description='Structured Distributional Model',
    author=['Ludovica Pannitto', 'Giulia Rambelli'],
    author_email=['ellepannitto@gmail.com', ''],
    version='0.1.0',
    license='MIT',
    packages=['sdm', 'sdm.logging_utils', 'sdm.utils', 'sdm.core'],
    package_data={'sdm': ['logging_utils/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sdm = sdm.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1'],
)