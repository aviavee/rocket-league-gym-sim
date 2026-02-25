from setuptools import setup, find_packages

__version__ = "2.0.0"

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='rlgym-sim',
    packages=find_packages(),
    version=__version__,
    description='A clone of RLGym for use with RocketSim in reinforcement learning projects.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucas Emery, Matthew Allen, Zealan, and Mtheall',
    url='https://github.com/AechPro/rocket-league-gym-sim',
    install_requires=[
        'numpy>=2.0',
    ],
    extras_require={
        'gym': ['gym>=0.17'],
        'gymnasium': ['gymnasium>=1.0'],
        'legacy': ['rocketsim>=2.2.0', 'gym>=0.17'],   # ZealanL/RocketSim (mtheall bindings)
        'jax': [
            'jax[cuda12]',
            'flax',
            'chex',
        ],
        'all': [
            'gymnasium>=1.0',
            'gym>=0.17',
            'rocketsim>=2.2.0',
            'jax[cuda12]',
            'flax',
            'chex',
        ],
    },
    python_requires='>=3.12',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gym', 'reinforcement-learning', 'simulation'],
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
