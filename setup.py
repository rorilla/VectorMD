from setuptools import setup, find_packages

setup(
    name='VectorMD',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'torch',
        'questionary',
        'faiss-cpu',
        'transformers',
        'sentence-transformers',
        'InstructorEmbedding',
        'pandas',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'vmd-init=vectormd.vmd:setup_cli',
            'vmd=vectormd.vmd:query_cli'
        ]
    },
)
