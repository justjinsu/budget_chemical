from setuptools import setup, find_packages

setup(
    name='carbonbudget',
    version='0.1.0',
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    install_requires=[
        'pandas',
        'numpy',
        'openpyxl',
        'requests',
        'scipy',
        'matplotlib',
        'xlrd',
    ],
    author='Sanghyun Hong',
    author_email='sanghyun@planit.institute',
    description='Carbon budget allocation and emission pathway modeling tools',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PLANiT-Institute/carbonbudget',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
