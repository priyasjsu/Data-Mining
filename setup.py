from setuptools import setup

setup(
    name='restaurant-recommendation',
    version='1.0',
    packages=['source', 'source.utils'],
    url='https://github.com/priyasjsu/Data-Mining.git',
    license='free',
    author='Priya Khandelwal',
    author_email='priyakhandelwal',
    description='A Data Mining project for restaurant recommendation using multiple technique',
    install_requires=[
        'requests==2.20.0'
    ]
)
