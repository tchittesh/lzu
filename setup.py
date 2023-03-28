from setuptools import find_packages, setup

print(find_packages(exclude=('configs', 'experiments', 'tools')))

if __name__ == '__main__':
    setup(
        name='lzu',
        packages=find_packages(include=('lzu')),
    )
