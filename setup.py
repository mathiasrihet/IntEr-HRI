from setuptools import setup

setup(name='IntEr-HRI',
      version='1.0',
      description='Code from ChocolaTeam during IntEr-HRI competition (IJCAI 2023)',
      author='Mathias Rihet',
      author_email='mathias.rihet@isae-supaero.fr',
      url='https://github.com/mathiasrihet/IntEr-HRI',
      packages=['Choco_IntEr-HRI'],
      install_requires = ['python == 3.10.9',
                          'numpy == 1.23.3',
                          'pandas == 1.5.3',
                          'mne == 1.3.1',
                          'scikit-learn == 1.2.2',
                          'pyriemann == 0.3',
                          'imbalanced-learn == 0.10.1',
                          'seaborn == 0.12.2',
                          'matplotlib == 3.6.2'],
)
