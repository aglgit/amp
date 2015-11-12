#!/usr/bin/env python

import os

from ase.structure import molecule
from ase.calculators.emt import EMT
from ase import Atoms
from amp import Amp
from amp import SimulatedAnnealing
from amp.regression import NeuralNetwork
from amp.descriptor import Behler
import shutil

###############################################################################


def make_training_images():
    atoms = molecule('CH4')
    atoms.set_calculator(EMT())
    atoms.get_potential_energy(apply_constraint=False)

    images = [atoms]

    atoms = Atoms(atoms)
    atoms.set_calculator(EMT())
    atoms[3].z += 0.5

    atoms.get_potential_energy(apply_constraint=False)

    images += [atoms]
    return images

###############################################################################


def test():

    images = make_training_images()

    for descriptor in [None, Behler()]:
        for global_search in [None, SimulatedAnnealing(temperature=10,
                                                       steps=5)]:
            for data_format in ['json', 'db']:
                for save_memory in [False,]:
                    for fortran in [False, True]:
                        for cores in range(1, 4):

                            print (descriptor, global_search, data_format,
                                   save_memory, fortran, cores)

                            pwd = os.getcwd()
                            testdir = 'read_write_test'
                            os.mkdir(testdir)
                            os.chdir(testdir)

                            regression = NeuralNetwork(hiddenlayers=(5, 5,))

                            calc = Amp(label='calc',
                                       descriptor=descriptor,
                                       regression=regression,
                                       fortran=fortran,)
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       cores=cores,)

                            # Test that we cannot overwrite. (Strange code
                            # here because we *want* it to raise an
                            # exception...)
                            try:
                                calc.train(images,
                                           energy_goal=0.01, force_goal=10.,
                                           global_search=global_search,
                                           extend_variables=True,
                                           data_format=data_format,
                                           save_memory=save_memory,
                                           cores=cores,)
                            except IOError:
                                pass
                            else:
                                raise RuntimeError(
                                    'Code allowed to overwrite!')

                            # Test that we can manually overwrite.
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       overwrite=True,
                                       cores=cores,)

                            # New directory calculator.
                            calc = Amp(label='testdir/calc',
                                       descriptor=descriptor,
                                       regression=regression,
                                       fortran=fortran,)
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       cores=cores,)

                            # Open existing, save under new name.
                            calc = Amp(load='calc',
                                       label='calc2',
                                       descriptor=descriptor,
                                       regression=regression,
                                       fortran=fortran,)
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       cores=cores,)

                            # Change label and re-train
                            calc.set_label('calc_new/calc')
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       cores=cores,)

                            # Open existing without specifying new name.
                            calc = Amp(load='calc',
                                       descriptor=descriptor,
                                       regression=regression,
                                       fortran=fortran,)
                            calc.train(images,
                                       energy_goal=0.01, force_goal=10.,
                                       global_search=global_search,
                                       extend_variables=True,
                                       data_format=data_format,
                                       save_memory=save_memory,
                                       cores=cores,)

                            os.chdir(pwd)
                            shutil.rmtree(testdir, ignore_errors=True)
                            del calc, regression

###############################################################################

if __name__ == '__main__':
    test()
