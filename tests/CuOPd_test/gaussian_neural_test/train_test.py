"""
Exact Gaussian-neural scheme loss function, energy loss and force loss
for five different non-periodic configurations and three three different
periodic configurations have been calculated in Mathematica. This script
checks the values calculated by the code during training with and without
fortran modules and also on different number of cores.

"""


import numpy as np
from collections import OrderedDict
from ase import Atoms
from ase.calculators.emt import EMT
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


# The test function for non-periodic systems

convergence = {'energy_rmse': 10.**10.,
               'energy_maxresid': 10.**10.,
               'force_rmse': 10.**10.,
               'force_maxresid': 10.**10., }


def non_periodic_0th_bfgs_step_test():

    # Making the list of periodic image

    images = [Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  3.],
                            [1.,  0.,  0.]])),
              Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  1.,  0.],
                         [1.,  2.,  1.],
                            [-1.,  1.,  2.],
                            [1.,  3.,  2.]])),
              Atoms(symbols='PdO',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[2.,  1., -1.],
                         [1.,  2.,  1.]])),
              Atoms(symbols='Pd2O',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[-2., -1., -1.],
                         [1.,  2.,  1.],
                         [3.,  4.,  4.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.]]))]

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': [
                    'Pd', 'Pd'], 'eta':0.2, 'gamma':0.3, 'zeta':1},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5},
                 {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.4,
                  'gamma':0.3, 'zeta':4}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'O'],
                  'eta':0.2, 'gamma':0.3, 'zeta':1},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                  'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2,), 'Pd': (2,), 'Cu': (2,)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9],
                                                             [-2.5, -1.5]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7],
                                                              [-3.0, 2.0]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9],
                                                              [-3.5, 0.5]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    # Correct values

    ref_loss = 7144.810783950215
    ref_energyloss = (24.318837496017185 ** 2.) * 5
    ref_forceloss = (144.70282475062052 ** 2.) * 5
    ref_dloss_dparameters = np.array([0, 0, 0, 0, 0, 0, 0.01374139170953901,
                                      0.36318423812749656,
                                      0.028312691567496464,
                                      0.6012336354445753, 0.9659002689921986,
                                      -1.2897770059416218, -0.5718960935176884,
                                      -2.6425667221503035, -1.1960399246712894,
                                      0, 0, -2.7256379713943852, -
                                      0.9080181026559658,
                                      -0.7739948323247023, -0.2915789426043727,
                                      -2.05998290443513, -0.6156374289747903,
                                      -0.0060865174621348985, -
                                      0.8296785483640939,
                                      0.0008092646748983969,
                                      0.041613027034688874,
                                      0.003426469079592851, -
                                      0.9578004568876517,
                                      -0.006281929608090211, -
                                      0.28835884773094056,
                                      -4.2457774110285245, -4.317412094174614,
                                      -8.02385959091948, -3.240512651984099,
                                      -27.289862194996896, -26.8177742762254,
                                      -82.45107056053345,
                                      -80.6816768350809])

    # Testing pure-python and fortran versions of Gaussian-neural on different
    # number of processes

    for fortran in [False, True]:
        for cores in range(1, 6):
            label = 'train-nonperiodic/%s-%i' % (fortran, cores)
            print label
            calc = Amp(descriptor=Gaussian(cutoff=6.5,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(
                       hiddenlayers=hiddenlayers,
                       weights=weights,
                       scalings=scalings,
                       activation='sigmoid',),
                       label=label,
                       dblabel=label,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)

            assert (abs(calc.model.lossfunction.loss -
                        ref_loss) < 10.**(-5.)), \
                'The calculated value of loss function is wrong!'

            assert (abs(calc.model.lossfunction.energy_loss -
                        ref_energyloss) < 10.**(-7.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.model.lossfunction.force_loss -
                        ref_forceloss) < 10 ** (-4)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                assert(abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_]) < 10 ** (-7)), \
                    "The calculated value of loss function derivative is \
                    'wrong!"

            dblabel = label
            secondlabel = '_' + label

            calc = Amp(descriptor=Gaussian(cutoff=6.5,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='sigmoid'),
                       label=secondlabel,
                       dblabel=dblabel,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)

            assert (abs(calc.model.lossfunction.loss -
                        ref_loss) < 10.**(-5.)), \
                'The calculated value of loss function is wrong!'

            assert (abs(calc.model.lossfunction.energy_loss -
                        ref_energyloss) < 10.**(-7.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.model.lossfunction.force_loss -
                        ref_forceloss) < 10 ** (-4)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                assert(abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_]) < 10 ** (-7)), \
                    'The calculated value of loss function derivative is \
                    wrong!'


# The test function for periodic systems and first BFGS step

def periodic_0th_bfgs_step_test():

    # Making the list of images

    images = [Atoms(symbols='PdOPd',
                    pbc=np.array([True, False, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                         [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.],
                         [1.5,  1.5,  1.5]])),
              Atoms(symbols='PdO',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[1.8,  0.,  0.],
                         [0.,  1.8,  0.],
                            [0.,  0.,  1.8]]),
                    positions=np.array(
                        [[0.,  0., 0.]]))]

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                          'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2,), 'Pd': (2,), 'Cu': (2,)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    # Correct values

    ref_loss = 8004.292841472513
    ref_energyloss = (43.736001940333836 ** 2.) * 3
    ref_forceloss = (137.4099476110887 ** 2.) * 3
    ref_dloss_dparameters = np.array([0.0814166874813534, 0.03231235582927526,
                                      0.04388650395741291,
                                      0.017417514465933048,
                                      0.0284312765975806, 0.011283700608821421,
                                      0.09416957265766414, -
                                      0.12322258890997816,
                                      0.12679918754162384, 63.5396007548815,
                                      0.016247700195771732, -86.62639558745185,
                                      -0.017777528287386473, 86.22415217678898,
                                      0.017745913074805372, 104.58358033260711,
                                      -96.7328020983672, -99.09843648854351,
                                      -8.302880631971407, -1.2590007162073242,
                                      8.3028773468822, 1.258759884181224,
                                      -8.302866610677315, -1.2563833805673688,
                                      28.324298392677846, 28.09315509472324,
                                      -29.378744559315365, -11.247473567051799,
                                      11.119951466671642, -87.08582317485761,
                                      -20.93948523898559, -125.73267675714658,
                                      -35.13852440758523])

    # Testing pure-python and fortran versions of Gaussian-neural on different
    # number of processes

    for fortran in [False, True]:
        for cores in range(1, 4):
            label = 'train-periodic/%s-%i' % (fortran, cores)
            print label
            calc = Amp(descriptor=Gaussian(cutoff=4.,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='tanh'),
                       label=label,
                       dblabel=label,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)

            assert (abs(calc.model.lossfunction.loss -
                        ref_loss) < 10.**(-4.)), \
                'The calculated value of loss function is wrong!'

            assert (abs(calc.model.lossfunction.energy_loss -
                        ref_energyloss) < 10.**(-5.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.model.lossfunction.force_loss -
                        ref_forceloss) < 10 ** (-3.)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                assert(abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_]) < 10 ** (-4.)), \
                    'The calculated value of loss function derivative is \
                    wrong!'

            dblabel = label
            secondlabel = '_' + label

            calc = Amp(descriptor=Gaussian(cutoff=4.,
                                           Gs=Gs,
                                           fortran=fortran),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='tanh',),
                       label=secondlabel,
                       dblabel=dblabel,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)

            assert (abs(calc.model.lossfunction.loss -
                        ref_loss) < 10.**(-4.)), \
                'The calculated value of loss function is wrong!'

            assert (abs(calc.model.lossfunction.energy_loss -
                        ref_energyloss) < 10.**(-5.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.model.lossfunction.force_loss -
                        ref_forceloss) < 10 ** (-3.)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                assert(abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_]) < 10 ** (-4.)), \
                    'The calculated value of loss function derivative is \
                    wrong!'


if __name__ == '__main__':
    non_periodic_0th_bfgs_step_test()
    periodic_0th_bfgs_step_test()