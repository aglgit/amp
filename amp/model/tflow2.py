import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from . import LossFunction
from ..utilities import ConvergenceOccurred


class NeuralNetwork(Model):
    def __init__(self,
                 convergence=None,
                 energy_coefficient=1.0,
                 force_coefficient=0.04,
                 hidden_layers=(10, 10),
                 activation='tanh',
                 optimizer='adam',
                 init_lr=4e-3,
                 regularization_strength=None,
                 keep_prob=1.,
                 batch_size=1,
                 ):
        self.import_name = '.model.neuralnetwork.tflow2'

        if convergence is None:
            convergence = {'energy_rmse': 1e-3,
                           'force_rmse': None,
                           'max_steps': int(1e3), }

        self.convergence = convergence
        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient

        self.hidden_layer_list = []
        for h in hidden_layers:
            self.hidden_layer_list.append(Dense(h, activation=activation))
        self.hidden_layer_list.append(Dense(1, activation="linear"))

        optimizer_args = {'optimizer': optimizer, 'learning_rate': init_lr}
        self.optimizer = tf.keras.optimizers.get(optimizer_args)

        self.keep_prob = keep_prob
        self.batch_size = batch_size

        if force_coefficient is None:
            self.forcetraining = False
        else:
            self.forcetraining = True

    def fit(self, trainingimages, descriptor, parallel, log=None):
        self.log = log

        fingerprintDB = descriptor.fingerprints
        if self.parameters['force_coefficient'] is None:
            fingerprintDerDB = None
        else:
            fingerprintDerDB = descriptor.fingerprintprimes

        keys = trainingimages.keys()
        num_images = len(keys)
        energies = []
        forces = []
        for i in range(num_images):
            atoms = trainingimages[keys[i]]
            energies.append(atoms.get_potential_energy())
            if self.forcetraining:
                forces.append(atoms.get_forces())

        try:
            self.train_model()
        except ConvergenceOccurred:
            return True

        return False

    def calculate_energy(self, fingerprint):
        """Get the energy by feeding in a list to the get_list version (which
        is more efficient for anything greater than 1 image)."""
        key = '1'
        energies, forces = self.get_energy_list([key], {key: fingerprint})
        return energies[0]

    def calculate_forces(self, fingerprint, derfingerprint):
        # calculate_forces function still needs to be implemented. Can't do
        # this without the fingerprint derivates working properly though
        key = '1'
        energies, forces = \
            self.get_energy_list([key],
                                 {key: fingerprint},
                                 fingerprintDerDB={key: derfingerprint},
                                 forces=True)
        return forces[0][0:len(fingerprint)]

def generateTensorFlowArrays(fingerprintDB, elements, keylist,
                             fingerprintDerDB=None):
    """
    This function generates the inputs to the tensorflow graph for the selected
    images.
    The essential problem is that each neural network is associated with a
    specific element type. Thus, atoms in each ASE image need to be sent to
    different networks.

    Inputs:

    fingerprintDB: a database of fingerprints, as taken from the descriptor

    elements: a list of element types (e.g. 'C','O', etc)

    keylist: a list of hashs into the fingerprintDB that we want to create
             inputs for

    fingerprintDerDB: a database of fingerprint derivatives, as taken from the
                      descriptor

    Outputs:

    atomArraysAll: a dictionary of fingerprint inputs to each element's neural
        network

    nAtomsDict: a dictionary for each element with lists of the number of
        atoms of each type in each image

    atomsIndsReverse: a dictionary that contains the index of each atom into
        the original keylist

    nAtoms: the number of atoms in each image

    atomArraysAllDerivs: dictionary of fingerprint derivates for each
        element's neural network
    """

    nAtomsDict = {}
    keylist = list(keylist)
    for element in elements:
        nAtomsDict[element] = np.zeros(len(keylist))

    for j in range(len(keylist)):
        fp = fingerprintDB[keylist[j]]
        atomSymbols, fpdata = zip(*fp)
        for i in range(len(fp)):
            nAtomsDict[atomSymbols[i]][j] += 1

    atomsPositions = {}
    for element in elements:
        atomsPositions[element] = np.cumsum(
            nAtomsDict[element]) - nAtomsDict[element]

    atomsIndsReverse = {}
    for element in elements:
        atomsIndsReverse[element] = []
        for i in range(len(keylist)):
            if nAtomsDict[element][i] > 0:
                atomsIndsReverse[element].append(
                    np.ones((nAtomsDict[element][i].astype(np.int64), 1)) * i)
        if len(atomsIndsReverse[element]) > 0:
            atomsIndsReverse[element] = np.concatenate(
                atomsIndsReverse[element])

    atomArraysAll = {}
    for element in elements:
        atomArraysAll[element] = []

    natoms = np.zeros((len(keylist), 1))
    for j in range(len(keylist)):
        fp = fingerprintDB[keylist[j]]
        atomSymbols, fpdata = zip(*fp)
        atomdata = zip(atomSymbols, range(len(atomSymbols)))
        for element in elements:
            atomArraysTemp = []
            curatoms = [atom for atom in atomdata if atom[0] == element]
            for i in range(len(curatoms)):
                atomArraysTemp.append(fp[curatoms[i][1]][1])
            if len(atomArraysTemp) > 0:
                atomArraysAll[element].append(atomArraysTemp)
        natoms[j] = len(atomSymbols)

    for element in elements:
        if len(atomArraysAll[element]) > 0:
            atomArraysAll[element] = np.concatenate(atomArraysAll[element])
        else:
            atomArraysAll[element] = []

    # Set up the array for atom-based fingerprint derivatives.

    dgdx = {}
    dgdx_Eindices = {}
    dgdx_Xindices = {}
    for element in elements:
        dgdx[element] = []  # Nxlen(fp)x3 array
        dgdx_Eindices[element] = []  # Nx1 array of which dE/dg to pull
        dgdx_Xindices[element] = []
        # Nx1 array representing which atom this force will represent
    if fingerprintDerDB is not None:
        for j in range(len(keylist)):
            fp = fingerprintDB[keylist[j]]
            fpDer = fingerprintDerDB[keylist[j]]
            atomSymbols, fpdata = zip(*fp)
            atomdata = list(zip(atomSymbols, range(len(atomSymbols))))

            for element in elements:
                curatoms = [atom for atom in atomdata if atom[0] == element]
                dgdx_temp = []
                dgdx_Eindices_temp = []
                dgdx_Xindices_temp = []
                if len(curatoms) > 0:
                    for i in range(len(curatoms)):
                        for k in range(len(atomdata)):
                            # check if fp derivative is present
                            dictkeys = [(k, atomdata[k][0], curatoms[
                                i][1], curatoms[i][0], 0),
                                (k, atomdata[k][0], curatoms[
                                 i][1], curatoms[i][0], 1),
                                (k, atomdata[k][0], curatoms[
                                 i][1], curatoms[i][0], 2)]
                            if ((dictkeys[0] in fpDer) or
                               (dictkeys[1] in fpDer) or
                                    (dictkeys[2] in fpDer)):
                                fptemp = []
                                for ix in range(3):
                                    dictkey = (k, atomdata[k][0], curatoms[
                                        i][1], curatoms[i][0], ix)
                                    fptemp.append(fpDer[dictkey])
                                dgdx_temp.append(np.array(fptemp).transpose())
                                dgdx_Eindices_temp.append(i)
                                dgdx_Xindices_temp.append(k)
                if len(dgdx_Eindices_temp) > 0:
                    dgdx[element].append(np.array(dgdx_temp))
                    dgdx_Eindices[element].append(np.array(dgdx_Eindices_temp))
                    dgdx_Xindices[element].append(np.array(dgdx_Xindices_temp))
                else:
                    dgdx[element].append([])
                    dgdx_Eindices[element].append([])
                    dgdx_Xindices[element].append([])
    return (atomArraysAll, nAtomsDict, atomsIndsReverse,
            natoms, dgdx, dgdx_Eindices, dgdx_Xindices)
