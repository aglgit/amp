import os
import shutil
import numpy as np
import tempfile
import multiprocessing as mp
import gc
import sqlite3
from datetime import datetime
from collections import OrderedDict
from scipy.optimize import fmin_bfgs

from ase.calculators.calculator import Calculator, Parameters
from ase.data import atomic_numbers
from ase.parallel import paropen
from ase import io as aseio
from ase.calculators.neighborlist import NeighborList

from .utilities import make_filename, load_parameters, ConvergenceOccurred, IO
from .utilities import (TrainingConvergenceError, ExtrapolateError,
                        hash_images)
from .utilities import Logger, save_parameters, string2dict
from .descriptor import Behler
from .model.neuralnetwork import NeuralNetwork

###############################################################################



class Amp(Calculator):

    """
    Atomistic Machine-Learning Potential (Amp) ASE calculator

    :param descriptor: Class representing local atomic environment. Can be
                        only None and Behler for now. Input arguments for
                        Behler are cutoff and Gs; for more information see
                        docstring for the class Behler.
    :type descriptor: object
    :param regression: Class representing the regression method. Can be only
                       NeuralNetwork for now. Input arguments for NeuralNetwork
                       are hiddenlayers, activation, weights, and scalings; for
                       more information see docstring for the class
                       NeuralNetwork.
    :type regression: object
    :param load: Path for loading an existing parameters of Amp calculator.
    :type load: str
    :param label: Default prefix/location used for all files.
    :type label: str
    :param dblabel: Optional separate prefix/location for database files,
                    including fingerprints, fingerprint derivatives, and
                    neighborlists. This file location can be shared between
                    calculator instances to avoid re-calculating redundant
                    information. If not supplied, just uses the value from
                    label.
    :type dblabel: str

    :param cores: Can specify cores to use for parallel training;
                  if None, will determine from environment
    :type cores: int

    :raises: RuntimeError
    """
    implemented_properties = ['energy', 'forces']

    #FIXME/ap This needs to be made compatible with the save function.
    # It should be do-able.
    #default_parameters = {
    #    'descriptor': None,
    #    'model': None,
    #    'extrapolate': True,
    #}

    ###########################################################################

    def __init__(self, load=None, label=None, dblabel=None, fortran=None,
                 cores=None, **kwargs):
        
        #FIXME/ap: Make sure this plays well with Amp.load.
        self.fortran = fortran

        #FIXME: Need to clean this up.

        #FIXME: The result of this is that we have both the model and descriptor
        # saved with two names, e.g., self.model and self.parameters.regression.
        # Clean this up to avoid confusion.
        self.dblabel = label if dblabel is None else dblabel


        if cores is None:
            from .utilities import count_allocated_cpus
            cores = count_allocated_cpus()
        self.cores = cores

        Calculator.__init__(self, label=label, **kwargs)

        param = self.parameters

        if param.descriptor is not None:
            self.descriptor = param.descriptor
            if hasattr(self.descriptor, 'dblabel'):
                if self.descriptor.dblabel is None:
                    self.descriptor.dblabel = self.dblabel

        self.model = param.model

    ###########################################################################

    @classmethod
    def load(Cls, filename, Descriptor=None, Model=None, **kwargs):
        """Attempts to load calculators and return a new instance of Amp.
        Only a filename is required, in typical cases.

        If using a home-rolled descriptor or model, also supply
        uninstantiated classes to those models, as in model=MyModel.

        Any additional keyword arguments (such as fortran=True) can be
        fed through to Amp.
        """
        with open(filename) as f:
            text = f.read()

        # Unpack parameter dictionaries.
        p = string2dict(text)
        for key in ['descriptor', 'model']:
            p[key] = string2dict(p[key])

        # If modules are not specified, find them.
        if Descriptor is None:
            Descriptor = importhelper(p['descriptor']['importname'])
        if Model is None:
            Model = importhelper(p['model']['importname'])
        # Clean out the importname keyword.
        for key in ['descriptor', 'model']:
            if 'importname' in p[key]:
                p[key].pop('importname')

        # Instantiate the descriptor and model.
        descriptor = Descriptor(**p['descriptor'])
        model = Model(**p['model'])

        # Instantiate Amp.
        calc = Cls(descriptor=descriptor, model=model, **kwargs)
        return calc

    def set(self, **kwargs):
        """
        Function to set parameters.
        """
        changed_parameters = Calculator.set(self, **kwargs)
        # FIXME. Decide whether to call reset. Decide if this is
        # meaningful in our implementation!
        if len(changed_parameters) > 0:
            self.reset()

    ###########################################################################

    def set_label(self, label):
        """
        Sets label, ensuring that any needed directories are made.

        :param label: Default prefix/location used for all files.
        :type label: str
        """
        Calculator.set_label(self, label)

        # Create directories for output structure if needed.
        if self.label:
            if (self.directory != os.curdir and
                    not os.path.isdir(self.directory)):
                os.makedirs(self.directory)

    ###########################################################################

    def initialize(self, atoms):
        """
        :param atoms: ASE atoms object.
        :type atoms: ASE dict
        """
        #FIXME/ap This needs a description of what this method does.
        # Specificially why it is necessary in addition to the regular
        # __init__ method. If it is required by the ASE Calculator
        # format, we should state that.
        self.par = {}
        self.rc = 0.0 # FIXME/ap What is this? It is set and never used,
                      # except just below.
        self.numbers = atoms.get_atomic_numbers()
        self.forces = np.empty((len(atoms), 3))
        self.nl = NeighborList([0.5 * self.rc + 0.25] * len(atoms),
                               self_interaction=False)

    ###########################################################################

    def calculate(self, atoms, properties, system_changes):
        """
        Calculation of the energy of system and forces of all atoms.
        """
        # The below method just sets the atoms object, if specified,
        # to self.atoms.
        Calculator.calculate(self, atoms, properties, system_changes)
        # FIXME/ap: The ASE Calculator has atoms=None as the default; that
        # is, not changing the atoms that may be saved. We don't. Why?
        # I think it is becuase this is always called from Calculator.
        # get_property, which could have atoms=None already.

        if self.label is not None:
            log = Logger(make_filename(self.label, 'call-log.txt'))
        else:
            log = Logger(None)

        log('Amp calculate started.')
        images = hash_images([self.atoms])
        key = images.keys()[0]
        self.descriptor.calculate_fingerprints(images=images, log=log)

        if properties == ['energy']:
            energy = self.model.get_energy(self.descriptor.fingerprints[key])
            self.results['energy'] = energy

        if properties == ['forces']:
            #FIXME/ap needs updating.
            raise NotImplementedError('Needs updating.')

    def train(self,
              images,
              overwrite=False,
              ):
        """
        Fits a variable set to the data, by default using the "fmin_bfgs"
        optimizer. The optimizer takes as input a cost function to reduce and
        an initial guess of variables and returns an optimized variable set.

        :param images: List of ASE atoms objects with positions, symbols,
                       energies, and forces in ASE format. This is the training
                       set of data. This can also be the path to an ASE
                       trajectory (.traj) or database (.db) file. Energies can
                       be obtained from any reference, e.g. DFT calculations.
        :type images: list or str
        :param overwrite: If a trained output file with the same name exists,
                          overwrite it.
        :type overwrite: bool
        """
        log = Logger(make_filename(self.label, 'train-log.txt'))
        log('Amp training started. ' + now() + '\n')
        
        log('Checking input parameters.')
        images = hash_images(images)

        log('Local environment descriptor: %s' %
                self.parameters.descriptor.__class__.__name__)
        log('Model: %s\n' % self.parameters.model.__class__.__name__)

        self.descriptor.calculate_fingerprints(images=images,
                                               cores=self.cores,
                                               fortran=self.fortran,
                                               log=log)

        result = self.model.fit(trainingimages=images,
                                fingerprints=self.descriptor,
                                fortran=self.fortran,
                                log=log)

        if result is True:
            log('Amp successfully trained. Saving current parameters.')
            self.save(self.label + '.amp', overwrite)
            log('Parameters saved.')
        else:
            log('Amp not trained successfully. Saving current parameters.')
            self.save(make_filename(self.label, '-untrained-parameters.amp'),
                      overwrite)
            log('Parameters saved.')

    def save(self, filename, overwrite=False):
        """Saves the calculator in way that it can be re-opened with
        load."""
        if os.path.exists(filename):
            if overwrite is False:
                raise RuntimeError('File exists')
            print('Overwriting file: %s. Saving original as %s.backup'
                  % (filename, filename))
            #FIXME. May be better to assure a unique filename with
            # tempfile.NamedTemporaryFile(delete=False, ...), 
            # but need to figure out absolute/relative path.
            # Or do this in the make_filename routine, possibly
            # with uuid instead of tempfile.
            shutil.move(filename, '%s.backup' % filename)
        descriptor = self.descriptor.tostring()
        model = self.model.tostring()
        p = Parameters({'descriptor': descriptor,
                        'model': model})
        p.write(filename)
        #FIXME/ap. I should get this to work with the todict method
        # of standard ASE calculators.


###############################################################################
###############################################################################
###############################################################################


class MultiProcess:

    """
    Class to do parallel processing, using multiprocessing package which works
    on Python versions 2.6 and above.

    :param fortran: If True, allows for extrapolation, if False, does not
                    allow.
    :type fortran: bool
    :param no_procs: Number of processors.
    :type no_procs: int
    """
    ###########################################################################

    def __init__(self, fortran, no_procs):

        self.fortran = fortran
        self.no_procs = no_procs

    ###########################################################################

    def make_list_of_sub_images(self, no_of_images, hashs, images):
        """
        Two lists are made each with one entry per core. The entry of the first
        list contains list of hashs to be calculated by that core, and the
        entry of the second list contains dictionary of images to be calculated
        by that core.

        :param no_of_images: Total number of images.
        :type no_of_images: int
        :param hashs: Unique keys, one key per image.
        :type hashs: list
        :param images: List of ASE atoms objects (the training set).
        :type images: list
        """
        quotient = int(no_of_images / self.no_procs)
        remainder = no_of_images - self.no_procs * quotient
        list_sub_hashs = [None] * self.no_procs
        list_sub_images = [None] * self.no_procs
        count0 = 0
        count1 = 0
        while count0 < self.no_procs:
            if count0 < remainder:
                len_sub_hashs = quotient + 1
            else:
                len_sub_hashs = quotient
            sub_hashs = [None] * len_sub_hashs
            sub_images = {}
            count2 = 0
            while count2 < len_sub_hashs:
                hash = hashs[count1]
                sub_hashs[count2] = hash
                sub_images[hash] = images[hash]
                count1 += 1
                count2 += 1
            list_sub_hashs[count0] = sub_hashs
            list_sub_images[count0] = sub_images
            count0 += 1

        self.list_sub_hashs = list_sub_hashs
        self.list_sub_images = list_sub_images

        del hashs, images, list_sub_hashs, list_sub_images, sub_hashs,
        sub_images

    ###########################################################################

    def share_fingerprints_task_between_cores(self, task, _args):
        """
        Fingerprint tasks are sent to cores for parallel processing.

        :param task: Function to be called on each process.
        :type task: function
        :param _args: Arguments to be fed to the function on each process.
        :type _args: function
        """
        args = {}
        count = 0
        while count < self.no_procs:
            sub_hashs = self.list_sub_hashs[count]
            sub_images = self.list_sub_images[count]
            args[count] = (count,) + (sub_hashs, sub_images,) + _args
            count += 1

        processes = [mp.Process(target=task, args=args[_])
                     for _ in range(self.no_procs)]

        count = 0
        while count < self.no_procs:
            processes[count].start()
            count += 1

        count = 0
        while count < self.no_procs:
            processes[count].join()
            count += 1

        count = 0
        while count < self.no_procs:
            processes[count].terminate()
            count += 1

        del sub_hashs, sub_images

    ###########################################################################

    def ravel_images_data(self,
                          param,
                          sfp,
                          snl,
                          elements,
                          train_forces,
                          log,
                          save_memory,):
        """
        Reshape data of images into lists.

        :param param: ASE dictionary object.
        :type param: dict
        :param sfp: SaveFingerprints object.
        :type sfp: object
        :param snl: SaveNeighborLists object.
        :type snl: object
        :param elements: List if elements in images.
        :type elements: list
        :param train_forces: Determining whether forces are also trained or
                             not.
        :type train_forces: bool
        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param save_memory: If True, memory efficient mode will be used.
        :type save_memory: bool
        """
        log('Re-shaping images data to send to fortran90...')
        log.tic()

        self.train_forces = train_forces

        if param.descriptor is None:
            self.fingerprinting = False
        else:
            self.fingerprinting = True

        self.no_of_images = {}
        self.real_energies = {}
        x = 0
        while x < self.no_procs:
            self.no_of_images[x] = len(self.list_sub_hashs[x])
            self.real_energies[x] = \
                [self.list_sub_images[x][
                    hash].get_potential_energy(apply_constraint=False)
                    for hash in self.list_sub_hashs[x]]
            x += 1

        if self.fingerprinting:

            self.no_of_atoms_of_images = {}
            self.atomic_numbers_of_images = {}
            self.raveled_fingerprints_of_images = {}
            x = 0
            while x < self.no_procs:
                self.no_of_atoms_of_images[x] = \
                    [len(self.list_sub_images[x][hash])
                     for hash in self.list_sub_hashs[x]]
                self.atomic_numbers_of_images[x] = \
                    [atomic_numbers[atom.symbol]
                     for hash in self.list_sub_hashs[x]
                     for atom in self.list_sub_images[x][hash]]
                self.raveled_fingerprints_of_images[x] = \
                    ravel_fingerprints_of_images(self.list_sub_hashs[x],
                                                 self.list_sub_images[x],
                                                 sfp, save_memory,)
                x += 1
        else:
            self.atomic_positions_of_images = {}
            x = 0
            while x < self.no_procs:
                self.atomic_positions_of_images[x] = \
                    [self.list_sub_images[x][hash].positions.ravel()
                     for hash in self.list_sub_hashs[x]]
                x += 1

        if train_forces is True:

            self.real_forces = {}
            x = 0
            while x < self.no_procs:
                self.real_forces[x] = \
                    [self.list_sub_images[x][hash].get_forces(
                        apply_constraint=False)[index]
                     for hash in self.list_sub_hashs[x]
                     for index in range(len(self.list_sub_images[x][hash]))]
                x += 1

            if self.fingerprinting:
                self.list_of_no_of_neighbors = {}
                self.raveled_neighborlists = {}
                self.raveled_der_fingerprints = {}
                x = 0
                while x < self.no_procs:
                    (self.list_of_no_of_neighbors[x],
                     self.raveled_neighborlists[x],
                     self.raveled_der_fingerprints[x]) = \
                        ravel_neighborlists_and_der_fingerprints_of_images(
                        self.list_sub_hashs[x],
                        self.list_sub_images[x],
                        sfp,
                        snl,
                        save_memory,)
                    x += 1

        log(' ...data re-shaped.', toc=True)

    ###########################################################################

    def share_cost_function_task_between_cores(self, task, _args,
                                               len_of_variables):
        """
        Cost function and its derivatives of with respect to variables are
        calculated in parallel.

        :param task: Function to be called on each process.
        :type task: function
        :param _args: Arguments to be fed to the function on each process.
        :type _args: function
        :param len_of_variables: Number of variables.
        :type len_of_variables: int
        """
        queues = {}
        x = 0
        while x < self.no_procs:
            queues[x] = mp.Queue()
            x += 1

        args = {}
        x = 0
        while x < self.no_procs:
            if self.fortran:
                args[x] = _args + (queues[x],)
            else:
                sub_hashs = self.list_sub_hashs[x]
                sub_images = self.list_sub_images[x]
                args[x] = (sub_hashs, sub_images,) + _args + (queues[x],)
            x += 1

        energy_square_error = 0.
        force_square_error = 0.

        der_variables_square_error = [0.] * len_of_variables

        processes = [mp.Process(target=task, args=args[_])
                     for _ in range(self.no_procs)]

        x = 0
        while x < self.no_procs:
            if self.fortran:
                # data particular to each process is sent to fortran modules
                self.send_data_to_fortran(x,)
            processes[x].start()
            x += 1

        x = 0
        while x < self.no_procs:
            processes[x].join()
            x += 1

        x = 0
        while x < self.no_procs:
            processes[x].terminate()
            x += 1

        sub_energy_square_error = []
        sub_force_square_error = []
        sub_der_variables_square_error = []

        # Construct total square_error and derivative with respect to variables
        # from subprocesses
        results = {}
        x = 0
        while x < self.no_procs:
            results[x] = queues[x].get()
            x += 1

        sub_energy_square_error = [results[_][0] for _ in range(self.no_procs)]
        sub_force_square_error = [results[_][1] for _ in range(self.no_procs)]
        sub_der_variables_square_error = [results[_][2]
                                          for _ in range(self.no_procs)]

        _ = 0
        while _ < self.no_procs:
            energy_square_error += sub_energy_square_error[_]
            force_square_error += sub_force_square_error[_]
            count = 0
            while count < len_of_variables:
                der_variables_square_error[count] += \
                    sub_der_variables_square_error[_][count]
                count += 1
            _ += 1

        if not self.fortran:
            del sub_hashs, sub_images

        return (energy_square_error, force_square_error,
                der_variables_square_error)

    ###########################################################################

    def send_data_to_fortran(self, x,):
        """
        Function to send images data to fortran90 code.

        :param x: The number of process.
        :type x: int
        """
        fmodules.images_props.no_of_images = self.no_of_images[x]
        fmodules.images_props.real_energies = self.real_energies[x]

        if self.fingerprinting:
            fmodules.images_props.no_of_atoms_of_images = \
                self.no_of_atoms_of_images[x]
            fmodules.images_props.atomic_numbers_of_images = \
                self.atomic_numbers_of_images[x]
            fmodules.fingerprint_props.raveled_fingerprints_of_images = \
                self.raveled_fingerprints_of_images[x]
        else:
            fmodules.images_props.atomic_positions_of_images = \
                self.atomic_positions_of_images[x]

        if self.train_forces is True:

            fmodules.images_props.real_forces_of_images = \
                self.real_forces[x]

            if self.fingerprinting:
                fmodules.images_props.list_of_no_of_neighbors = \
                    self.list_of_no_of_neighbors[x]
                fmodules.images_props.raveled_neighborlists = \
                    self.raveled_neighborlists[x]
                fmodules.fingerprint_props.raveled_der_fingerprints = \
                    self.raveled_der_fingerprints[x]

###############################################################################
###############################################################################
###############################################################################


class SaveNeighborLists:

    """
    Neighborlists for all images with the given cutoff value are calculated
    and saved. As well as neighboring atomic indices, neighboring atomic
    offsets from the main cell are also saved. Offsets become important when
    dealing with periodic systems. Neighborlists are generally of two types:
    Type I which consists of atoms within the cutoff distance either in the
    main cell or in the adjacent cells, and Type II which consists of atoms in
    the main cell only and within the cutoff distance.

    :param cutoff: Cutoff radius, in Angstroms, around each atom.
    :type cutoff: float
    :param no_of_images: Total number of images.
    :type no_of_images: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param label: Prefix name used for all files.
    :type label: str
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param train_forces: Determining whether forces are also trained or not.
    :type train_forces: bool
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    """
    ###########################################################################

    def __init__(self, cutoff, no_of_images, hashs, images, label, log,
                 train_forces, io, data_format, save_memory):

        self.cutoff = cutoff
        self.images = images

        if train_forces is True:
            new_images = images
            log('Calculating neighborlist for each atom...')
            log.tic()
            if data_format is 'json':
                filename = make_filename(label, 'neighborlists.json')
            elif data_format is 'db':
                filename = make_filename(label, 'neighborlists.db')
            self.ncursor = None
            if save_memory:
                nconn = sqlite3.connect(filename)
                self.ncursor = nconn.cursor()
                # Create table
                self.ncursor.execute('''CREATE TABLE IF NOT EXISTS
                neighborlists (image text, atom integer, nl_index integer,
                neighbor_atom integer,
                offset1 integer, offset2 integer, offset3 integer)''')
            else:
                self.nl_data = {}
            if not os.path.exists(filename):
                log(' No saved neighborlist file found.')
            else:
                if save_memory:
                    self.ncursor.execute("SELECT image FROM neighborlists")
                    old_hashs = set([_ for _ in self.ncursor])
                else:
                    log.tic('read_neighbors')
                    old_hashs, self.nl_data = io.read(filename,
                                                      'neighborlists',
                                                      self.nl_data,
                                                      data_format)
                    log(' Saved neighborlist file %s loaded with %i entries.'
                        % (filename, len(old_hashs)), toc='read_neighbors')

                new_images = {}
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    if hash not in old_hashs:
                        new_images[hash] = images[hash]
                    count += 1
                del old_hashs

            log(' Calculating %i of %i neighborlists (%i exist in file).'
                % (len(new_images), len(images),
                   len(images) - len(new_images)))

            if len(new_images) != 0:
                log.tic('calculate_neighbors')
                new_hashs = sorted(new_images.keys())
                no_of_new_images = len(new_hashs)
                count = 0
                while count < no_of_new_images:
                    hash = new_hashs[count]
                    image = new_images[hash]
                    no_of_atoms = len(image)
                    nl = NeighborList(cutoffs=([self.cutoff / 2.] *
                                               no_of_atoms),
                                      self_interaction=False,
                                      bothways=True, skin=0.)
                    # FIXME: Is update necessary?
                    nl.update(image)
                    # FIXME/ap: It seems so, to know the image.

                    if save_memory:
                        self_index = 0
                        while self_index < no_of_atoms:
                            neighbor_indices, neighbor_offsets = \
                                nl.get_neighbors(self_index)
                            if len(neighbor_offsets) == 0:
                                n_self_offsets = [[0, 0, 0]]
                            else:
                                n_self_offsets = \
                                    np.vstack(([[0, 0, 0]],
                                               neighbor_offsets))
                            n_self_indices = np.append(self_index,
                                                       neighbor_indices)
                            len_of_neighbors = len(n_self_indices)
                            _ = 0
                            while _ < len_of_neighbors:
                                value = n_self_offsets[_]
                                # Insert a row of data
                                row = (hash, self_index, _,
                                       n_self_indices[_], value[0],
                                       value[1], value[2])
                                self.ncursor.execute('''INSERT INTO
                                neighborlists VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                                     row)
                                _ += 1
                            self_index += 1

                    else:
                        self.nl_data[hash] = {}
                        self_index = 0
                        while self_index < no_of_atoms:
                            neighbor_indices, neighbor_offsets = \
                                nl.get_neighbors(self_index)
                            if len(neighbor_offsets) == 0:
                                n_self_offsets = [[0, 0, 0]]
                            else:
                                n_self_offsets = \
                                    np.vstack(([[0, 0, 0]], neighbor_offsets))
                            n_self_indices = np.append(self_index,
                                                       neighbor_indices)
                            self.nl_data[hash][self_index] = \
                                (n_self_indices, n_self_offsets,)
                            self_index += 1

                    count += 1
                del new_hashs

                if save_memory:
                    nconn.commit()
                else:
                    io.save(filename, 'neighborlists', self.nl_data,
                            data_format)

                log(' ...neighborlists calculated and saved to %s.' %
                    filename, toc='calculate_neighbors')

            del new_images
        del images, self.images

###############################################################################
###############################################################################
###############################################################################


class SaveFingerprints:

    """
    Memory class to not recalculate fingerprints and their derivatives if
    not necessary. This could cause runaway memory usage; use with caution.

    :param fp: Fingerprint object.
    :type fp: object
    :param elements: List if elements in images.
    :type elements: list
    :param no_of_images: Total number of images.
    :type no_of_images: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param label: Prefix name used for all files.
    :type label: str
    :param train_forces: Determining whether forces are also trained or not.
    :type train_forces: bool
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param _mp: MultiProcess object.
    :type _mp: object
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    """
    ###########################################################################

    def __init__(self, fp, elements, no_of_images, hashs, images, label,
                 train_forces, snl, log, _mp, io, data_format, save_memory):

        self.Gs = fp.Gs
        self.train_forces = train_forces
        new_images = images

        log('Calculating atomic fingerprints...')
        log.tic()
        if data_format is 'json':
            filename = make_filename(label, 'fingerprints.json')
        elif data_format is 'db':
            filename = make_filename(label, 'fingerprints.db')
        self.fcursor = None
        if save_memory:
            fconn = sqlite3.connect(filename)
            self.fcursor = fconn.cursor()
            # Create table
            self.fcursor.execute('''CREATE TABLE IF NOT EXISTS fingerprints
            (image text, atom integer, fp_index integer, value real)''')
        else:
            self.fp_data = {}
        if not os.path.exists(filename):
            log('No saved fingerprint file found.')
        else:
            if save_memory:
                self.fcursor.execute("SELECT image FROM fingerprints")
                old_hashs = set([_ for _ in self.fcursor])
            else:
                log.tic('read_fps')
                old_hashs, self.fp_data = io.read(filename, 'fingerprints',
                                                  self.fp_data, data_format)
                log(' Saved fingerprints file %s loaded with %i entries.'
                    % (filename, len(old_hashs)), toc='read_fps')

            new_images = {}
            count = 0
            while count < no_of_images:
                hash = hashs[count]
                if hash not in old_hashs:
                    new_images[hash] = images[hash]
                count += 1
            del old_hashs

        log(' Calculating %i of %i fingerprints (%i exist in file).'
            % (len(new_images), len(images), len(images) - len(new_images)))

        if len(new_images) != 0:
            log.tic('calculate_fps')
            new_hashs = sorted(new_images.keys())
            no_of_new_images = len(new_hashs)
            # new images are shared between cores for fingerprint calculations
            _mp.make_list_of_sub_images(no_of_new_images, new_hashs,
                                        new_images)
            del new_hashs

            # Temporary files to hold child fingerprint calculations.
            if data_format is 'json':
                childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                          suffix='.json')
                              for _ in range(_mp.no_procs)]
            elif data_format is 'db':
                childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                          suffix='.db')
                              for _ in range(_mp.no_procs)]
            _ = 0
            while _ < _mp.no_procs:
                log('  Processor %i calculations stored in file %s.'
                    % (_, childfiles[_].name))
                _ += 1

            task_args = (fp, childfiles, io, data_format, save_memory,)
            _mp.share_fingerprints_task_between_cores(
                task=_calculate_fingerprints, _args=task_args)

            log(' Calculated %i new images.' % no_of_new_images,
                toc='calculate_fps')

            log.tic('read_fps')
            log(' Reading calculated fingerprints...')
            if save_memory:
                for f in childfiles:
                    _fconn = sqlite3.connect(f.name)
                    _fcursor = _fconn.cursor()
                    for row in _fcursor.execute("SELECT * FROM fingerprints"):
                        self.fcursor.execute('''INSERT INTO fingerprints
                        VALUES (?, ?, ?, ?)''', row)
                fconn.commit()
            else:
                for f in childfiles:
                    _, self.fp_data = io.read(f.name, 'fingerprints',
                                              self.fp_data, data_format)
                io.save(filename, 'fingerprints', self.fp_data, data_format)
            log(' ...child fingerprints read and saved to %s.' % filename,
                toc='read_fps')

        fingerprint_values = {}
        for element in elements:
            fingerprint_values[element] = {}
            len_of_fingerprint = len(self.Gs[element])
            _ = 0
            while _ < len_of_fingerprint:
                fingerprint_values[element][_] = []
                _ += 1

        count = 0
        while count < no_of_images:
            hash = hashs[count]
            image = images[hash]
            no_of_atoms = len(image)
            index = 0
            while index < no_of_atoms:
                symbol = image[index].symbol
                len_of_fingerprint = len(self.Gs[symbol])
                _ = 0
                while _ < len_of_fingerprint:
                    if save_memory:
                        self.fcursor.execute('''SELECT * FROM fingerprints
                        WHERE image=? AND atom=? AND fp_index=?''',
                                             (hash, index, _))
                        row = self.fcursor.fetchall()
                        fingerprint_values[symbol][_].append(row[0][3])
                    else:
                        fingerprint_values[symbol][_].append(
                            self.fp_data[hash][index][_])
                    _ += 1
                index += 1
            count += 1
        del _

        fingerprints_range = OrderedDict()
        for element in elements:
            fingerprints_range[element] = \
                [[min(fingerprint_values[element][_]),
                  max(fingerprint_values[element][_])]
                 for _ in range(len(self.Gs[element]))]

        self.fingerprints_range = fingerprints_range

        del new_images, fingerprint_values

        if train_forces is True:
            new_images = images
            log('''Calculating derivatives of atomic fingerprints
                with respect to coordinates...''')
            log.tic()
            if data_format is 'json':
                filename = make_filename(label,
                                         'fingerprint-derivatives.json')
            elif data_format is 'db':
                filename = make_filename(label,
                                         'fingerprint-derivatives.db')
            self.fdcursor = None
            if save_memory:
                fdconn = sqlite3.connect(filename)
                self.fdcursor = fdconn.cursor()
                # Create table
                self.fdcursor.execute('''CREATE TABLE IF NOT EXISTS
                fingerprint_derivatives
                (image text, atom integer, neighbor_atom integer,
                 direction integer, fp_index integer, value real)''')
            else:
                self.der_fp_data = {}
            if not os.path.exists(filename):
                log('No saved fingerprint-derivatives file found.')
            else:
                if save_memory:
                    self.fdcursor.execute(
                        "SELECT image FROM fingerprint_derivatives")
                    old_hashs = set([_ for _ in self.fdcursor])
                else:
                    log.tic('read_der_fps')
                    old_hashs, self.der_fp_data = \
                        io.read(filename, 'fingerprint_derivatives',
                                self.der_fp_data, data_format)
                    log(' Saved fingerprint derivatives file %s loaded '
                        'with %i entries.' % (filename, len(old_hashs)),
                        toc='read_der_fps')

                new_images = {}
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    if hash not in old_hashs:
                        new_images[hash] = images[hash]
                    count += 1
                del old_hashs

            log(' Calculating %i of %i fingerprint derivatives '
                '(%i exist in file).'
                % (len(new_images), len(images),
                   len(images) - len(new_images)))

            if len(new_images) != 0:
                log.tic('calculate_der_fps')
                new_hashs = sorted(new_images.keys())
                no_of_new_images = len(new_hashs)
                # new images are shared between cores for calculating
                # fingerprint derivatives
                _mp.make_list_of_sub_images(no_of_new_images, new_hashs,
                                            new_images)
                del new_hashs

                # Temporary files to hold child fingerprint calculations.
                if data_format is 'json':
                    childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                              suffix='.json')
                                  for _ in range(_mp.no_procs)]
                elif data_format is 'db':
                    childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                              suffix='.db')
                                  for _ in range(_mp.no_procs)]
                _ = 0
                while _ < _mp.no_procs:
                    log('  Processor %i calculations stored in file %s.'
                        % (_, childfiles[_].name))
                    _ += 1

                task_args = (fp, snl, childfiles, io, data_format, save_memory,
                             snl.ncursor,)
                _mp.share_fingerprints_task_between_cores(
                    task=_calculate_der_fingerprints,
                    _args=task_args)

                log(' Calculated %i new images.' % no_of_new_images,
                    toc='calculate_der_fps')

                log.tic('read_der_fps')
                log(' Reading calculated fingerprint-derivatives...')
                if save_memory:
                    for f in childfiles:
                        _fdconn = sqlite3.connect(f.name)
                        _fdcursor = _fdconn.cursor()
                        for row in _fdcursor.execute('''SELECT * FROM
                        fingerprint_derivatives'''):
                            self.fdcursor.execute('''INSERT INTO
                            fingerprint_derivatives
                            VALUES (?, ?, ?, ?, ?, ?)''', row)
                    fdconn.commit()
                else:
                    for f in childfiles:
                        _, self.der_fp_data = \
                            io.read(f.name,
                                    'fingerprint_derivatives',
                                    self.der_fp_data,
                                    data_format)
                    io.save(filename, 'fingerprint_derivatives',
                            self.der_fp_data, data_format)
                log(''' ...child fingerprint-derivatives read and saved to
                %s.''' % filename, toc='read_der_fps')

            del new_images
        del images

        log(' ...all fingerprint operations complete.', toc=True)

###############################################################################
###############################################################################
###############################################################################


class CostFxnandDer:

    """
    Cost function and its derivative based on sum of squared deviations in
    energies and forces to be optimized by setting variables.

    :param reg: Regression object.
    :type reg: object
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    :param no_of_images: Number of images.
    :type no_of_images: int
    :param label: Prefix used for all files.
    :type label: str
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param energy_goal: Threshold energy per atom rmse at which simulation is
                        converged.
    :type energy_goal: float
    :param force_goal: Threshold force rmse at which simulation is converged.
    :type force_goal: float
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param _mp: Multiprocess object.
    :type _mp: object
    :param overfitting_constraint: The constant to constraint overfitting.
    :type overfitting_constraint: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function. This controls how tight force-fit is as
                              compared to energy fit. It also depends on force
                              and energy units. Working with eV and Angstrom,
                              0.04 seems to be a reasonable value.
    :type force_coefficient: float
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    """
    ###########################################################################

    def __init__(self, reg, param, no_of_images, label, log, energy_goal,
                 force_goal, train_forces, _mp, overfitting_constraint,
                 force_coefficient, fortran, save_memory, sfp=None, snl=None,):

        self.reg = reg
        self.param = param
        self.no_of_images = no_of_images
        self.label = label
        self.log = log
        self.energy_goal = energy_goal
        self.force_goal = force_goal
        self.train_forces = train_forces
        self._mp = _mp
        self.overfitting_constraint = overfitting_constraint
        self.force_coefficient = force_coefficient
        self.fortran = fortran
        self.save_memory = save_memory
        self.sfp = sfp
        self.snl = snl
        self.steps = 0

        if param.descriptor is not None:  # pure atomic-coordinates scheme
            self.cutoff = param.descriptor.cutoff

        self.energy_convergence = False
        self.force_convergence = False
        if not self.train_forces:
            self.force_convergence = True

        self.energy_coefficient = 1.0

        if fortran:
            # regression data is sent to fortran modules
            self.reg.send_data_to_fortran(param)

        gc.collect()

    ###########################################################################

    def f(self, variables):
        """
        Function to calculate cost function.

        :param variables: Calibrating variables.
        :type variables: list
        """
        calculate_gradient = True
        log = self.log
        self.param.regression._variables = variables
        len_of_variables = len(variables)

        if self.fortran:
            task_args = (self.param, calculate_gradient)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len_of_variables)
        else:
            task_args = (self.reg, self.param, self.sfp, self.snl,
                         self.energy_coefficient, self.force_coefficient,
                         self.train_forces, len_of_variables,
                         calculate_gradient,
                         self.save_memory,)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_python,
                _args=task_args, len_of_variables=len_of_variables)

        square_error = self.energy_coefficient * energy_square_error + \
            self.force_coefficient * force_square_error

        self.cost_function = square_error

        self.energy_per_atom_rmse = \
            np.sqrt(energy_square_error / self.no_of_images)
        self.force_rmse = np.sqrt(force_square_error / self.no_of_images)

        if self.steps == 0:
            if self.train_forces is True:
                head1 = ('%5s  %19s  %9s  %9s  %9s')
                log(head1 % ('', '', '',
                             ' (Energy',
                             ''))
                log(head1 % ('', '', 'Cost',
                             'per Atom)',
                             'Force'))
                log(head1 % ('Step', 'Time', 'Function',
                             'RMSE',
                             'RMSE'))
                log(head1 %
                    ('=' * 5, '=' * 19, '=' * 9, '=' * 9, '=' * 9))
            else:
                head1 = ('%3s  %16s %18s %10s')
                log(head1 % ('Step', 'Time', 'Cost',
                             '    RMS (Energy'))
                head2 = ('%3s  %28s %10s %12s')
                log(head2 % ('', '', 'Function',
                             'per Atom)'))
                head3 = ('%3s  %26s %10s %11s')
                log(head3 % ('', '', '',
                             '   Error'))
                log(head3 %
                    ('=' * 5, '=' * 26, '=' * 9, '=' * 10))

        self.steps += 1

        if self.train_forces is True:
            line = ('%5s' '  %19s' '  %9.3e' '  %9.3e' '  %9.3e')
            log(line % (self.steps - 1,
                        now(),
                        self.cost_function,
                        self.energy_per_atom_rmse,
                        self.force_rmse))
        else:
            line = ('%5s' ' %26s' ' %10.3e' ' %12.3e')
            log(line % (self.steps - 1,
                        now(),
                        self.cost_function,
                        self.energy_per_atom_rmse))

        if self.steps % 100 == 0:
            log('Saving checkpoint data.')
            filename = make_filename(
                self.label,
                'parameters-checkpoint.json')
            save_parameters(filename, self.param)

        if self.energy_per_atom_rmse < self.energy_goal and \
                self.energy_convergence is False:
            log('Energy convergence!')
            self.energy_convergence = True
        elif self.energy_per_atom_rmse > self.energy_goal and \
                self.energy_convergence is True:
            log('Energy unconverged!')
            self.energy_convergence = False

        if self.train_forces:
            if self.force_rmse < self.force_goal and \
                    self.force_convergence is False:
                log('Force convergence!')
                self.force_convergence = True
            elif self.force_rmse > self.force_goal and \
                    self.force_convergence is True:
                log('Force unconverged!')
                self.force_convergence = False

        if (self.energy_convergence and self.force_convergence):
            raise ConvergenceOccurred

        gc.collect()

        return self.cost_function

    ###########################################################################

    def fprime(self, variables):
        """
        Function to calculate derivative of cost function.

        :param variables: Calibrating variables.
        :type variables: list
        """
        if self.steps == 0:

            calculate_gradient = True
            self.param.regression._variables = variables
            len_of_variables = len(variables)

            if self.fortran:
                task_args = (self.param, calculate_gradient)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len_of_variables)
            else:
                task_args = (self.reg, self.param, self.sfp, self.snl,
                             self.energy_coefficient, self.force_coefficient,
                             self.train_forces, len_of_variables,
                             calculate_gradient, self.save_memory,)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_python,
                    _args=task_args, len_of_variables=len_of_variables)

        der_cost_function = self.der_variables_square_error

        der_cost_function = np.array(der_cost_function)

        return der_cost_function

###############################################################################
###############################################################################
###############################################################################


def _calculate_fingerprints(proc_no, hashs, images, fp, childfiles, io,
                            data_format, save_memory,):
    """
    Function to be called on all processes simultaneously for calculating
    fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param childfiles: Temporary files
    :type childfiles: file
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    """
    if save_memory:
        fconn = sqlite3.connect(childfiles[proc_no].name)
        fcursor = fconn.cursor()
        # Create table
        fcursor.execute('''CREATE TABLE IF NOT EXISTS fingerprints
        (image text, atom integer, fp_index integer, value real)''')

    data = {}
    no_of_images = len(hashs)
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        data[hash] = {}
        atoms = images[hash]
        no_of_atoms = len(atoms)
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        #FIXME/ap Weren't nl's calculated previously??
        _nl.update(atoms)
        index = 0
        while index < no_of_atoms:
            symbol = atoms[index].symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            if save_memory:
                len_of_indexfp = len(indexfp)
                _ = 0
                while _ < len_of_indexfp:
                    # Insert a row of data
                    row = (hash, index, _, indexfp[_])
                    fcursor.execute('''INSERT INTO fingerprints VALUES
                    (?, ?, ?, ?)''', row)
                    _ += 1
            else:
                data[hash][index] = indexfp
            index += 1
        count += 1

    if save_memory:
        fconn.commit()
        fconn.close()
    else:
        io.save(childfiles[proc_no].name, 'fingerprints', data, data_format)

    del hashs, images, data

###############################################################################


def _calculate_der_fingerprints(proc_no, hashs, images, fp, snl, childfiles,
                                io, data_format, save_memory, ncursor):
    """
    Function to be called on all processes simultaneously for calculating
    derivatives of fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param childfiles: Temporary files
    :type childfiles: file
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    :param ncursor: Cursor connecting to neighborlists database in the
                    save_memory mode.
    :type ncursor: object
    """
    if save_memory:
        fdconn = sqlite3.connect(childfiles[proc_no].name)
        fdcursor = fdconn.cursor()
        # Create table
        fdcursor.execute('''CREATE TABLE IF NOT EXISTS fingerprint_derivatives
        (image text, atom integer, neighbor_atom integer,
         direction integer, fp_index integer, value real)''')

    data = {}
    no_of_images = len(hashs)
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        data[hash] = {}
        atoms = images[hash]
        no_of_atoms = len(atoms)
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] * no_of_atoms),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        self_index = 0
        while self_index < no_of_atoms:
            if save_memory:
                ncursor.execute('''SELECT * FROM neighborlists WHERE
                image=? AND atom=?''', (hash, self_index))
                rows = ncursor.fetchall()
                n_self_indices = [row[3]
                                  for _ in range(len(rows))
                                  for row in rows if row[2] == _]
                n_self_offsets = [[row[4], row[5], row[6]]
                                  for _ in range(len(rows))
                                  for row in rows if row[2] == _]
            else:
                n_self_indices = snl.nl_data[hash][self_index][0]
                n_self_offsets = snl.nl_data[hash][self_index][1]
            len_of_n_self_indices = len(n_self_indices)
            n_symbols = [atoms[n_index].symbol for n_index in n_self_indices]

            n_count = 0
            while n_count < len_of_n_self_indices:
                n_symbol = n_symbols[n_count]
                n_index = n_self_indices[n_count]
                n_offset = n_self_offsets[n_count]
                # derivative of fingerprints are needed only with respect to
                # coordinates of atoms of type II (within the main cell only)
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    i = 0
                    while i < 3:
                        neighbor_indices, neighbor_offsets = \
                            _nl.get_neighbors(n_index)
                        # for calculating derivatives of fingerprints,
                        # summation runs over neighboring atoms of type I
                        # (either inside or outside the main cell)
                        neighbor_symbols = \
                            [atoms[
                                _index].symbol for _index in neighbor_indices]
                        Rs = [atoms.positions[_index] +
                              np.dot(_offset, atoms.get_cell())
                              for _index, _offset
                              in zip(neighbor_indices, neighbor_offsets)]
                        der_indexfp = fp.get_der_fingerprint(
                            n_index, n_symbol,
                            neighbor_indices,
                            neighbor_symbols,
                            Rs, self_index, i)
                        if save_memory:
                            len_of_der_indexfp = len(der_indexfp)
                            _ = 0
                            while _ < len_of_der_indexfp:
                                # Insert a row of data
                                row = (hash, self_index, n_index, i, _,
                                       der_indexfp[_])
                                fdcursor.execute('''INSERT INTO
                                fingerprint_derivatives
                                VALUES (?, ?, ?, ?, ?, ?)''', row)
                                _ += 1
                        else:
                            data[hash][(n_index, self_index, i)] = der_indexfp
                        i += 1
                n_count += 1
            self_index += 1
        count += 1

    if save_memory:
        fdconn.commit()
        fdconn.close()
    else:
        io.save(childfiles[proc_no].name, 'fingerprint_derivatives',
                data, data_format)

    del hashs, images, data

###############################################################################


def _calculate_cost_function_fortran(param, calculate_gradient, queue):
    """
    Function to be called on all processes simultaneously for calculating cost
    function and its derivative with respect to variables in fortran.

    :param param: ASE dictionary.
    :type param: dict
    :param calculate_gradient: Determines whether or not gradient of the cost
                               function with respect to variables should also
                               be calculated.
    :type calculate_gradient: bool
    :param queue: multiprocessing queue.
    :type queue: object
    """
    variables = param.regression._variables

    (energy_square_error,
     force_square_error,
     der_variables_square_error) = \
        fmodules.share_cost_function_task_between_cores(
        variables=variables,
        len_of_variables=len(variables),
        calculate_gradient=calculate_gradient)

    queue.put([energy_square_error,
               force_square_error,
               der_variables_square_error])

###############################################################################


def _calculate_cost_function_python(hashs, images, reg, param, sfp,
                                    snl, energy_coefficient,
                                    force_coefficient, train_forces,
                                    len_of_variables, calculate_gradient,
                                    save_memory, queue):
    """
    Function to be called on all processes simultaneously for calculating cost
    function and its derivative with respect to variables in python.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param reg: Regression object.
    :type reg: object
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param energy_coefficient: Multiplier of energy per atom RMSE in
                               constructing the cost function.
    :type energy_coefficient: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function.
    :type force_coefficient: float
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param len_of_variables: Number of calibrating variables.
    :type len_of_variables: int
    :param calculate_gradient: Determines whether or not gradient of the cost
                               function with respect to variables should also
                               be calculated.
    :type calculate_gradient: bool
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    :param queue: multiprocessing queue.
    :type queue: object
    """
    der_variables_square_error = np.zeros(len_of_variables)

    energy_square_error = 0.
    force_square_error = 0.

    reg.update_variables(param)

    no_of_images = len(hashs)
    count0 = 0
    while count0 < no_of_images:
        hash = hashs[count0]
        atoms = images[hash]
        no_of_atoms = len(atoms)
        real_energy = atoms.get_potential_energy(apply_constraint=False)
        real_forces = atoms.get_forces(apply_constraint=False)

        reg.reset_energy()
        amp_energy = 0.

        if param.descriptor is None:  # pure atomic-coordinates scheme

            input = (atoms.positions).ravel()
            amp_energy = reg.get_energy(input,)

        else:  # fingerprinting scheme
            index = 0
            while index < no_of_atoms:
                symbol = atoms[index].symbol
                if save_memory:
                    sfp.fcursor.execute('''SELECT * FROM fingerprints WHERE
                    image=? AND atom=?''', (hash, index))
                    rows = sfp.fcursor.fetchall()
                    indexfp = [row[3]
                               for _ in range(len(rows))
                               for row in rows
                               if row[2] == _]
                else:
                    indexfp = sfp.fp_data[hash][index]

                atomic_amp_energy = reg.get_energy(indexfp,
                                                   index, symbol,)
                amp_energy += atomic_amp_energy
                index += 1

        energy_square_error += ((amp_energy - real_energy) ** 2.) / \
            (no_of_atoms ** 2.)

        if calculate_gradient:

            if param.descriptor is None:  # pure atomic-coordinates scheme

                partial_der_variables_square_error = \
                    reg.get_variable_der_of_energy()
                der_variables_square_error += \
                    energy_coefficient * 2. * (amp_energy - real_energy) * \
                    partial_der_variables_square_error / (no_of_atoms ** 2.)

            else:  # fingerprinting scheme

                index = 0
                while index < no_of_atoms:
                    symbol = atoms[index].symbol
                    partial_der_variables_square_error =\
                        reg.get_variable_der_of_energy(index, symbol)
                    der_variables_square_error += \
                        energy_coefficient * 2. * (amp_energy - real_energy) \
                        * partial_der_variables_square_error / \
                        (no_of_atoms ** 2.)
                    index += 1

        if train_forces is True:

            real_forces = atoms.get_forces(apply_constraint=False)
            amp_forces = np.zeros((no_of_atoms, 3))
            self_index = 0
            while self_index < no_of_atoms:
                reg.reset_forces()

                if param.descriptor is None:  # pure atomic-coordinates scheme

                    i = 0
                    while i < 3:
                        _input = [0.] * (3 * no_of_atoms)
                        _input[3 * self_index + i] = 1.
                        force = reg.get_force(i, _input,)
                        amp_forces[self_index][i] = force
                        i += 1

                else:  # fingerprinting scheme
                    if save_memory:
                        snl.ncursor.execute('''SELECT * FROM neighborlists WHERE
                        image=? AND atom=?''', (hash, self_index))
                        rows = snl.ncursor.fetchall()
                        n_self_indices = [row[3]
                                          for _ in range(len(rows))
                                          for row in rows if row[2] == _]
                        n_self_offsets = [[row[4], row[5], row[6]]
                                          for _ in range(len(rows))
                                          for row in rows if row[2] == _]
                    else:
                        n_self_indices = snl.nl_data[hash][self_index][0]
                        n_self_offsets = snl.nl_data[hash][self_index][1]
                    len_of_n_self_indices = len(n_self_indices)
                    n_symbols = [atoms[n_index].symbol
                                 for n_index in n_self_indices]

                    n_count = 0
                    while n_count < len_of_n_self_indices:
                        n_symbol = n_symbols[n_count]
                        n_index = n_self_indices[n_count]
                        n_offset = n_self_offsets[n_count]
                        # for calculating forces, summation runs over neighbor
                        # atoms of type II (within the main cell only)
                        if n_offset[0] == 0 and n_offset[1] == 0 and \
                                n_offset[2] == 0:
                            i = 0
                            while i < 3:
                                if save_memory:
                                    sfp.fdcursor.execute('''SELECT * FROM
                                    fingerprint_derivatives
                                    WHERE image=? AND atom=? AND
                                    neighbor_atom=? AND direction=?''',
                                                         (hash,
                                                          self_index,
                                                          n_index, i))
                                    rows = sfp.fdcursor.fetchall()
                                    der_indexfp = \
                                        [row[5]
                                         for der_fp_index in range(len(rows))
                                         for row in rows
                                         if row[4] == der_fp_index]
                                else:
                                    der_indexfp = \
                                        sfp.der_fp_data[hash][(n_index,
                                                               self_index,
                                                               i)]

                                force = reg.get_force(i, der_indexfp,
                                                      n_index, n_symbol,)

                                amp_forces[self_index][i] += force
                                i += 1
                        n_count += 1

                i = 0
                while i < 3:
                    force_square_error += \
                        ((1.0 / 3.0) * (amp_forces[self_index][i] -
                                        real_forces[self_index][i]) **
                         2.) / no_of_atoms
                    i += 1

                if calculate_gradient:

                    i = 0
                    while i < 3:
                        # pure atomic-coordinates scheme
                        if param.descriptor is None:

                            partial_der_variables_square_error = \
                                reg.get_variable_der_of_forces(self_index, i)
                            der_variables_square_error += \
                                force_coefficient * (2.0 / 3.0) * \
                                (- amp_forces[self_index][i] +
                                 real_forces[self_index][i]) * \
                                partial_der_variables_square_error \
                                / no_of_atoms

                        else:  # fingerprinting scheme

                            n_count = 0
                            while n_count < len_of_n_self_indices:
                                n_symbol = n_symbols[n_count]
                                n_index = n_self_indices[n_count]
                                n_offset = n_self_offsets[n_count]
                                if n_offset[0] == 0 and n_offset[1] == 0 and \
                                        n_offset[2] == 0:

                                    partial_der_variables_square_error = \
                                        reg.get_variable_der_of_forces(
                                            self_index,
                                            i,
                                            n_index,
                                            n_symbol,)
                                    der_variables_square_error += \
                                        force_coefficient * (2.0 / 3.0) * \
                                        (- amp_forces[self_index][i] +
                                         real_forces[self_index][i]) * \
                                        partial_der_variables_square_error \
                                        / no_of_atoms
                                n_count += 1
                        i += 1
                self_index += 1
        count0 += 1

    del hashs, images

    queue.put([energy_square_error,
               force_square_error,
               der_variables_square_error])

###############################################################################


def xcalculate_fingerprints_range(fp, elements, atoms, nl):
    """
    Function to calculate fingerprints range.

    :param fp: Fingerprint object.
    :type fp: object
    :param elements: List of atom symbols.
    :type elements: list of str
    :param atoms: ASE atoms object.
    :type atoms: ASE dict
    :param nl: ASE NeighborList object.
    :type nl: object

    :returns: Range of fingerprints of elements.
    """
    #FIXME/ap This seems like it is re-calculating all the fingerprints
    # in order to get the range. Why not just cycle over the existing
    # fingerprint range?
    fingerprint_values = {}
    for element in elements:
        fingerprint_values[element] = {}
        i = 0
        len_of_fingerprints = len(fp.Gs[element])
        while i < len_of_fingerprints:
            fingerprint_values[element][i] = []
            i += 1

    no_of_atoms = len(atoms)
    index = 0
    while index < no_of_atoms:
        symbol = atoms[index].symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        len_of_fingerprints = len(fp.Gs[symbol])
        _ = 0
        while _ < len_of_fingerprints:
            fingerprint_values[symbol][_].append(indexfp[_])
            _ += 1
        index += 1

    fingerprints_range = {}
    for element in elements:
        len_of_fingerprints = len(fp.Gs[element])
        fingerprints_range[element] = [None] * len_of_fingerprints
        count = 0
        while count < len_of_fingerprints:
            if len(fingerprint_values[element][count]) > 0:
                minimum = min(fingerprint_values[element][count])
                maximum = max(fingerprint_values[element][count])
            else:
                minimum = -1.0
                maximum = -1.0
            fingerprints_range[element][count] = [minimum, maximum]
            count += 1

    return fingerprints_range

###############################################################################


def compare_train_test_fingerprints(fp, atoms, fingerprints_range, nl):
    """
    Function to compare train images with the test image and decide whether
    the prediction is interpolation or extrapolation.

    :param fp: Fingerprint object.
    :type fp: object
    :param atoms: ASE atoms object.
    :type atoms: ASE dict
    :param fingerprints_range: Range of fingerprints of each chemical species.
    :type fingerprints_range: dict
    :param nl: ASE NeighborList object.
    :type nl: object

    :returns: Zero for interpolation, and one for extrapolation.
    """
    compare_train_test_fingerprints = 0

    no_of_atoms = len(atoms)
    index = 0
    while index < no_of_atoms:
        symbol = atoms[index].symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        len_of_fingerprints = len(indexfp)
        i = 0
        while i < len_of_fingerprints:
            if indexfp[i] < fingerprints_range[symbol][i][0] or \
                    indexfp[i] > fingerprints_range[symbol][i][1]:
                compare_train_test_fingerprints = 1
                break
            i += 1
        index += 1
    return compare_train_test_fingerprints

###############################################################################


def interpolate_images(images, load, fortran=True):
    """
    Function to remove extrapolation images from the "images" set based on
    load data.

    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This can also be the path to an
                   ASE trajectory (.traj) or database (.db) file.
    :type images: list or str
    :param load: Path for loading an existing parameters of Amp calculator.
    :type load: str
    :param fortran: If True, will use fortran modules, if False, will not.
    :type fortran: bool

    :returns: Two dictionary of all images, and interpolated images,
              respectively.
    """
    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = aseio.Trajectory(images, 'r')
        elif extension == '.db':
            images = aseio.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    no_of_images = len(images)
    count = 0
    while count < no_of_images:
        image = images[count]
        hash = hash_image(image)
        dict_images[hash] = image
        count += 1
    images = dict_images.copy()
    del dict_images

    amp = Amp(load=load, fortran=fortran)
    param = amp.parameters
    fp = param.descriptor
    # FIXME: fingerprints_range has been removed from param
    fingerprints_range = param.fingerprints_range
    # FIXME: this function should be extended to no fingerprints scheme.
    if fp is not None:  # fingerprinting scheme
        cutoff = fp.cutoff

    # Dictionary of interpolated images set is initialized
    interpolated_images = {}
    hashs = images.keys()
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        atoms = images[hash]
        fp.atoms = atoms
        _nl = NeighborList(cutoffs=([cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        fp._nl = _nl
        compare_train_test_fingerprints = 0
        no_of_atoms = len(atoms)
        index = 0
        while index < no_of_atoms:
            symbol = atoms[index].symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            len_of_fingerprints = len(indexfp)
            i = 0
            while i < len_of_fingerprints:
                if indexfp[i] < fingerprints_range[symbol][i][0] or \
                        indexfp[i] > fingerprints_range[symbol][i][1]:
                    compare_train_test_fingerprints = 1
                    break
                i += 1
            index += 1
        if compare_train_test_fingerprints == 0:
            interpolated_images[hash] = image
        count += 1

    return images, interpolated_images

###############################################################################


def send_data_to_fortran(sfp, elements, train_forces,
                         energy_coefficient, force_coefficient, param):
    """
    Function to send images data to fortran code. Is used just once.

    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param elements: List of atom symbols.
    :type elements: list of str
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param energy_coefficient: Multiplier of energy per atom RMSE in
                               constructing the cost function.
    :type energy_coefficient: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function.
    :type force_coefficient: float
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    """
    if param.descriptor is None:
        fingerprinting = False
    else:
        fingerprinting = True

    if fingerprinting:
        no_of_elements = len(elements)
        elements_numbers = [atomic_numbers[elm] for elm in elements]
        min_fingerprints = \
            [[sfp.fingerprints_range[elm][_][0]
              for _ in range(len(sfp.fingerprints_range[elm]))]
             for elm in elements]
        max_fingerprints = [[sfp.fingerprints_range[elm][_][1]
                             for _
                             in range(len(sfp.fingerprints_range[elm]))]
                            for elm in elements]
        len_fingerprints_of_elements = [len(sfp.Gs[elm]) for elm in elements]
    else:
        no_of_atoms_of_image = param.no_of_atoms

    fmodules.images_props.energy_coefficient = energy_coefficient
    fmodules.images_props.force_coefficient = force_coefficient
    fmodules.images_props.train_forces = train_forces
    fmodules.images_props.fingerprinting = fingerprinting

    if fingerprinting:
        fmodules.images_props.no_of_elements = no_of_elements
        fmodules.images_props.elements_numbers = elements_numbers
        fmodules.fingerprint_props.min_fingerprints = min_fingerprints
        fmodules.fingerprint_props.max_fingerprints = max_fingerprints
        fmodules.fingerprint_props.len_fingerprints_of_elements = \
            len_fingerprints_of_elements
    else:
        fmodules.images_props.no_of_atoms_of_image = no_of_atoms_of_image

##############################################################################


def ravel_fingerprints_of_images(hashs, images, sfp, save_memory,):
    """
    Reshape fingerprints of all images into a matrix.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    """
    if save_memory:
        raveled_fingerprints = []
        for hash in hashs:
            for index in range(len(images[hash])):
                sfp.fcursor.execute('''SELECT * FROM fingerprints WHERE
                image=? AND atom=?''', (hash, index))
                rows = sfp.fcursor.fetchall()
                indexfp = [row[3]
                           for _ in range(len(rows))
                           for row in rows
                           if row[2] == _]
                raveled_fingerprints.append(indexfp)
    else:
        raveled_fingerprints = [sfp.fp_data[hash][index]
                                for hash in hashs
                                for index in range(len(images[hash]))]

    del hashs, images

    return raveled_fingerprints

###############################################################################


def ravel_neighborlists_and_der_fingerprints_of_images(hashs, images, sfp,
                                                       snl, save_memory,):
    """
    Reshape neighborlists and derivatives of fingerprints of all images into a
    matrix.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param save_memory: If True, memory efficient mode will be used.
    :type save_memory: bool
    """
    # Only neighboring atoms of type II (within the main cell) needs to be sent
    # to fortran for force training
    list_of_no_of_neighbors = []
    raveled_neighborlists = []
    raveled_der_fingerprints = []
    no_of_images = len(hashs)
    _ = 0
    while _ < no_of_images:
        hash = hashs[_]
        atoms = images[hash]
        no_of_atoms = len(atoms)
        self_index = 0
        while self_index < no_of_atoms:
            if save_memory:
                snl.ncursor.execute('''SELECT * FROM neighborlists WHERE
                image=? AND atom=?''', (hash, self_index,))
                rows = snl.ncursor.fetchall()
                n_self_indices = [row[3]
                                  for __ in range(len(rows))
                                  for row in rows if row[2] == __]
                n_self_offsets = [[row[4], row[5], row[6]]
                                  for __ in range(len(rows))
                                  for row in rows if row[2] == __]
            else:
                n_self_indices = snl.nl_data[hash][self_index][0]
                n_self_offsets = snl.nl_data[hash][self_index][1]
            len_of_n_self_offsets = len(n_self_offsets)

            count = 0
            n_count = 0
            while n_count < len_of_n_self_offsets:
                n_index = n_self_indices[n_count]
                n_offset = n_self_offsets[n_count]
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    raveled_neighborlists.append(n_index)
                    for i in range(3):
                        if save_memory:
                            sfp.fdcursor.execute('''SELECT * FROM
                            fingerprint_derivatives WHERE image=? AND atom=?
                            AND neighbor_atom=? AND direction=?''',
                                                 (hash,
                                                  self_index,
                                                  n_index, i))
                            rows = sfp.fdcursor.fetchall()
                            der_indexfp = [row[5]
                                           for __ in range(len(rows))
                                           for row in rows
                                           if row[4] == __]
                        else:
                            der_indexfp = sfp.der_fp_data[hash][(n_index,
                                                                 self_index,
                                                                 i)]
                        raveled_der_fingerprints.append(der_indexfp)
                    count += 1
                del n_offset
                n_count += 1
            list_of_no_of_neighbors.append(count)
            self_index += 1
        del hash
        _ += 1

    del hashs, images

    return (list_of_no_of_neighbors,
            raveled_neighborlists,
            raveled_der_fingerprints)

###############################################################################


def now():
    """
    :returns: String of current time.
    """
    return datetime.now().isoformat().split('.')[0]

###############################################################################

def importhelper(importname):
    """Manually compiled list of available modules. This is to prevent the
    execution of arbitrary (potentially malicious) code.
    """
    if importname == '.descriptor.behler.Behler':
        from .descriptor.behler import Behler as Module
    elif importname == '.model.neuralnetwork.NeuralNetwork':
        from .model.neuralnetwork import NeuralNetwork as Module
    else:
        raise NotImplementedError(
            'Attempt to import the module %s. Was this intended? '
            'If so, trying manually importing this module and '
            'feeding it to Amp.load. To avoid this error, this '
            'module can be added to amp.importhelper.' %
            importname)

    return Module
