import os
import sys
import shutil
import numpy as np
import tempfile
import platform
from getpass import getuser
from socket import gethostname
import subprocess
import warnings

import ase
from ase.calculators.calculator import Calculator, Parameters
try:
    from ase import __version__ as aseversion
except ImportError:
    # We're on ASE 3.9 or older
    from ase.version import version as aseversion

from .utilities import (make_filename, hash_images, Logger, string2dict,
                        logo, now, assign_cores, TrainingConvergenceError,
                        check_images)

try:
    from amp import fmodules
except ImportError:
    warnings.warn('Did not find fortran modules.')
else:
    fmodules_version = 12
    wrong_version = fmodules.check_version(version=fmodules_version)
    if wrong_version:
        raise RuntimeError('fortran modules are not updated. Recompile '
                           'with f2py as described in the README. '
                           'Correct version is %i.' % fmodules_version)

version_file = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                            'VERSION')
_ampversion = open(version_file).read().strip()


class Amp(Calculator, object):

    """Atomistic Machine-Learning Potential (Amp) ASE calculator

    Parameters
    ----------
    descriptor : object
        Class representing local atomic environment.
    model : object
        Class representing the regression model. Can be only NeuralNetwork for
        now. Input arguments for NeuralNetwork are hiddenlayers, activation,
        weights, and scalings; for more information see docstring for the class
        NeuralNetwork.
    label : str
        Default prefix/location used for all files. Setting to None disables
        logging.
    dblabel : str
        Optional separate prefix/location for database files, including
        fingerprints, fingerprint derivatives, and neighborlists. This file
        location can be shared between calculator instances to avoid
        re-calculating redundant information. If not supplied, just uses the
        value from label.
    cores : int
        Can specify cores to use for parallel training; if None, will determine
        from environment
    envcommand : string
        For parallel processing across nodes, a command can be supplied
        here to load the appropriate environment before starting workers.
    logging : boolean
        Option to turn off logging; e.g., to speed up force calls.
    atoms : object
        ASE atoms objects with positions, symbols, energy, and forces in ASE
        format.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, model, label='amp', dblabel=None,
                 cores=None, envcommand=None, logging=True, atoms=None):

        self.logging = logging
        Calculator.__init__(self, label=label, atoms=atoms)
        # Note self._log is set and self._printheader is called by above
        # call when it runs self.set_label.

        self._parallel = {'envcommand': envcommand}

        # Note the following are properties: these are setter functions.
        self.descriptor = descriptor
        self.model = model
        self.cores = cores  # Note this calls 'assign_cores'.

        self.dblabel = label if dblabel is None else dblabel

    @property
    def cores(self):
        """
        Get or set the cores for the parallel environment.

        Parameters
        ----------
        cores : int or dictionary
            Parallel configuration. If cores is an integer, parallelizes over
            this many processes on machine localhost. cores can also be
            a dictionary of the type {'node324': 16, 'node325': 16}. If not
            specified, tries to determine from environment, using
            amp.utilities.assign_cores.
        """
        return self._parallel['cores']

    @cores.setter
    def cores(self, cores):
        self._parallel['cores'] = assign_cores(cores, log=self._log)

    @property
    def descriptor(self):
        """Get or set the atomic descriptor.

        Parameters
        ----------
        descriptor : object
            Class instance representing the local atomic environment.
        """
        return self._descriptor

    @descriptor.setter
    def descriptor(self, descriptor):
        descriptor.parent = self  # gives the descriptor object a reference to
        # the main Amp instance. Then descriptor can pull parameters directly
        # from Amp without needing them to be passed in each method call.
        self._descriptor = descriptor
        self.reset()  # Clears any calculation results.

    @property
    def model(self):
        """Get or set the machine-learning model.

        Parameters
        ----------
        model : object
            Class instance representing the regression model.
        """
        return self._model

    @model.setter
    def model(self, model):
        model.parent = self  # gives the model object a reference to the main
        # Amp instance. Then model can pull parameters directly from Amp
        # without needing them to be passed in each method call.
        self._model = model
        self.reset()  # Clears any calculation results.

    @classmethod
    def load(Cls, file, Descriptor=None, Model=None, **kwargs):
        """Attempts to load calculators and return a new instance of Amp.

        Only a filename or file-like object is required, in typical cases.

        If using a home-rolled descriptor or model, also supply uninstantiated
        classes to those models, as in Model=MyModel.
        (Not as Model=MyModel()!)

        Any additional keyword arguments (such as label or dblabel) can be
        fed through to Amp.

        Parameters
        ----------
        file : str
            Name of the file to load data from.
        Descriptor : object
            Class representing local atomic environment.
        Model : object
            Class representing the regression model.
        """
        if hasattr(file, 'read'):
            text = file.read()
        else:
            with open(file) as f:
                text = f.read()

        # Unpack parameter dictionaries.
        p = string2dict(text)
        for key in ['descriptor', 'model']:
            p[key] = string2dict(p[key])

        # If modules are not specified, find them.
        if Descriptor is None:
            Descriptor = importhelper(p['descriptor'].pop('importname'))
        if Model is None:
            Model = importhelper(p['model'].pop('importname'))
        # Key 'importname' and the value removed so that it is not splatted
        # into the keyword arguments used to instantiate in the next line.

        # Instantiate the descriptor and model.
        descriptor = Descriptor(**p['descriptor'])
        # ** sends all the key-value pairs at once.
        model = Model(**p['model'])

        # Instantiate Amp.
        calc = Cls(descriptor=descriptor, model=model, **kwargs)
        calc._log('Loaded file: %s' % file)
        return calc

    def set(self, **kwargs):
        """Function to set parameters.

        For now, this doesn't do anything as all parameters are within the
        model and descriptor.
        """
        changed_parameters = Calculator.set(self, **kwargs)
        if len(changed_parameters) > 0:
            self.reset()

    def set_label(self, label):
        """Sets label, ensuring that any needed directories are made.

        Parameters
        ----------
        label : str
            Default prefix/location used for all files.
        """
        Calculator.set_label(self, label)

        # Create directories for output structure if needed.
        # Note ASE doesn't do this for us.
        if self.label:
            if (self.directory != os.curdir and
                    not os.path.isdir(self.directory)):
                os.makedirs(self.directory)

        if self.logging is True:
            self._log = Logger(make_filename(self.label, '-log.txt'))
        else:
            self._log = Logger(None)

        self._printheader(self._log)

    def calculate(self, atoms, properties, system_changes):
        """Calculation of the energy of system and forces of all atoms."""
        # The inherited method below just sets the atoms object,
        # if specified, to self.atoms.
        Calculator.calculate(self, atoms, properties, system_changes)

        log = self._log
        log('Calculation requested.')

        images = hash_images([self.atoms])
        key = list(images.keys())[0]

        if properties == ['energy']:
            log('Calculating potential energy...', tic='pot-energy')
            self.descriptor.calculate_fingerprints(images=images,
                                                   log=log,
                                                   calculate_derivatives=False)
            if self.model.__class__.__name__ != 'KernelRidge':
                energy = self.model.calculate_energy(
                        self.descriptor.fingerprints[key]
                        )

            else:  # KRR needs training images.
                fingerprints = self.descriptor.fingerprints

                log('Loading the training set')

                if isinstance(self.model.trainingimages, str):
                    trainingimages = hash_images(
                            ase.io.Trajectory(self.model.trainingimages)
                            )
                else:
                    trainingimages = hash_images(self.model.trainingimages)

                self.descriptor.calculate_fingerprints(
                        images=trainingimages,
                        log=log,
                        calculate_derivatives=False
                        )
                fp_trainingimages = self.descriptor.fingerprints

                energy = self.model.calculate_energy(
                        fingerprints,
                        hash=key,
                        trainingimages=trainingimages,
                        fp_trainingimages=fp_trainingimages
                        )
            self.results['energy'] = energy
            log('...potential energy calculated.', toc='pot-energy')

        if properties == ['forces']:
            if self.model.__class__.__name__ != 'KernelRidge':
                log('Calculating forces...', tic='forces')
                self.descriptor.calculate_fingerprints(
                        images=images,
                        log=log,
                        calculate_derivatives=True
                        )
                forces = \
                    self.model.calculate_forces(
                        self.descriptor.fingerprints[key],
                        self.descriptor.fingerprintprimes[key])
                self.results['forces'] = forces
                log('...forces calculated.', toc='forces')

            else:
                log('Calculating forces...', tic='forces')
                self.descriptor.calculate_fingerprints(
                        images=images,
                        log=log,
                        calculate_derivatives=True
                        )
                log('Loading the training set')
                if isinstance(self.model.trainingimages, str):
                    trainingimages = hash_images(
                            ase.io.Trajectory(self.model.trainingimages)
                            )
                else:
                    trainingimages = hash_images(self.model.trainingimages)
                self.descriptor.calculate_fingerprints(
                        images=trainingimages,
                        log=log,
                        calculate_derivatives=True
                        )
                t_descriptor = self.descriptor
                forces = \
                    self.model.calculate_forces(
                        self.descriptor.fingerprints[key],
                        self.descriptor.fingerprintprimes[key],
                        hash=key,
                        trainingimages=trainingimages,
                        t_descriptor=t_descriptor,
                        )
                self.results['forces'] = forces
                log('...forces calculated.', toc='forces')

    def train(self,
              images,
              overwrite=False,
              ):
        """Fits the model to the training images.

        Parameters
        ----------
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the training set of data. This can
            also be the path to an ASE trajectory (.traj) or database (.db)
            file. Energies can be obtained from any reference, e.g. DFT
            calculations.
        overwrite : bool
            If an output file with the same name exists, overwrite it.
        """

        log = self._log
        log('\nAmp training started. ' + now() + '\n')
        log('Descriptor: %s\n  (%s)' % (self.descriptor.__class__.__name__,
                                        self.descriptor))
        log('Model: %s\n  (%s)' % (self.model.__class__.__name__, self.model))

        images = hash_images(images, log=log)

        log('\nDescriptor\n==========')
        train_forces = self.model.forcetraining  # True / False
        check_images(images, forces=train_forces)
        self.descriptor.calculate_fingerprints(
                images=images,
                parallel=self._parallel,
                log=log,
                calculate_derivatives=train_forces)

        log('\nModel fitting\n=============')
        result = self.model.fit(trainingimages=images,
                                descriptor=self.descriptor,
                                log=log,
                                parallel=self._parallel)

        if result is True:
            log('Amp successfully trained. Saving current parameters.')
            filename = self.label + '.amp'
            self.reset()  # Clears any calculation results.
        else:
            log('Amp not trained successfully. Saving current parameters.')
            filename = make_filename(self.label, '-untrained-parameters.amp')
        filename = self.save(filename, overwrite)
        log('Parameters saved in file "%s".' % filename)
        log("This file can be opened with `calc = Amp.load('%s')`" %
            filename)
        if result is False:
            raise TrainingConvergenceError('Amp did not converge upon '
                                           'training. See log file for'
                                           ' more information.')

    def train_return_loss(self,
                          images,
                          ):

        log = self._log
        log('\nAmp training started. ' + now() + '\n')
        log('Descriptor: %s\n  (%s)' % (self.descriptor.__class__.__name__,
                                        self.descriptor))
        log('Model: %s\n  (%s)' % (self.model.__class__.__name__, self.model))

        images = hash_images(images, log=log)

        log('\nDescriptor\n==========')
        train_forces = self.model.forcetraining  # True / False
        check_images(images, forces=train_forces)
        self.descriptor.calculate_fingerprints(
                images=images,
                parallel=self._parallel,
                log=log,
                calculate_derivatives=train_forces)

        log('\nModel fitting\n=============')
        result = self.model.fit(trainingimages=images,
                                descriptor=self.descriptor,
                                log=log,
                                parallel=self._parallel)
        loss, energy_rmse, force_rmse = self.model.lossfunction.loss, \
                                        self.model.lossfunction.energy_loss, \
                                        self.model.lossfunction.force_loss
        self.reset()  # Clears any calculation results.

        return loss, energy_rmse, force_rmse


    def save(self, filename, overwrite=False):
        """Saves the calculator in a way that it can be re-opened with
        load.

        Parameters
        ----------
        filename : str
            File object or path to the file to write to.
        overwrite : bool
            If an output file with the same name exists, overwrite it.
        """
        if os.path.exists(filename):
            if overwrite is False:
                oldfilename = filename
                filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                       suffix='.amp').name
                self._log('File "%s" exists. Instead saving to "%s".' %
                          (oldfilename, filename))
            else:
                oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                          delete=False,
                                                          suffix='.amp').name

                self._log('Overwriting file: "%s". Moving original to "%s".'
                          % (filename, oldfilename))
                shutil.move(filename, oldfilename)
        descriptor = self.descriptor.tostring()
        model = self.model.tostring()
        p = Parameters({'descriptor': descriptor,
                        'model': model})
        p.write(filename)
        return filename

    def _printheader(self, log):
        """Prints header to log file; inspired by that in GPAW.
        """
        log(logo)
        log('Amp: Atomistic Machine-learning Package')
        log('Developed by Andrew Peterson, Alireza Khorshidi, and others,')
        log('Brown University.')
        log('PI Website: http://brown.edu/go/catalyst')
        log('Official repository: http://bitbucket.org/andrewpeterson/amp')
        log('Official documentation: http://amp.readthedocs.io/')
        log('Citation:')
        log('  Alireza Khorshidi & Andrew A. Peterson,')
        log('  Computer Physics Communications 207: 310-324 (2016).')
        log('  http://doi.org/10.1016/j.cpc.2016.05.010')
        log('=' * 70)
        log('User: %s' % getuser())
        log('Hostname: %s' % gethostname())
        log('Date: %s' % now(with_utc=True))
        uname = platform.uname()
        log('Architecture: %s' % uname[4])
        log('PID: %s' % os.getpid())
        log('Amp version: %s' % _ampversion)
        ampdirectory = os.path.dirname(os.path.abspath(__file__))
        log('Amp directory: %s' % ampdirectory)
        commithash, commitdate = get_git_commit(ampdirectory)
        log(' Last commit: {:s}'.format(commithash))
        log(' Last commit date: {:s}'.format(commitdate))
        log('Python: v{0}.{1}.{2}: %s'.format(*sys.version_info[:3]) %
            sys.executable)
        log('ASE v%s: %s' % (aseversion, os.path.dirname(ase.__file__)))
        log('NumPy v%s: %s' %
            (np.version.version, os.path.dirname(np.__file__)))
        # SciPy is not a strict dependency.
        try:
            import scipy
            log('SciPy v%s: %s' %
                (scipy.version.version, os.path.dirname(scipy.__file__)))
        except ImportError:
            log('SciPy: not available')
        # ZMQ and pxssh are only necessary for parallel calculations.
        try:
            import zmq
            log('ZMQ/PyZMQ v%s/v%s: %s' %
                (zmq.zmq_version(), zmq.pyzmq_version(),
                 os.path.dirname(zmq.__file__)))
        except ImportError:
            log('ZMQ: not available')
        try:
            import pxssh
            log('pxssh: %s' % os.path.dirname(pxssh.__file__))
        except ImportError:
            log('pxssh: Not available from pxssh.')
            try:
                from pexpect import pxssh
            except ImportError:
                log('pxssh: Not available from pexpect.')
            else:
                import pexpect
                log('pxssh (via pexpect v%s): %s' %
                    (pexpect.__version__, pxssh.__file__))
        log('=' * 70)


def importhelper(importname):
    """Manually compiled list of available modules.

    This is to prevent the execution of arbitrary (potentially malicious) code.

    However, since there is an `eval` statement in string2dict maybe this
    is silly.
    """
    if importname == '.descriptor.gaussian.Gaussian':
        from .descriptor.gaussian import Gaussian as Imported
    elif importname == '.descriptor.zernike.Zernike':
        from .descriptor.zernike import Zernike as Imported
    elif importname == '.descriptor.bispectrum.Bispectrum':
        from .descriptor.bispectrum import Bispectrum as Imported
    elif importname == '.model.neuralnetwork.NeuralNetwork':
        from .model.neuralnetwork import NeuralNetwork as Imported
    elif importname == '.model.neuralnetwork.tflow':
        from .model.tflow import NeuralNetwork as Imported
    elif importname == '.model.kernelridge.KernelRidge':
        from .model.kernelridge import KernelRidge as Imported
    elif importname == '.model.LossFunction':
        from .model import LossFunction as Imported
    else:
        raise NotImplementedError(
            'Attempt to import the module %s. Was this intended? '
            'If so, trying manually importing this module and '
            'feeding it to Amp.load. To avoid this error, this '
            'module can be added to amp.importhelper.' %
            importname)

    return Imported


def get_git_commit(ampdirectory):
    """Attempts to get the last git commit from the amp directory.
    """
    pwd = os.getcwd()
    os.chdir(ampdirectory)
    try:
        with open(os.devnull, 'w') as devnull:
            output = subprocess.check_output(['git', 'log', '-1',
                                              '--pretty=%H\t%ci'],
                                             stderr=devnull)
    except:
        output = b'unknown hash\tunknown date'
    output = output.decode('utf-8')
    output = output.strip()
    commithash, commitdate = output.split('\t')
    os.chdir(pwd)
    return commithash, commitdate
