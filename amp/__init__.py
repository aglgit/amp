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
    fmodules_version = 9
    wrong_version = fmodules.check_version(version=fmodules_version)
    if wrong_version:
        raise RuntimeError('fortran modules are not updated. Recompile '
                           'with f2py as described in the README. '
                           'Correct version is %i.' % fmodules_version)

_ampversion = '(development)'


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
        Default prefix/location used for all files.
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
        """
        Get or set the atomic descriptor.

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
        """
        Get or set the machine-learning model.

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
        classes to those models, as in Model=MyModel.  (Not as
        Model=MyModel()!)

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
        """Calculation of the energy of system and forces of all atoms.

        """
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
            energy = self.model.calculate_energy(
                self.descriptor.fingerprints[key])
            self.results['energy'] = energy
            log('...potential energy calculated.', toc='pot-energy')

        if properties == ['forces']:
            log('Calculating forces...', tic='forces')
            self.descriptor.calculate_fingerprints(images=images,
                                                   log=log,
                                                   calculate_derivatives=True)
            forces = \
                self.model.calculate_forces(
                    self.descriptor.fingerprints[key],
                    self.descriptor.fingerprintprimes[key])
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

    def save_to_prophet(self, filename='potential_', overwrite=False,
                        units="metal"):
        """Saves the calculator in a way that it can be used with PROPhet.

        Parameters
        ----------
        filename : str
            File object or path to the file to write to.
        overwrite : bool
            If an output file with the same name exists, overwrite it.
        units : str
            LAMMPS units style to be used with the outfile file.
        """

        from ase.calculators.lammpslib import unit_convert

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

        desc_pars = self.descriptor.parameters
        model_pars = self.model.parameters
        if ((desc_pars['mode']!='atom-centered') or
            (model_pars['mode']!='atom-centered')):
            raise NotImplementedError(
                'PROPhet requires atom-centered symmetry functions.')
        if desc_pars['cutoff']['name']!='Cosine':
            raise NotImplementedError(
                'PROPhet requires cosine cutoff functions.')
        if model_pars['activation']!='tanh':
            raise NotImplementedError(
                'PROPhet requires tanh activation functions.')
        els = desc_pars['elements']
        n_els = len(els)
        length_G2 = int(n_els)
        length_G4 = int(n_els*(n_els+1)/2)
        cutoff = (desc_pars['cutoff']['kwargs']['Rc'] /
                  unit_convert('distance', units))
        # Get correct order of elements listed in the Amp object
        el = desc_pars['elements'][0]
        n_G2 = sum(1 for k in desc_pars['Gs'][el] if k['type']=='G2')
        n_G4 = sum(1 for k in desc_pars['Gs'][el] if k['type']=='G4')
        els_ordered = []
        if n_G2>0:
            for Gs in range(n_els):
                els_ordered.append(desc_pars['Gs'][el][Gs]['element'])
        elif n_G4>0:
            for Gs in range(n_els):
                els_ordered.append(desc_pars['Gs'][el][Gs]['elements'][0])
        else:
            raise RuntimeError('There must be at least one G2 or G4 symmetry '
                               'function.')
        # Write each element's PROPhet input file
        for el in desc_pars['elements']:
            f = open(filename+el, 'w')
            #write header
            f.write('nn\n')
            f.write('structure\n')
            #write elements
            f.write(el + ':  ')
            for el_i in els_ordered:
                f.write(el_i+' ')
            f.write('\n')
            n_G2_el = sum(1 for k in desc_pars['Gs'][el] if k['type']=='G2')
            n_G4_el = sum(1 for k in desc_pars['Gs'][el] if k['type']=='G4')
            if ((n_G2_el!=n_G2) or (n_G4_el!=n_G4)):
                raise NotImplementedError(
                    'PROPhet requires each element to have the same number of '
                    'symmetry functions.')
            f.write(str(int(n_G2/length_G2+n_G4/length_G4))+'\n')
            #write G2s
            for Gs in range(0, n_G2, length_G2):
                eta = desc_pars['Gs'][el][Gs]['eta']
                if (eta > 10):
                    warnings.warn(
                        'Conversion from Amp to PROPhet leads to energies and '
                        'forces being calculated correctly to within machine '
                        'precision. With the chosen eta of ' +str(eta) + ' '
                        'being greater than 10, it is possible that the '
                        'results of the two codes will not be equal, so the '
                        'neural net should not be used with both codes.'
                        'Please lower the eta values.')
                for i in range(length_G2):
                    eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                    if eta!=eta_2:
                        raise NotImplementedError(
                            'PROPhet requires each G2 function to have the '
                            'same eta value for all element pairs.')
                f.write('G2 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                        ' 0\n')
            #write G4s (G3s in PROPhet)
            for Gs in range(n_G2, n_G2+n_G4, length_G4):
                eta = desc_pars['Gs'][el][Gs]['eta']
                if (eta > 10):
                    warnings.warn(
                        'Conversion from Amp to PROPhet leads to energies and '
                        'forces being calculated correctly to within machine '
                        'precision. With the chosen eta of ' +str(eta) + ' '
                        'being greater than 10, it is possible that the '
                        'results of the two codes will not be equal, so the '
                        'neural net should not be used with both codes.'
                        'Please lower the eta values.')
                gamma = desc_pars['Gs'][el][Gs]['gamma']
                zeta = desc_pars['Gs'][el][Gs]['zeta']
                for i in range(length_G4):
                    eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                    gamma_2 = desc_pars['Gs'][el][Gs+i]['gamma']
                    zeta_2 = desc_pars['Gs'][el][Gs+i]['zeta']
                    if (eta!=eta_2) or (gamma!=gamma_2) or (zeta!=zeta_2):
                        raise NotImplementedError(
                            'PROPhet requires each G4 function to have the '
                            'same eta, gamma, and zeta values for all '
                            'element pairs.')
                f.write('G3 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                        ' ' + str(int(zeta)) + ' ' + str(int(gamma)) + '\n')
            # write input means for G2
            for i in range(n_els):
                for Gs in range(0, n_G2, length_G2):
                    # for debugging, to see the order of the PROPhet file
                    #if el==desc_pars['elements'][0]:
                    #    print(desc_pars['Gs'][el][Gs+i])
                    mean = (model_pars['fprange'][el][Gs+i][1] +
                            model_pars['fprange'][el][Gs+i][0]) / 2.
                    f.write(str(mean )+ ' ')
            # write input means for G4
            for i in range(n_els):
                for j in range(n_els-i):
                    for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                        # for debugging, to see the order of the PROPhet file
                        #if el==desc_pars['elements'][0]:
                        #    print(desc_pars['Gs'][el][Gs+j+n_els*i+int((i-i**2)/2)])
                        mean = (model_pars['fprange'][el][Gs+j+n_els*i+int((i-i**2)/2)][1] +
                                model_pars['fprange'][el][Gs+j+n_els*i+int((i-i**2)/2)][0])
                        # NB the G4 mean is doubled to correct for PROPhet
                        # counting each neighbor pair twice as much as Amp
                        f.write(str(mean )+ ' ')
            f.write('\n')
            # write input variances for G2
            for i in range(n_els):
                for Gs in range(0, n_G2, length_G2):
                    variance = (model_pars['fprange'][el][Gs+i][1] -
                                model_pars['fprange'][el][Gs+i][0]) / 2.
                    f.write(str(variance )+ ' ')
            # write input variances for G4
            for i in range(n_els):
                for j in range(n_els-i):
                    for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                        variance = (model_pars['fprange'][el][Gs+j+n_els*i+int((i-i**2)/2)][1] -
                                    model_pars['fprange'][el][Gs+j+n_els*i+int((i-i**2)/2)][0])
                        # NB the G4 variance is doubled to correct for PROPhet
                        # counting each neighbor pair twice as much as Amp
                        f.write(str(variance )+ ' ')
            f.write('\n')
            f.write('energy\n')
            # write output mean
            f.write('0\n')
            # write output variance
            f.write('1\n')
            curr_node = 0
            # write NN layer architecture
            for nodes in model_pars['hiddenlayers'][el]:
                f.write(str(nodes)+' ')
            f.write('1\n')
            # write first hidden layer of the NN for the symmetry functions
            layer = 0
            f.write('[[ layer ' + str(layer) + ' ]]\n')
            for node in range(model_pars['hiddenlayers'][el][layer]):
                # write each node of the layer
                f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
                f.write('   ')
                # G2
                for i in range(n_els):
                    for Gs in range(0, n_G2, length_G2):
                        f.write(str(model_pars['weights'][el][layer+1][Gs+i][node]))
                        f.write('     ')
                # G4
                for i in range(n_els):
                    for j in range(n_els-i):
                        for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                            f.write(str(model_pars['weights'][el][layer+1][Gs+j+n_els*i+int((i-i**2)/2)][node]))
                            f.write('     ')
                f.write('\n')
                f.write('   ')
                f.write(str(model_pars['weights'][el][layer+1][-1][node]))
                f.write('\n')
                curr_node += 1
            # write remaining hidden layers of the NN
            for layer in range(1, len(model_pars['hiddenlayers'][el])):
                f.write('[[ layer ' + str(layer) + ' ]]\n')
                for node in range(model_pars['hiddenlayers'][el][layer]):
                    f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
                    f.write('   ')
                    for i in range(len(model_pars['weights'][el][layer+1])-1):
                        f.write(str(model_pars['weights'][el][layer+1][i][node]))
                        f.write('     ')
                    f.write('\n')
                    f.write('   ')
                    f.write(str(model_pars['weights'][el][layer+1][-1][node]))
                    f.write('\n')
                    curr_node += 1
            # write output layer of the NN, consisting of an activated node
            f.write('[[ layer ' + str(layer+1) + ' ]]\n')
            f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
            f.write('   ')
            for i in range(len(model_pars['weights'][el][layer+2])-1):
                f.write(str(model_pars['weights'][el][layer+2][i][0]))
                f.write('     ')
            f.write('\n')
            f.write('   ')
            f.write(str(model_pars['weights'][el][layer+2][-1][0]))
            f.write('\n')
            curr_node += 1
            # write output layer of the NN, consisting of a linear node,
            # representing Amp's scaling
            f.write('[[ layer ' + str(layer+2) + ' ]]\n')
            f.write('  [ node ' + str(curr_node) + ' ]  linear\n')
            f.write('   ')
            f.write(str(model_pars['scalings'][el]['slope'] /
                        unit_convert('energy', units)))
            f.write('\n')
            f.write('   ')
            f.write(str(model_pars['scalings'][el]['intercept'] /
                        unit_convert('energy', units)))
            f.write('\n')
            f.close()

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
        log(' Last commit: %s' % commithash)
        log(' Last commit date: %s' % commitdate)
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
        # ZMQ an pxssh are only necessary for parallel calculations.
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
        from .descriptor.gaussian import Gaussian as Module
    elif importname == '.descriptor.zernike.Zernike':
        from .descriptor.zernike import Zernike as Module
    elif importname == '.descriptor.bispectrum.Bispectrum':
        from .descriptor.bispectrum import Bispectrum as Module
    elif importname == '.model.neuralnetwork.NeuralNetwork':
        from .model.neuralnetwork import NeuralNetwork as Module
    elif importname == '.model.neuralnetwork.tflow':
        from .model.tflow import NeuralNetwork as Module
    elif importname == '.model.LossFunction':
        from .model import LossFunction as Module
    else:
        raise NotImplementedError(
            'Attempt to import the module %s. Was this intended? '
            'If so, trying manually importing this module and '
            'feeding it to Amp.load. To avoid this error, this '
            'module can be added to amp.importhelper.' %
            importname)

    return Module


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
    output = output.strip()
    commithash, commitdate = output.split(b'\t')
    os.chdir(pwd)
    return commithash, commitdate
