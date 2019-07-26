#!/usr/bin/env python

import os
import numpy as np
from ase.io import Trajectory

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

from . import Amp
from .utilities import now, hash_images, make_filename

rcParams.update({'figure.autolayout': True})


def plot_sensitivity(calc,
                     images,
                     d=0.0001,
                     label='sensitivity',
                     dblabel=None,
                     plotfile=None,
                     overwrite=False,
                     energy_coefficient=1.0,
                     force_coefficient=0.04,
                     ):
    """Returns the plot of loss function in terms of perturbed parameters.

    Takes the load file and images. Any other keyword taken by the Amp
    calculator can be fed to this class also.

    Parameters
    ----------
    calc : Amp object or str
        Either an existing instantiated Amp calculator or a path for loading an
        existing ".amp" file. In the latter case, should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    d : float
        The amount of perturbation in each parameter.
    label : str
        Default prefix/location used for all files.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    plotfile : Object
        File for the plot.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    """

    from amp.model import LossFunction

    if isinstance(calc, str):
        calc = Amp.load(file=calc)

    if plotfile is None:
        plotfile = make_filename(label, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.' % plotfile)

    calc.dblabel = label if dblabel is None else dblabel

    if force_coefficient == 0.:
        calculate_derivatives = False
    else:
        calculate_derivatives = True

    calc._log('\nAmp sensitivity analysis started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    vector = calc.model.vector.copy()

    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                parallel=calc._parallel,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    calc.model.lossfunction.attach_model(
        calc.model,
        log=calc._log,
        fingerprints=calc.descriptor.fingerprints,
        fingerprintprimes=calc.descriptor.fingerprintprimes,
        images=images)

    originalloss = calc.model.lossfunction.get_loss(
        vector, lossprime=False)['loss']

    calc._log('\n Perturbing parameters...', tic='perturb')

    allparameters = []
    alllosses = []
    num_parameters = len(vector)

    for count in range(num_parameters):
        calc._log('parameter %i out of %i' % (count + 1, num_parameters))
        parameters = []
        losses = []
        # parameter is perturbed -d and loss function calculated.
        vector[count] -= d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        vector[count] += d
        parameters.append(vector[count])
        losses.append(originalloss)
        # parameter is perturbed +d and loss function calculated.
        vector[count] += d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        allparameters.append(parameters)
        alllosses.append(losses)
        # returning back to the original value.
        vector[count] -= d

    calc._log('...parameters perturbed and loss functions calculated',
              toc='perturb')

    calc._log('Plotting loss function vs perturbed parameters...',
              tic='plot')

    with PdfPages(plotfile) as pdf:
        count = 0
        for parameter in vector:
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.plot(allparameters[count],
                    alllosses[count],
                    marker='o', linestyle='--', color='b',)

            xmin = allparameters[count][0] - \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            xmax = allparameters[count][-1] + \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            ymin = min(alllosses[count]) - \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ymax = max(alllosses[count]) + \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            ax.set_xlabel('parameter no %i' % count)
            ax.set_ylabel('loss function')
            pdf.savefig(fig)
            pyplot.close(fig)
            count += 1

    calc._log(' ...loss functions plotted.', toc='plot')


def plot_parity_and_error(calc,
                          images,
                          label_parity='parity',
                          label_error='error',
                          dblabel=None,
                          xtic_angle=45.,
                          plot_forces=True,
                          plotfile_parity=None,
                          plotfile_error=None,
                          color='b.',
                          overwrite=False,
                          returndata=False,
                          ):
    """Makes a parity plot and an error plot of Amp energies and forces versus
    real energies and forces.

    Parameters
    ----------
    calc : Amp object or str
        Either an existing instantiated Amp calculator or a path for loading an
        existing ".amp" file. In the latter case, should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    label_parity : str
        Default prefix/location used for the parity plot.
    label_error : str
        Default prefix/location used for the error plot.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    xtic_angle : float
        Set the xtics angles. Default is 45.
    plot_forces : bool
        Determines whether or not forces should be plotted as well.
    plotfile_parity : Object
        File for the parity plot.
    plotfile_error : Object
        File for the error plot.
    color : str
        Plot color.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    returndata : bool
        Whether to return a reference to the figures and their data or not.
    """

    if isinstance(calc, str):
        calc = Amp.load(file=calc, dblabel=dblabel)

    if plotfile_parity is None:
        plotfile_parity = make_filename(label_parity, '-plot.pdf')
    if plotfile_error is None:
        plotfile_error = make_filename(label_error, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile_parity):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile_parity)
    if (not overwrite) and os.path.exists(plotfile_error):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile_error)

    if plot_forces is True:
        calculate_derivatives = True
    else:
        calculate_derivatives = False

    calc._log('\nAmp parity and error plots started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=calc._log)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    calc._log('Calculating potential energies...', tic='pot-energy')
    energy_data = {}
    for hash, image in images.items():
        no_of_atoms = len(image)
        energy_args = dict(
                fingerprints=calc.descriptor.fingerprints[hash],
                )

        model_name = calc.model.__class__.__name__

        if model_name == 'KernelRidge':
            if calc.model.trainingimages is not None:
                trainingimages = hash_images(
                    Trajectory(calc.model.trainingimages))
                energy_args['trainingimages'] = trainingimages
                calc.descriptor.calculate_fingerprints(
                        images=trainingimages,
                        parallel=calc._parallel,
                        log=calc._log,
                        calculate_derivatives=calculate_derivatives
                        )
                fp_trainingimages = calc.descriptor.fingerprints
                energy_args['fp_trainingimages'] = fp_trainingimages
                energy_args['hash'] = hash

        amp_energy = calc.model.calculate_energy(**energy_args)
        actual_energy = image.get_potential_energy(apply_constraint=False)
        act_energy_per_atom = actual_energy / no_of_atoms
        energy_error = abs(amp_energy - actual_energy) / no_of_atoms
        energy_data[hash] = [actual_energy, amp_energy, act_energy_per_atom,
                             energy_error]
    calc._log('...potential energies calculated.', toc='pot-energy')

    # calculating minimum and maximum energies
    min_act_energy = min([energy_data[hash][0]
                          for hash, image in images.items()])
    max_act_energy = max([energy_data[hash][0]
                          for hash, image in images.items()])
    min_act_energy_per_atom = min([energy_data[hash][2]
                                   for hash, image in images.items()])
    max_act_energy_per_atom = max([energy_data[hash][2]
                                   for hash, image in images.items()])

    # calculating energy per atom rmse
    energy_square_error = 0.
    for hash, image in images.items():
        energy_square_error += energy_data[hash][3] ** 2.
    energy_per_atom_rmse = np.sqrt(energy_square_error / len(images))

    if plot_forces is True:
        calc._log('Calculating forces...', tic='forces')
        force_data = {}
        for hash, image in images.items():
            forces_args = dict(
                    fingerprints=calc.descriptor.fingerprints[hash],
                    fingerprintprimes=calc.descriptor.fingerprintprimes[hash]
                    )

            if model_name == 'KernelRidge':
                if calc.model.trainingimages is not None:
                    trainingimages = \
                            hash_images(Trajectory(calc.model.trainingimages))
                    calc.descriptor.calculate_fingerprints(
                            images=trainingimages,
                            calculate_derivatives=True
                            )
                    t_descriptor = calc.descriptor
                    forces_args['trainingimages'] = trainingimages
                    forces_args['t_descriptor'] = t_descriptor

            amp_forces = calc.model.calculate_forces(**forces_args)
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [actual_forces, amp_forces,
                                abs(np.array(amp_forces) -
                                    np.array(actual_forces))]
        calc._log('...forces calculated.', toc='forces')

        min_act_force = min([force_data[hash][0][index][k]
                             for hash, image in images.items()
                             for index in range(len(image))
                             for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                             for hash, image in images.items()
                             for index in range(len(image))
                             for k in range(3)])

        # calculating force rmse
        force_square_error = 0.
        for hash, image in images.items():
            no_of_atoms = len(image)
            for index in range(no_of_atoms):
                for k in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][2][index][k] ** 2.) / \
                        no_of_atoms
        force_rmse = np.sqrt(force_square_error / len(images))

    # make parity plot
    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energy parities...', tic='energy-plot')
    ax.plot(list(zip(*np.vstack(energy_data.values())))[0],
            list(zip(*np.vstack(energy_data.values())))[1], color)
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel("ab initio energy, eV")
    ax.set_ylabel("Amp energy, eV")
    ax.set_title("Energies")
    ax.tick_params(axis='x', rotation=xtic_angle)
    calc._log('...energies plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        calc._log('Plotting forces...', tic='force-plot')
        ax.plot(np.hstack(force_data.values())[0].flatten(),
                np.hstack(force_data.values())[1].flatten(), color)
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("Amp force, eV/Ang")
        ax.set_title("Forces")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...forces plotted.', toc='force-plot')

    fig.savefig(plotfile_parity)
    pyplot.close(fig)

    # make error plot
    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energy errors...', tic='energy-plot')
    ax.plot(list(zip(*np.vstack(energy_data.values())))[2],
            list(zip(*np.vstack(energy_data.values())))[3], color)
    # draw horizontal line for rmse
    ax.plot([min_act_energy_per_atom, max_act_energy_per_atom],
            [energy_per_atom_rmse, energy_per_atom_rmse],
            color='black', linestyle='dashed', lw=1,)
    ax.text(max_act_energy_per_atom,
            energy_per_atom_rmse,
            'energy rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='black')
    ax.set_xlabel("ab initio energy (eV) per atom")
    ax.set_ylabel("$|$ab initio energy - Amp energy$|$ / number of atoms")
    ax.set_title("Energies")
    ax.tick_params(axis='x', rotation=xtic_angle)
    calc._log('...energy errors plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        calc._log('Plotting force errors...', tic='force-plot')
        ax.plot(np.hstack(force_data.values())[0].flatten(),
                np.hstack(force_data.values())[2].flatten(), color)
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                color='black',
                linestyle='dashed',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'force rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='black',)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("$|$ab initio force - Amp force$|$")
        ax.set_title("Forces")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...force errors plotted.', toc='force-plot')

    fig.savefig(plotfile_error)
    pyplot.close(fig)

    if returndata:
        if plot_forces is False:
            return energy_data
        else:
            return energy_data, force_data


def calc_rmse(calc_paths,
               images,
               cores=None,
               dblabel=None,
               energy_coefficient=1.0,
               force_coefficient=0.04,
              ):
    """Calculates energy and force RMSEs for a set of Amp calculators. All
    calculators must have the same descriptors and models.

    Parameters
    ----------
    calc_paths : list
        List of paths for loading existing ".amp" files.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    cores : int
        Can specify cores to use for parallel processing; if None, will
        determine from environment
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    """

    from amp.model import LossFunction

    calcs = []
    calc = Amp.load(file=calc_paths[0], cores=cores, dblabel=dblabel)
    calcs.append(calc)
    for i in range(1,len(calc_paths)):
        calcs.append(Amp.load(file=calc_paths[i], cores=cores,
                              dblabel=dblabel, logging=False))
        calc._log('Loaded file: %s' % calc_paths[i])

    if force_coefficient == 0.:
        calculate_derivatives = False
        convergence = {'energy_rmse': 0.001}
    else:
        calculate_derivatives = True
        convergence = {'energy_rmse': 0.001, 'force_rmse': 0.01}
    # Setting the convergence is a kludgy way to keep LossFunction.__init__()
    # from resetting the force_coefficient to 0

    calc._log('\nAmp calc_rmse started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                parallel=calc._parallel,
                                raise_ConvergenceOccurred=False,
                                convergence=convergence,
                                )
    calc.model.lossfunction = lossfunction

    if force_coefficient == 0.:
        calc.model.lossfunction.attach_model(
            calc.model,
            log=calc._log,
            fingerprints=calc.descriptor.fingerprints,
            images=images)
    else:
        calc.model.lossfunction.attach_model(
            calc.model,
            log=calc._log,
            fingerprints=calc.descriptor.fingerprints,
            fingerprintprimes=calc.descriptor.fingerprintprimes,
            images=images)

    steps, loss, energy_loss, force_loss = [], [], [], []
    energy_maxresid, force_maxresid, energy_rmse, force_rmse = [], [], [], []
    for i in range(len(calc_paths)):
        steps.append(int(os.path.basename(calc_paths[i])[:-4]))
        vector = calcs[i].model.vector.copy()
        results = calc.model.lossfunction.get_loss(
                      vector,
                      lossprime=calculate_derivatives)
        loss.append(results['loss'])
        energy_loss.append(results['energy_loss'])
        force_loss.append(results['force_loss'])
        energy_maxresid.append(results['energy_maxresid'])
        force_maxresid.append(results['force_maxresid'])
        energy_rmse.append(np.sqrt(energy_loss[i] / len(images)))
        if force_coefficient == 0.:
            calc._log('%5i %19s %12.4e   %10.4e   %10.4e' %
                      (steps[i], now(), loss[i], energy_rmse[i],
                       energy_maxresid[i]))
        else:
            force_rmse.append(np.sqrt(force_loss[i] / len(images)))
            calc._log('%5i %19s %12.4e   %10.4e  '
                      ' %10.4e   %10.4e   %10.4e' %
                      (steps[i], now(), loss[i], energy_rmse[i],
                       energy_maxresid[i], force_rmse[i] ,force_maxresid[i]))

    data = {}
    data['steps'] = steps
    data['loss'] = loss
    data['energy_loss'] = energy_loss
    data['force_loss'] = force_loss
    data['energy_maxresid'] = energy_maxresid
    data['force_maxresid'] = force_maxresid
    data['energy_rmse'] = energy_rmse
    if force_coefficient != 0.:
        data['force_rmse'] = force_rmse
    return data


def calculate_rmses(calc_path,
               images,
               dblabel=None,
               energy_coefficient=1.0,
               force_coefficient=1.0,
              ):
    from amp.model import LossFunction

    calc = Amp.load(calc_path, dblabel=dblabel)
    # Setting the convergence is a kludgy way to keep LossFunction.__init__()
    # from resetting the force_coefficient to 0
    convergence = {'energy_rmse': 1e-16, 'force_rmse': 1e-16}

    calc._log('\nAmp calculate_rmses started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=True)
    
    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                parallel=calc._parallel,
                                raise_ConvergenceOccurred=False,
                                convergence=convergence,
                                )
    calc.model.lossfunction = lossfunction
    calc.model.lossfunction.attach_model(
            calc.model,
            log=calc._log,
            fingerprints=calc.descriptor.fingerprints,
            fingerprintprimes=calc.descriptor.fingerprintprimes,
            images=images)

    vector = calc.model.vector.copy()
    results = calc.model.lossfunction.get_loss(
                  vector,
                  lossprime=True)

    calc._log('Finished calculating RMSE on validation set')

    data = results
    data["energy_rmse"] = np.sqrt(data["energy_loss"] / len(images))
    data["force_rmse"] = np.sqrt(data["force_loss"] / len(images))

    return data


def calculate_error(calc_path,
                    images,
                    dblabel=None,
                    ):
    calc = Amp.load(calc_path, dblabel=dblabel)
    calc._log('\nAmp calculate_error started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=calc._log)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=True)

    keys = list(images.keys())
    num_images = len(images)
    num_atoms = len(images[keys[0]])
    energy_exact = np.zeros(num_images)
    energy_amp = np.zeros(num_images)
    force_exact = np.zeros((num_images, 3 * num_atoms))
    force_amp = np.zeros((num_images, 3 * num_atoms))
    energy_diff = np.zeros(num_images)
    force_diff = np.zeros((num_images, 3 * num_atoms))

    energy_rmse = 0.0
    force_rmse = 0.0
    for i, (hash, image) in enumerate(images.items()):
        energy_args = dict(
            fingerprints=calc.descriptor.fingerprints[hash]
        )

        energy_amp[i] = calc.model.calculate_energy(**energy_args)
        energy_exact[i] = image.get_potential_energy(apply_constraint=False)

        forces_args = dict(
            fingerprints=calc.descriptor.fingerprints[hash],
            fingerprintprimes=calc.descriptor.fingerprintprimes[hash]
        )

        force_amp[i] = calc.model.calculate_forces(**forces_args).reshape(-1)
        force_exact[i] = image.get_forces(apply_constraint=False).reshape(-1)

        energy_diff[i] = abs(energy_amp[i] - energy_exact[i])
        force_diff[i] = abs(force_amp[i] - force_exact[i])
        energy_rmse += np.sum((energy_amp[i] - energy_exact[i]) ** 2)
        force_rmse += np.sum((force_amp[i] - force_exact[i]) ** 2)

    energy_rmse = np.sqrt(energy_rmse / num_images)
    force_rmse = np.sqrt(force_rmse / (num_images * num_atoms))
    force_exact = force_exact.reshape(-1)
    force_diff = force_diff.reshape(-1)

    return (
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )


def read_trainlog(logfile, verbose=True, multiple=0):
    """Reads the log file from the training process, returning the relevant
    parameters.

    Parameters
    ----------
    logfile : str
        Name or path to the log file.

    verbose : bool
        Write out logfile during analysis.

    multiple : int or True
        If multiple training sessions are recorded in the same log file,
        return session number <multiple> (counting from 0). If set to True,
        returns all sessions as list.
    """
    data = {}

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()

    # Get number of training sets.
    multiple_starts = []
    for index, line in enumerate(lines):
        if line.startswith('Amp training started.'):
            multiple_starts.append(index)
    if multiple is True:
        datalist = []
        for index in range(len(multiple_starts)):
            datalist.append(read_trainlog(logfile, verbose,
                                          multiple=index))
        return datalist
    else:
        lines = lines[multiple_starts[multiple]:]

    def print_(text):
        if verbose:
            print(text)

    # Get number of images.
    for line in lines:
        if 'unique images after hashing.' in line:
            no_images = int(line.split()[0])
            break
    data['no_images'] = no_images

    # Find where convergence data starts.
    startline = None
    for index, line in enumerate(lines):
        if 'Loss function convergence criteria:' in line:
            startline = index
            data['convergence'] = {}
            d = data['convergence']
            break
    else:
        return data

    # Get convergence parameters.
    ready = [False] * 7
    for index, line in enumerate(lines[startline:]):
        if 'energy_rmse:' in line:
            ready[0] = True
            d['energy_rmse'] = float(line.split(':')[-1])
        elif 'force_rmse:' in line:
            ready[1] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_rmse'] = None
                trainforces = False
            else:
                d['force_rmse'] = float(line.split(':')[-1])
                trainforces = True
            print_('train forces: %s' % trainforces)
        elif 'force_coefficient:' in line:
            ready[2] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_coefficient'] = 0.
            else:
                d['force_coefficient'] = float(_)
        elif 'energy_coefficient:' in line:
            ready[3] = True
            d['energy_coefficient'] = float(line.split(':')[-1])
        elif 'energy_maxresid:' in line:
            ready[5] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['energy_maxresid'] = None
            else:
                d['energy_maxresid'] = float(_)
        elif 'force_maxresid:' in line:
            ready[6] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_maxresid'] = None
            else:
                d['force_maxresid'] = float(_)
        elif 'Step' in line and 'Time' in line:
            ready[4] = True
            startline += index + 2
        if ready == [True] * 7:
            break

    for _ in d.items():
        print_('{}: {}'.format(_[0], _[1]))
    E = d['energy_rmse']**2 * no_images
    if trainforces:
        F = d['force_rmse']**2 * no_images
    else:
        F = 0.
    costfxngoal = d['energy_coefficient'] * E + d['force_coefficient'] * F
    d['costfxngoal'] = costfxngoal

    # Extract data (emrs and fmrs are max residuals).
    steps, es, fs, emrs, fmrs, costfxns = [], [], [], [], [], []
    costfxnEs, costfxnFs = [], []
    index = startline
    d['converged'] = None
    while index < len(lines):
        line = lines[index]
        if 'Saving checkpoint data.' in line:
            index += 1
            continue
        elif 'Overwriting file' in line:
            index += 1
            continue
        elif 'optimization completed successfully.' in line:  # old version
            d['converged'] = True
            break
        elif '...optimization successful.' in line:
            d['converged'] = True
            break
        elif 'could not find parameters for the' in line:
            break
        elif '...optimization unsuccessful.' in line:
            d['converged'] = False
            break
        elif len(line) == 0:
            # Job apparently timed out.
            break
        print_(line)
        if trainforces:
            step, time, costfxn, e, _, emr, _, f, _, fmr, _ = line.split()
            fs.append(float(f))
            fmrs.append(float(fmr))
            F = float(f)**2 * no_images
            costfxnFs.append(d['force_coefficient'] * F / float(costfxn))
        else:
            step, time, costfxn, e, _, emr, _ = line.split()
        steps.append(int(step))
        es.append(float(e))
        emrs.append(float(emr))
        costfxns.append(float(costfxn))
        E = float(e)**2 * no_images
        costfxnEs.append(d['energy_coefficient'] * E / float(costfxn))
        index += 1
    d['steps'] = steps
    d['es'] = es
    d['fs'] = fs
    d['emrs'] = emrs
    d['fmrs'] = fmrs
    d['costfxns'] = costfxns
    d['costfxnEs'] = costfxnEs
    d['costfxnFs'] = costfxnFs

    return data


def plot_convergence(data, plotfile='convergence.pdf'):
    """Makes a plot of the convergence of the cost function and its energy
    and force components.

    Parameters
    ----------
    data : dict
        Convergence data dictionary as returned by read_trainlog.
    plotfile : str or None
        Name or path to the plot file. If None, instead returns reference to
        the created figure.
    """

    # Find if multiple runs contained in data set.
    d = data['convergence']
    steps = range(len(d['steps']))
    breaks = []
    for index, step in enumerate(d['steps'][1:]):
        if step < d['steps'][index]:
            breaks.append(index)

    # Make plots.
    fig = pyplot.figure(figsize=(6., 8.))
    # Margins, vertical gap, and top-to-bottom ratio of figure.
    lm, rm, bm, tm, vg, tb = 0.12, 0.05, 0.08, 0.03, 0.08, 4.
    bottomaxheight = (1. - bm - tm - vg) / (tb + 1.)

    ax = fig.add_axes((lm, bm + bottomaxheight + vg,
                       1. - lm - rm, tb * bottomaxheight))
    ax.semilogy(steps, d['es'], 'b', lw=2, label='energy rmse')
    ax.semilogy(steps, d['emrs'], 'b:', lw=2, label='energy maxresid')
    if d['force_rmse']:
        ax.semilogy(steps, d['fs'], 'g', lw=2, label='force rmse')
        ax.semilogy(steps, d['fmrs'], 'g:', lw=2, label='force maxresid')
    ax.semilogy(steps, d['costfxns'], color='0.5', lw=2,
                label='loss function')
    # Targets.
    if d['energy_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_rmse']] * 2,
                    color='b', linestyle='-', alpha=0.5)
    if d['energy_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_maxresid']] * 2,
                    color='b', linestyle=':', alpha=0.5)
    if d['force_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['force_rmse']] * 2,
                    color='g', linestyle='-', alpha=0.5)
    if d['force_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['force_maxresid']] * 2,
                    color='g', linestyle=':', alpha=0.5)
    ax.set_ylabel('error')
    ax.legend(loc='best', fontsize=9.)
    if len(breaks) > 0:
        ylim = ax.get_ylim()
        for b in breaks:
            ax.plot([b] * 2, ylim, '--k')

    if d['force_rmse']:
        # Loss function component plot.
        axf = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
        axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                         color='blue')
        axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                         y2=np.array(d['costfxnEs']) +
                         np.array(d['costfxnFs']),
                         color='green')
        axf.set_ylabel('loss function component')
        axf.set_xlabel('loss function call')
        axf.set_ylim(0, 1)
    else:
        ax.set_xlabel('loss function call')

    if plotfile is None:
        return fig
    fig.savefig(plotfile)
    pyplot.close(fig)
