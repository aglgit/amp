.. _UseAmp:

==================================
Using Amp
==================================

If you are familiar with ASE, the use of Amp should be intuitive.
At it's most basic, Amp behaves like any other ASE calculator, except that it has a key extra method, called `train`, which allows you to fit the calculator to a set of atomic images.
This means you can use Amp as a substitute for an expensive calculator in any atomistic routine, such as molecular dynamics, global optimization, transition-state searches, normal-mode analyses, phonon analyses, etc.

----------------------------------
Basic use
----------------------------------

To use Amp, you need to specify a `descriptor` and a `model`.
The below shows a basic example of training Amp with Gaussian descriptors and a neural network model---the Behler-Parinello scheme.

.. code-block:: python

   from amp import Amp
   from amp.descriptor.gaussian import Gaussian
   from amp.model.neuralnetwork import NeuralNetwork

   calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(),
              label='calc')
   calc.train(images='my-images.traj')

After training is successful you can use your trained calculator just like any other ASE calculator (although you should be careful that you can only trust it within the trained regime).
This will also result in the saving the calculator parameters to "<label>.amp", which can be used to re-load the calculator in a future session:

.. code-block:: python

   calc = Amp.load('calc.amp')


The modular nature of Amp is meant such that you can easily mix-and-match different descriptor and model schemes.
See the theory section for more details.

----------------------------------
Adjusting convergence parameters
----------------------------------

To control how tightly the energy is converged, you can adjust the `LossFunction`. Just insert before the `calc.train` line some code like:

.. code-block:: python

   from amp.model import LossFunction

   convergence = {'energy_rmse': 0.02, 'force_rmse': 0.04}
   calc.model.lossfunction = LossFunction(convergence=convergence)

You can see the adjustable parameters and their default values in the dictionary `LossFunction.default_parameters`.

To change how the code manages the regression process, you can use the `Regressor` class. For example, to switch from the scipy's fmin_bfgs optimizer (the default) to scipy's basin hopping optimizer, try inserting the following lines before initializing training:

.. code-block:: python

   from amp.regression import Regressor
   from scipy.optimize import basinhopping

   regressor = Regressor(optimizer=basinhopping)
   calc.model.regressor = regressor

----------------------------------
Parallel processing
----------------------------------

Most tasks in Amp are "embarrassingly parallel" and thus you should see a performance boost by specifying more cores.
Our standard parallel processing approach requires the modules pxssh (part of Pexpect, establish SSH connections) and ZMQ (to pass messages between processes).
The code will try to automatically guess the parallel configuration from the environment variables that your batching system produces, using the function `amp.utilities.assign_cores`.
(We only use SLURM on your system, so we welcome patches to get this utility working on other systems!)
If you want to override the automatic guess, use the `cores` keyword when initializing Amp.
To specify serial operation, use `cores=1`.
For parallel operation, cores should be a dictionary where the keys are the hostnames and the values are the number of processors (cores) available on that node; e.g.,

.. code-block:: python

   cores = {'node241': 16,
            'node242': 16}

----------------------------------
Advanced use
----------------------------------

Under the hood, the train function is pretty simple; it just runs:

.. code-block:: python

   images = hash_images(images, ...)
   self.descriptor.calculate_fingerprints(images, ...)
   result = self.model.fit(images, self.descriptor, ...)
   if result is True:
       self.save(filename)

* In the first line, the images are read and converted to a dictionary, addressed by a hash.
  This makes addressing the images simpler across modules and eliminates duplicate images.
  This also facilitates keeping a database of fingerprints, such that in future scripts you do not need to re-fingerprint images you have already encountered.

* In the second line, the descriptor converts the images into fingerprints, one fingerprint per image. There are two possible modes a descriptor can operate in: "image-centered" in which one vector is produced per image, and "atom-centered" in which one vector is produced per atom. The resulting fingerprint is stored in `descriptor.fingerprints`, and the mode is stored in self.parameters.mode.

* In the third line, the model (e.g., a neural network) is fit to the data. As it is passed a reference to `self.descriptor`, it has access to the fingerprints as well as the mode. Many options are available to customize this in terms of the loss function, the regression method, etc.

* In the final pair of lines, if the target fit was achieved, the model is saved to disk.


----------------------------------
Example scripts
----------------------------------

With the modular nature, it's straightforward to analyze how fingerprints change with changes in images.
The below script makes an animated GIF that shows how a fingerprint about the O atom in water changes as one of the O-H bonds is stretched.
Note that most of the lines of code below are either making the atoms or making the figure; very little effort is needed to produce the fingerprints themselves---this is done in three lines.

.. code-block:: python

 # Make a series of images.
 import numpy as np
 from ase.structure import molecule
 from ase import Atoms
 atoms = molecule('H2O')
 atoms.rotate('y', -np.pi/2.)
 atoms.set_pbc(False)
 displacements = np.linspace(0.9, 8.0, 20)
 vec = atoms[2].position - atoms[0].position
 images = []
 for displacement in displacements:
     atoms = Atoms(atoms)
     atoms[2].position = (atoms[0].position + vec * displacement)
     images.append(atoms)
 
 # Fingerprint using Amp.
 from amp.utilities import hash_images
 from amp.descriptor.gaussian import Gaussian
 images = hash_images(images, ordered=True)
 descriptor = Gaussian()
 descriptor.calculate_fingerprints(images)
 
 # Plot the data.
 from matplotlib import pyplot
 
 def barplot(hash, name, title):
     """Makes a barplot of the fingerprint about the O atom."""
     fp = descriptor.fingerprints[hash][0]
     fig, ax = pyplot.subplots()
     ax.bar(range(len(fp[1])), fp[1])
     ax.set_title(title)
     ax.set_ylim(0., 2.)
     ax.set_xlabel('fingerprint')
     ax.set_ylabel('value')
     fig.savefig(name)
 
 for index, hash in enumerate(images.keys()):
     barplot(hash, 'bplot-%02i.png' % index,
             '%.2f$\\times$ equilibrium O-H bondlength'
             % displacements[index])
 
 # For fun, make an animated gif.
 import os
 filenames = ['bplot-%02i.png' % index for index in range(len(images))]
 command = ('convert -delay 100 %s -loop 0 animation.gif' %
            ' '.join(filenames))
 os.system(command)


.. image:: _static/animation.gif
   :scale: 80 %
   :align: center
