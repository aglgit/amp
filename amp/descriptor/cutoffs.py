#!/usr/bin/env python
"""
This script contains different cutoff function forms.

Note all cutoff functions need to have a "todict" method
to support saving/loading as an Amp object.

All cutoff functions also need to have an `Rc` attribute which
is the maximum distance at which properties are calculated; this
will be used in calculating neighborlists.

"""

import numpy as np


def dict2cutoff(dct):
    """This function converts a dictionary (which was created with the
    to_dict method of one of the cutoff classes) into an instantiated
    version of the class. Modeled after ASE's dict2constraint function.
    """
    if len(dct) != 2:
        raise RuntimeError('Cutoff dictionary must have only two values,'
                           ' "name" and "kwargs".')
    return globals()[dct['name']](**dct['kwargs'])


class Cosine(object):

    """Cosine functional form suggested by Behler.

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return 0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.)

    def prime(self, Rij):
        """
        Derivative of the Cosine cutoff function.

        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return -0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc)

    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc}}

    def __repr__(self):
        return ('Cosine cutoff with Rc=%.3f from amp.descriptor.cutoffs'
                % self.Rc)


class Polynomial(object):

    """Polynomial functional form suggested by Khorshidi and Peterson.

    :param gamma: The power of polynomial.
    :type gamma: float

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    """

    def __init__(self, Rc, gamma=4):
        self.gamma = gamma
        self.Rc = Rc

    def __call__(self, Rij):
        """
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            value = 1. + self.gamma * (Rij / self.Rc) ** (self.gamma + 1) - \
                (self.gamma + 1) * (Rij / self.Rc) ** self.gamma
            return value

    def prime(self, Rij):
        """
        Derivative of the Cosine cutoff function.

        :param Rc: Radius above which neighbor interactions are ignored.
        :type Rc: float
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            value = (self.gamma * (self.gamma + 1) / self.Rc) * \
                ((Rij / self.Rc) ** self.gamma -
                 (Rij / self.Rc) ** (self.gamma - 1))
            return value

    def todict(self):
        return {'name': 'Polynomial',
                'kwargs': {'Rc': self.Rc}}

    def __repr__(self):
        return ('Polynomial cutoff with Rc=%.3f and gamma=%i '
                'from amp.descriptor.cutoffs'
                % (self.Rc, self.gamma))
