"""Nengo OCL version information.

We use semantic versioning (see http://semver.org/).
and confrom to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""
from pkg_resources import SetuptoolsVersion

class Version(SetuptoolsVersion):
    def __getitem__(self, items):
        return Version('.'.join(str(self).split('.')[items]))


# --- version of this release
name = "nengo_ocl"
version_info = (1, 0, 0)  # (major, minor, patch)
dev = 0
version = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev=('.dev%d' % dev) if dev is not None else '')

# --- latest Nengo version at time of release
latest_nengo_version_info = (2, 1, 0)  # (major, minor, patch)
latest_nengo_version = '.'.join(str(v) for v in latest_nengo_version_info)
