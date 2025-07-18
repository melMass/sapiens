# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# __version__ = '0.8.4' ## forked mmengine version
# __version__ = '1.0.0' ## forked sapiens-engine version
__version__ = "1.0.1"


def parse_version_info(version_str):
    """Parse the version information.

    Args:
        version_str (str): version string like '0.1.0'.

    Returns:
        tuple: version information contains major, minor, micro version.
    """
    version_info = []
    for x in version_str.split("."):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find("rc") != -1:
            patch_version = x.split("rc")
            version_info.append(int(patch_version[0]))
            version_info.append(f"rc{patch_version[1]}")
    return tuple(version_info)


version_info = parse_version_info(__version__)
