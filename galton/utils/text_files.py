# -*- coding: utf-8 -*-
import io


def read(*filenames, encoding='utf-8', sep='\n'):
    """Read text files and concatenate their contents.

    Parameters
    ----------
    *filenames: str
        Arbitrary number of text file paths.

    **kwargs:
        Arbitrary keyword arguments.
        encoding: str
            File encondings. Default: 'utf-8'
        sep: str
            Separator between files. Default: '\n'
    Returns
    -------
    str
        The concatenated contents of all text files.
    """
    #encoding = kwargs.get('encoding', 'utf-8')
    #sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)
