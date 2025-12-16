"""
Spectral TIFF Utilities for ODSI-DB Dataset
============================================
Functions for reading and writing spectral TIFF (stiff) and mask TIFF (mtiff) files.

Based on the ODSI-DB (Oral and Dental Spectral Image Database) format.
"""

import warnings
from typing import Any, Optional, Tuple, Dict

from tifffile import TiffFile, TiffWriter
import numpy as np


# Custom TIFF tags for spectral data
TIFFTAG_WAVELENGTHS = 65000
TIFFTAG_MASK_LABEL = 65001
TIFFTAG_METADATA = 65111


def read_stiff(filename: str, silent: bool = False, rgb_only: bool = False) -> Tuple[
    Optional[np.ndarray], np.ndarray, Optional[np.ndarray], str
]:
    """
    Read a spectral TIFF file.
    
    :param filename: filename of the spectral tiff to read.
    :param silent: if True, suppress warnings about duplicated tags.
    :param rgb_only: if True, only read the RGB preview image.
    :return: Tuple[spim, wavelengths, rgb, metadata], where
        spim: spectral image cube of form [height, width, bands],
        wavelengths: the center wavelengths of the bands,
        rgb: a color render of the spectral image [height, width, channels] or None
        metadata: a free-form metadata string stored in the image, or an empty string
    """
    spim = None
    wavelengths = None
    rgb = None
    metadata = None
    first_band_page = 0

    with TiffFile(filename) as tiff:
        # The RGB image is optional, the first band image maybe on the first page:
        first_band_page = 0
        if tiff.pages[first_band_page].ndim == 3:
            rgb = tiff.pages[0].asarray()
            # Ok, the first band image is on the second page
            first_band_page = first_band_page + 1

        multiple_wavelength_lists = False
        multiple_metadata_fields = False

        for band_page in range(first_band_page, len(tiff.pages)):
            # The wavelength list is supposed to be on the first band image.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_WAVELENGTHS)
            tag_value = tag.value if tag else tuple()
            if tag_value:
                if wavelengths is None:
                    wavelengths = tag_value
                elif wavelengths == tag_value:
                    multiple_wavelength_lists = True
                elif wavelengths != tag_value:
                    raise RuntimeError(
                        f'Spectral-Tiff "{filename}" contains multiple differing wavelength lists!'
                    )

            # The metadata string
            tag = tiff.pages[band_page].tags.get(TIFFTAG_METADATA)
            tag_value = tag.value if tag else ''
            if tag_value:
                if metadata is None:
                    metadata = tag_value
                elif metadata == tag_value:
                    multiple_metadata_fields = True
                elif metadata != tag_value:
                    raise RuntimeError(
                        f'Spectral-Tiff "{filename}" contains multiple differing metadata fields!'
                    )

        # Decode metadata from ASCII with unicode escapes
        if metadata:
            metadata = metadata.encode('ascii').decode('unicode-escape')
        else:
            metadata = ''

        # Fix erroneous metadata strings
        if metadata and metadata[0] == "'" and metadata[-1] == "'":
            while metadata[0] == "'":
                metadata = metadata[1:]
            while metadata[-1] == "'":
                metadata = metadata[:-1]
            if '\\n' in metadata:
                metadata = metadata.replace('\\n', '\n')

        # Generate fake wavelength list if missing
        if not wavelengths:
            wavelengths = range(0, len(tiff.pages) - 1 if rgb is not None else len(tiff.pages))

        if multiple_wavelength_lists and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated wavelength lists!')

        if multiple_metadata_fields and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated metadata fields!')

        if not rgb_only:
            spim = tiff.asarray(key=range(first_band_page, len(tiff.pages)))
            spim = np.transpose(spim, (1, 2, 0))
        else:
            spim = None

    # Make sure wavelengths are in ascending order
    if wavelengths[0] > wavelengths[-1]:
        spim = spim[:, :, ::-1] if spim is not None else None
        wavelengths = wavelengths[::-1]

    # Convert uint16 cube back to float32 cube
    if spim is not None and spim.dtype == 'uint16':
        spim = spim.astype('float32') / (2**16 - 1)

    return spim, np.array(wavelengths), rgb, metadata


def write_stiff(
    filename: str,
    spim: np.ndarray,
    wls: np.ndarray,
    rgb: Optional[Any],
    metadata: str = ''
) -> None:
    """
    Write a spectral image cube into a Spectral Tiff.
    
    A spectral tiff contains two custom tags to describe the data cube:
    - wavelength list is stored in tag 65000 as a list of float32s
    - a metadata string is stored in tag 65111 as a UTF-8 encoded byte string
    
    :param filename: the filename of the spectral tiff to save the data cube in
    :param spim: the spectral image data cube, expected dimensions [height, width, bands]
    :param wls: the wavelength list, length must match number of bands
    :param rgb: color image render of the spectral image cube (optional, shown as preview)
    :param metadata: a free-form metadata string to be saved in the spectral tiff
    """
    if wls.dtype != 'float32':
        warnings.warn(
            f'Wavelength list dtype {wls.dtype} will be saved as float32. Precision may be lost.'
        )
        wls = wls.astype('float32')

    wavelengths = list(wls)
    metadata_bytes = str(metadata).encode('ascii', 'backslashreplace')

    stiff_tags = [
        (65000, 'f', len(wavelengths), wavelengths, True),
        (65111, 's', len(metadata_bytes), metadata_bytes, True)
    ]

    if len(wls) != spim.shape[2]:
        raise ValueError(
            f'Wavelength list length {len(wls)} does not match number of bands {spim.shape[2]}'
        )

    # RGB image must have three channels and dtype uint8
    if rgb is not None and rgb.ndim != 3:
        raise TypeError(f'RGB preview image must have three channels! (ndim = {rgb.ndim} != 3)')

    if rgb is not None and rgb.dtype != 'uint8':
        warnings.warn(f'RGB preview image is not a uint8 array (dtype: {rgb.dtype}).')
        if rgb.dtype == 'float':
            rgb = (rgb * (2**8 - 1)).astype('uint8')
        else:
            raise RuntimeError(f'How should {rgb.dtype} be handled here?')

    with TiffWriter(filename) as tiff:
        if rgb is not None:
            tiff.save(rgb)

        # Save the first page with tags
        spim_page = spim[:, :, 0]
        tiff.save(spim_page, extratags=stiff_tags)

        # Continue saving pages
        for i in range(1, spim.shape[2]):
            spim_page = spim[:, :, i]
            tiff.save(spim_page)


def read_mtiff(filename: str) -> Dict[str, np.ndarray]:
    """
    Read a mask bitmap tiff.
    
    Mask bitmap tiff contains multiple pages of bitmap masks. The mask label
    is stored in tag 65001 in each page as an ASCII string with unicode escapes.
    
    :param filename: filename of the mask tiff to read.
    :return: Dict[label: str, mask: ndarray], where
        label: the mask label
        mask: the boolean bitmap associated with the label
    """
    masks = dict()

    with TiffFile(filename) as tiff:
        for p in range(0, len(tiff.pages)):
            label_tag = tiff.pages[p].tags.get(TIFFTAG_MASK_LABEL)
            label = label_tag.value.encode('ascii').decode('unicode-escape')
            mask = tiff.asarray(key=p)
            masks[label] = mask > 0

    return masks


def write_mtiff(filename: str, masks: Dict[str, np.ndarray]) -> None:
    """
    Write a mask bitmap tiff.
    
    Mask bitmap tiff contains multiple pages of bitmap masks. The mask label
    is stored in tag 65001 in each page as an ASCII string with unicode escapes.
    
    :param filename: filename of the mask tiff to write to.
    :param masks: Dict[label: str, mask: ndarray], where
        label: the mask label
        mask: the boolean bitmap associated with the label
    """
    with TiffWriter(filename) as tiff:
        for label in masks:
            label_bytes = str(label).encode('ascii', 'backslashreplace')
            tiff.save(
                masks[label] > 0,
                photometric='MINISBLACK',
                contiguous=False,
                extratags=[(65001, 's', len(label_bytes), label_bytes, True)]
            )
