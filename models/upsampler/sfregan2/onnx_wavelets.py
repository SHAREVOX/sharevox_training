import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


class AFB1D(nn.Module):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """
    def forward(self, x, h0, h1, mode):
        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        lohi = lowlevel.afb1d(x, h0, h1, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1


class SFB1D(nn.Module):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        low (torch.Tensor): Lowpass to reconstruct of shape (N, C, L)
        high (torch.Tensor): Highpass to reconstruct of shape (N, C, L)
        g0: lowpass
        g1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*2, L')
    """
    def forward(self, low, high, g0, g1, mode):
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        return lowlevel.sfb1d(low, high, g0, g1, mode=mode, dim=3)[:, :, 0]


class DWT1DForward(nn.Module):
    """ Performs a 1d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]

        # Prepare the filters - this makes them into column filters
        filts = lowlevel.prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])
        self.register_buffer('h1', filts[1])
        self.J = J
        self.mode = mode
        self.afb1d = AFB1D()

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x

        # Do a multilevel transform
        for j in range(self.J):
            x0, x1 = self.afb1d(x0, self.h0, self.h1, self.mode)
            highs.append(x1)

        return x0, highs


class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # Prepare the filters
        filts = lowlevel.prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])
        self.mode = mode
        self.sfb1d = SFB1D()

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = self.sfb1d(x0, x1, self.g0, self.g1, self.mode)
        return x0
