#pragma once
#include <random>
// helper functions
namespace Celesitial
{
  
    const double g_CCDphotons = 19100;    // exposure time 1s, d_length 1 mm^2
    const double g_dLY2Parsec = 0.30660139378795273107333862551539;   // lightyear -> parsec
    // absolute magnitude -> apparent magnitude
    template<class T>
    T absToAppMag(T absMag, T lyrs)
	{
		return (T) (absMag - 5 + 5 * log10(lyrs * g_dLY2Parsec));
	}

    // get photons from star magitude on CCD
    double getPhotonsFromApparentMagnitude(const double mv, const double d_len, const double t_in)
    {
        double n_pe = g_CCDphotons * 1 / std::pow(2.5, mv) * t_in * std::_Pi_val * (d_len / 2) * (d_len / 2) * 1e6;
        return n_pe;
    }
}