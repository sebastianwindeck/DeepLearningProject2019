downloaded from: https://github.com/jsleep/wav2mid
on December 26, 2018

Installation: Needs quite a number of other packages. I just went through and 
imported the dependencies that i didn't have already. Or just let it run and
import all the packages that the compiles complains that they are missing ;-)

Some packages are not available in anaconda navigator. I found them here:
- pyfftw (dependency of madmom): https://github.com/pyFFTW/pyFFTW/
- madmom: https://github.com/CPJKU/madmom, installation as decribed in the 
  readme therein.
- pretty_midi: https://github.com/craffel/pretty-midi 
  -> needs to be cited: Colin Raffel and Daniel P. W. Ellis. Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi. In Proceedings of the 15th International Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.
  >pip install pretty_midi
- librosa: https://librosa.github.io/librosa/install.html
  >pip install librosa