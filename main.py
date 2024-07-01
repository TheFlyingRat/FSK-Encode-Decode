"""
Frequency Shift Keying in Python
Copyright (c) 2023 Joey Manani
License found at https://cdn.theflyingrat.com/LICENSE
Permission is hereby granted to modify, copy and use this app as per the license WITH credit to me and a link to my license
"""


import math
import wave
import struct
import numpy as np
from scipy.fft import rfft, rfftfreq


class FrequencyShiftKeying:
    """Generate a wave file containing FSK encoded data."""
    def __init__(self, duration: int, sample_rate: int):
        self._high_frequency = 2000
        self._low_frequency = 1000
        self._amplitude = 32767  # 16 bit unsigned wave maximum amplitude
        self._duration = duration  # Duration in samples per tone
        self._sample_rate = sample_rate
        self._global_signal = []  # Samples where wave file is stored


    # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
    def _create_sine_wave(self, frequency: int) -> None:
        """Create a sine wave of given frequency and append it to global_signal

        :param frequency: Frequency in hertz
        """
        signal = []
        num_samples = self._duration

        for i in range(num_samples):
            value = self._amplitude * math.sin(2 * math.pi * frequency * i / self._sample_rate)
            signal.append(value)

        self._global_signal.extend(signal)


    def encode(self, binary_bits: list) -> None:
        """Encode the data given as...

        :param binary_bits: an iterable of binary bits
        :raises ValueError: if any binary digit is not 0 or 1 (as integers)
        """
        for bit in binary_bits:
            if bit == 1:
                self._create_sine_wave(self._high_frequency)
            elif bit == 0:
                self._create_sine_wave(self._low_frequency)
            else:
                raise ValueError("Invalid bit value")


    def decode(self) -> list:
        """Decode imported wave signal

        :raises ValueError: if a bad frequency is detected. Duration of tone may be too short 
        :return: Decoded data as a list of bits
        """
        binary_data = []
        samples_per_tone = self._duration
        num_samples = len(self._global_signal) // 2  # Dividing by 2 to work with stereo signal
        num_tones = num_samples // samples_per_tone # FSK tones (or bits) in the file

        if num_samples == 0 or num_tones == 0:
            raise ValueError("Could not find any samples to decode!")

        for i in range(num_tones):
            start_idx = i * samples_per_tone * 2  # Multiply by 2 for stereo signal
            end_idx = start_idx + samples_per_tone * 2
            tone_samples = self._global_signal[start_idx:end_idx]

            # Convert tone_samples to mono and calculate frequency
            tone_samples_mono = np.frombuffer(tone_samples, dtype=np.int16)
            freq = self._get_frequency(tone_samples_mono)

            # Test if calculated frequency is 10% of expected frequency
            # https://www.w3schools.com/python/ref_math_isclose.asp
            if math.isclose(freq, self._low_frequency, rel_tol=0.1):
                binary_data.append(0)
            elif math.isclose(freq, self._high_frequency, rel_tol=0.1):
                binary_data.append(1)
            else:
                raise ValueError("Invalid frequency found while trying to decode global signal")

        return binary_data


    # https://stackoverflow.com/questions/54612204/trying-to-get-the-frequencies-of-a-wav-file-in-python
    # https://realpython.com/python-scipy-fft/
    def _get_frequency(self, samples: np.ndarray) -> float:
        """Gets a frequency from ndarray samples

        :param samples: Numpy ndarray representation of wave samples
        :return: Frequency in hertz
        """

        # No clue how this works to be honest. I don't like math.
        # I think it calculates how many peaks of the samples are within 1/sample_rate samples?
        # I could be entirely wrong
        number_of_samples = len(samples)
        yf = rfft(samples)
        xf = rfftfreq(number_of_samples, 1 / self._sample_rate)
        idx = np.argmax(np.abs(yf))
        frequency = xf[idx]
        return frequency


    def save_to_wave_file(self, filename: str) -> None:
        """Save global signal to a wave file

        :param filename: File name to save the global signal under
        """
        output_wave = wave.open(filename, 'w')
        output_wave.setparams((2,
                               2,
                               self._sample_rate,
                               len(self._global_signal),
                               'NONE',
                               'not compressed'
                               ))

        for value in self._global_signal:
            packed_value = struct.pack('h', int(value))
            output_wave.writeframes(packed_value)

        output_wave.close()


    # https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
    def load_from_wave_file(self, filename: str) -> list:
        """Load a wave file from disk and return decode it 

        :param filename: The filename to open and load to the global signal
        """
        input_wave = wave.open(filename, 'r')
        signal = input_wave.readframes(-1)
        input_wave.close()

        self._global_signal = signal




if __name__ == "__main__":
    try:
        print("Example of using FSK transcoder\n\n")
        fsk = FrequencyShiftKeying(duration=250, sample_rate=44100)

        DATA = "This is a test of FSK encoding".encode("utf-8") # Encode the characters to bytes
        raw_bits = []
        for byte in DATA:
            raw_bits.extend(int(bit) for bit in bin(byte)[2:].zfill(8)) # Create bits from bytes

        fsk.encode(raw_bits) # Encode the bits into wave
        fsk.save_to_wave_file("save.wav") # Save the wave file
        fsk.load_from_wave_file("save.wav") # Open the wave file
        DECODED_BITS = fsk.decode() # Decode the wave file

        BIT_STRING = ''.join(map(str, DECODED_BITS)) # [0,1,1,0] --> "0110"
        BIT_CHUNKS = [BIT_STRING[i:i+8] for i in range(0, len(BIT_STRING), 8)] # 8 bits to byte
        ASCII_TEXT = ''.join(chr(int(chunk, 2)) for chunk in BIT_CHUNKS) # "01000001" --> "A"

        print("Raw data:", DATA.decode())
        print("Encoded bits:", BIT_STRING)
        print("Decoded text:", ASCII_TEXT) # ASCII decoded text
        print("\n\nThere is a file located named 'save.wav' which contains the FSK encoded data")

    except Exception as e:
        print(f"An error occurred: {e}")
