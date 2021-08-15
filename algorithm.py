import os

import wave
import numpy as np


class RecordStretcher:
    WIDTH_TYPES = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }

    def __init__(self, path):
        filename, file_ext = os.path.splitext(path)
        assert file_ext == '.wav'
        self.path = path

        self.read_wav_file(path)

    def read_wav_file(self, path):
        wav = wave.open(path, 'rb')

        # we'll process 1-channel records
        assert wav.getnchannels() == 1

        self.framerate = wav.getframerate()
        self.nframes = wav.getnframes()
        self.sampwidth = wav.getsampwidth()
        self.dtype = self.WIDTH_TYPES[self.sampwidth]

        content = wav.readframes(self.nframes)

        self.samples = np.fromstring(content, dtype=self.dtype)

        wav.close()

    @staticmethod
    def create_frames(x, hop, window_size):
        # find the max number of slices that can be obtained
        number_slices = (x.size - window_size) // hop + 1

        # Truncate if needed to get only a integer number of hop
        lower_bound = (number_slices - 1) * hop + window_size
        x = x[0:lower_bound]

        # Create a matrix with time slices
        vector_frames = np.zeros((number_slices, window_size))

        # Fill the matrix
        for i in range(number_slices):
            vector_frames[i, :] = x[i * hop:i * hop + window_size]

        return vector_frames, number_slices

    @staticmethod
    def fusion_frames(frames_matrix, hop):
        number_frames, size_frames = frames_matrix.shape

        # Define an empty vector to receive result
        result = np.zeros((number_frames - 1) * hop + size_frames)

        for i in range(number_frames - 1):
            result[i * hop:(i + 1) * hop] = frames_matrix[i, :hop]

        result[(i + 1) * hop:] = frames_matrix[number_frames - 1, :]

        return result

    def pitch_shift(self, output_path, window_size=512, stretch_ratio=2):
        # Recommended space between windows
        hop = window_size // 4
        hop_out = round(stretch_ratio * hop)

        # Hanning window
        wn = np.hanning(window_size)

        # add zeros at the end
        x = np.concatenate((self.samples, np.zeros(3 * hop)))

        ### INITIALIZATION ###

        # Create a frame matrix for the current input
        y, number_frames_input = RecordStretcher.create_frames(x, hop, window_size)

        # Create a frame matrix to receive processed frames
        number_frames_output = number_frames_input
        output_y = np.zeros((number_frames_output, window_size))

        # Initialize cumulative phase
        phase_cumulative = np.zeros(window_size)

        # Initialize previous frame phase
        previous_phase = np.zeros(window_size)

        expected_phase = hop * 2 * np.pi * np.linspace(0, 1, window_size)

        for i in range(number_frames_input):
            ### ANALYSIS ###
            # Get current frame to be processed
            current_frame = y[i, :]

            # Window the frame
            current_frame_windowed = current_frame * wn / np.sqrt(((window_size / hop) / 2))

            # Get the FFT
            current_frame_windowed_fft = np.fft.fft(current_frame_windowed)

            # Get the magnitude
            mag_frame = np.abs(current_frame_windowed_fft)

            # Get the angle
            phase_frame = np.angle(current_frame_windowed_fft)

            ### PROCESSING ###
            # Get the phase difference
            delta_phi = phase_frame - previous_phase
            previous_phase = phase_frame

            # Remove the expected phase difference
            delta_phi_prime = delta_phi - expected_phase

            # Map to -pi/pi range
            delta_phi_prime_mod = np.mod(delta_phi_prime + np.pi, 2 * np.pi) - np.pi

            # Get the true frequency
            true_freq = 2 * np.pi * np.linspace(0, 1, window_size) + delta_phi_prime_mod / hop

            # Get the final phase
            phase_cumulative += hop_out * true_freq

            ### SYNTHESIS ###
            # Get the magnitude
            output_mag = mag_frame

            # Produce output frame
            output_frame = np.real(np.fft.ifft(output_mag * np.exp(1j * phase_cumulative)))

            # Save frame that has been processed
            output_y[i, :] = output_frame * wn / np.sqrt(((window_size / hop_out) / 2))

        # Collect frames
        stretched_record = RecordStretcher.fusion_frames(output_y, hop_out)
        stretched_record = stretched_record.astype(self.dtype)

        # Write record
        wout = wave.open(output_path, 'wb')
        wout.setnchannels(1)
        wout.setsampwidth(self.sampwidth)
        wout.setframerate(self.framerate)
        wout.setnframes(len(stretched_record))
        wout.writeframes(stretched_record.tostring())
        wout.close()
