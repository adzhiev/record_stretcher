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

        time_index = 0

        for i in range(number_frames):
            result[time_index:time_index + size_frames] += frames_matrix[i, :]
            time_index += hop

        return result

    def pitch_shift(self, output_path, window_size=512, stretch_ratio=2):
        # Recommended space between windows
        hop = window_size // 4
        hop_out = round(stretch_ratio * hop)

        # Hanning window
        wn = np.hanning(window_size)

        # add zeros at the beginning
        x = np.concatenate((np.zeros(3 * hop), self.samples))

        ### INITIALIZATION ###

        # Create a frame matrix for the current input
        y, number_frames_input = RecordStretcher.create_frames(x, hop, window_size)

        # Create a frame matrix to receive processed frames
        number_frames_output = number_frames_input
        output_y = np.zeros((number_frames_output, window_size))

        # Initialize synthesis phase
        synthesis_phase = np.zeros(window_size)

        # Initialize previous frame analysis phase
        previous_analysis_phase = np.zeros(window_size)

        omega_bin = self.framerate * np.arange(window_size) / window_size

        for i in range(number_frames_input):
            ### ANALYSIS ###
            # Get current frame to be processed
            current_frame = y[i, :]

            # Window the frame
            current_frame_windowed = current_frame * wn

            # Get the FFT
            current_frame_windowed_fft = np.fft.fft(current_frame_windowed)

            # Get the magnitude
            mag_frame = np.abs(current_frame_windowed_fft)

            # Get the angle
            analysis_phase = np.angle(current_frame_windowed_fft)

            ### PROCESSING ###
            # Get the phase difference
            delta_analysis_phase = analysis_phase - previous_analysis_phase
            previous_analysis_phase = analysis_phase

            # Remove the expected phase difference
            delta_omega = self.framerate * delta_analysis_phase / hop - omega_bin

            # Map to -pi/pi range
            delta_omega_wrapped = np.mod(delta_omega + np.pi, 2 * np.pi) - np.pi

            # Get the true frequency
            omega_true = omega_bin + delta_omega_wrapped

            # Get the final phase
            synthesis_phase += hop_out * omega_true / self.framerate

            ### SYNTHESIS ###
            # Produce output frame
            output_frame = np.real(np.fft.ifft(mag_frame * np.exp(1j * synthesis_phase)))

            # Save frame that has been processed
            output_y[i, :] = output_frame * wn
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
