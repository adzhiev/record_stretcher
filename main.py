import argparse
import os

from algorithm import RecordStretcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--record-path', type=str, help="Path to record",
        default=f'{os.path.dirname(os.path.abspath(__file__))}/samples/test_mono.wav'
    )
    parser.add_argument('-sr', '--stretch-ratio', type=float, nargs='+', help='Stretcher ratios', default=[2, 0.5])
    parser.add_argument('-ws', '--window-size', type=int, help='Window size', default=512)

    return parser.parse_args()


def main():
    args = parse_args()

    stretcher = RecordStretcher(args.record_path)

    # generate output paths
    filename, file_ext = os.path.splitext(args.record_path)
    output_files = [
        filename + f'_{i}' + file_ext
        for i in range(1, len(args.stretch_ratio) + 1)
    ]

    for stretch_ratio, output_file in zip(args.stretch_ratio, output_files):
        stretcher.pitch_shift(
            output_file,
            stretch_ratio=stretch_ratio,
            window_size=args.window_size
        )


if __name__ == '__main__':
    main()
