import argparse
import logging
import os

import cv2
import numpy as np

import color_transfer


logging.basicConfig(
    format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
    level='INFO'
)
LOG = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--palette", required=True,
        help="Path to the palette image or video"
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="Path to input image / video"
    )
    ap.add_argument(
        "-n", "--frame-num",
        help=(
            "Frame number from target video which will be the source pallet. "
            "Negative number means count from the end"
        ),
        type=int,
        default=10,
        metavar='<int>',
    )
    ap.add_argument(
        "--output", "-o",
        help="Path to the target file / stream"
    )
    ap.add_argument(
        "-c", "--clip", type=str2bool, default='t',
        help=(
            "Should np.clip scale L*a*b* values before final conversion to BGR? "
            "Approptiate min-max scaling used if False."
        )
    )
    ap.add_argument(
        "-p", "--preserve", type=str2bool, default='t',
        help="Should color transfer strictly follow methodology layed out in original paper?"
    )
    ap.add_argument("--show", action='store_true')
    return ap.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_image(title, image, height=300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = height / float(image.shape[0])
    dim = (int(image.shape[1] * r), height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # show the resized image
    cv2.imshow(title, resized)


def is_image(filename):
    base = os.path.basename(filename)
    _, ext = os.path.splitext(base)
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']


def process_video(input, palette, output):
    if not output:
        raise RuntimeError("--output must be supplied for video processing")

    is_rtmp = output.startswith('rtmp://')

    vc = cv2.VideoCapture(input)
    if not vc.isOpened():
        raise RuntimeError("input file %s cannot be opened" % input)

    fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
    fps = vc.get(cv2.CAP_PROP_FPS)
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))

    if is_rtmp:
        # Open ffmpeg for streaming
        pass
    else:
        # Open file for writing
        writer = cv2.VideoWriter(output, fourcc, fps, frameSize=(width, height))

    i = 0
    while True:
        _, frame = vc.read()
        if frame is None:
            break
        i += 1

        processed = color_transfer.color_transfer(palette, frame)
        if i % 100 == 0:
            LOG.info("Processed %s frames." % i)

        if is_rtmp:
            # Send processed frame to ffmpeg
            pass
        else:
            # write frame
            writer.write(processed)

    vc.release()
    if is_rtmp:
        # Close streaming
        pass
    else:
        writer.release()


if __name__ == '__main__':
    args = parse_args()
    # load the palette
    if is_image(args.palette):
        palette = cv2.imread(args.palette)
    else:
        vc = cv2.VideoCapture(args.palette)
        total_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = args.frame_num
        if frame_num < 0:
            frame_num = total_count - frame_num

        print("Extract %s frame from video..." % frame_num)
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        _, palette = vc.read()

    if is_image(args.input):
        input_img = cv2.imread(args.input)
        # transfer the color distribution from the source image
        # to the target image
        transfer = color_transfer.color_transfer(palette, input_img, clip=args.clip, preserve_paper=args.preserve)

        # check to see if the output image should be saved
        if args.output is not None:
            cv2.imwrite(args.output, transfer)

        if args.show:
            # show the images and wait for a key press
            show_image("Source", palette)
            show_image("Target", input_img)
            show_image("Transfer", transfer)
            cv2.waitKey(0)
    else:
        process_video(args.input, palette, args.output)
