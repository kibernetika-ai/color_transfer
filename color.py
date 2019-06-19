import argparse
import logging
import os
import signal
import subprocess

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
        "--each-frame",
        help="Process only each N frame. Reduces fps of video and stream accordingly.",
        type=int,
        default=1,
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


def start_ffmpeg(width, height, fps, rtmp_url):
    ffmpeg_binary = 'ffmpeg'
    command = []
    command.extend([
        ffmpeg_binary,
        '-loglevel', 'verbose',
        '-y',  # overwrite previous file/stream
        # '-re',    # native frame-rate
        '-analyzeduration', '1',
        '-f', 'rawvideo',
        '-r', '%d' % fps,  # set a fixed frame rate
        '-vcodec', 'rawvideo',
        # size of one frame
        '-s', '%dx%d' % (width, height),
        '-pix_fmt', 'rgb24',  # The input are raw bytes
        '-thread_queue_size', '1024',
        '-i', '/tmp/videopipe0',  # The input comes from a pipe
        '-an',            # Tells FFMPEG not to expect any audio
    ])
    command.extend([
        # VIDEO CODEC PARAMETERS
        '-vcodec', 'libx264',
        '-r', '%d' % fps,
        # AUDIO CODEC PARAMETERS
        '-acodec', 'libmp3lame', '-ar', '44100', '-b:a', '160k',
        '-ac', '1',

        # NUMBER OF THREADS
        '-threads', '2',

        # STREAM TO RTMP
        '-f', 'flv', '%s' % rtmp_url
    ])

    # devnullpipe = open("/dev/null", "w")  # Throw away stream
    devnullpipe = None
    ffmpeg_process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=devnullpipe,
        stdout=devnullpipe
    )
    return ffmpeg_process


def process_video(input, palette, output, scale_fps=1):
    if not output:
        raise RuntimeError("--output must be supplied for video processing")

    is_rtmp = output.startswith('rtmp://')

    vc = cv2.VideoCapture(input)
    if not vc.isOpened():
        raise RuntimeError("input file %s cannot be opened" % input)

    fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
    fps = vc.get(cv2.CAP_PROP_FPS)
    if scale_fps != 0:
        fps = fps / scale_fps
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))

    if is_rtmp:
        # Open ffmpeg for streaming
        ffmpeg = start_ffmpeg(width, height, fps, output)
        # Create pipe
        pipe = '/tmp/videopipe0'
        if not os.path.exists(pipe):
            os.mkfifo(pipe)
        video_pipe = os.open(pipe, os.O_WRONLY)
    else:
        # Open file for writing
        writer = cv2.VideoWriter(output, fourcc, fps, frameSize=(width, height))

    prepared_stats = color_transfer.prepare_for_transfer(palette)

    i = 1
    processed_i = 0
    while True:
        _, frame = vc.read()
        if frame is None:
            break
        i += 1
        # Skip frame
        if scale_fps != 0 and i % scale_fps != 0:
            continue

        processed = color_transfer.color_transfer_prepared(prepared_stats, frame)
        processed_i += 1
        if processed_i % 100 == 0:
            LOG.info("Processed %s frames." % processed_i)

        if is_rtmp:
            # Convert to RGB
            processed = processed[:, :, ::-1]
            # Send processed frame to ffmpeg
            try:
                os.write(video_pipe, processed.tostring())
            except OSError as e:
                LOG.error(e)
                break
        else:
            # write frame
            writer.write(processed)

    vc.release()
    if is_rtmp:
        # Close streaming
        ffmpeg.poll()
        ffmpeg.send_signal(signal.SIGINT)
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
        process_video(args.input, palette, args.output, scale_fps=args.each_frame)
