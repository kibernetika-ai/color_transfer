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


def process_video(input, input2, output, scale_fps=1):
    if not output:
        raise RuntimeError("--output must be supplied for video processing")

    is_rtmp = output.startswith('rtmp://')

    vc = cv2.VideoCapture(input)
    vc2 = cv2.VideoCapture(input2)
    if not vc.isOpened():
        raise RuntimeError("input file %s cannot be opened" % input)
    if not vc2.isOpened():
        raise RuntimeError("input file %s cannot be opened" % input2)

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
        writer = cv2.VideoWriter(output, fourcc, fps, frameSize=(width // 2, height))

    prepared_stats = None

    i = 1
    processed_i = 0

    switch_time = 30
    limit_time = 60

    while True:
        _, frame1 = vc.read()
        if frame1 is None:
            break
        _, frame2 = vc2.read()
        if frame2 is None:
            break
        i += 1
        # Skip frame
        if scale_fps != 0 and i % scale_fps != 0:
            continue

        frame1 = cv2.resize(frame1, (width // 2, height // 2))
        frame2 = cv2.resize(frame2, (width // 2, height // 2))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame2, "Original", (10, 30), font, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame2, "Original", (10, 30), font, 1, (200, 200, 200), 1, lineType=cv2.LINE_AA)

        if i / fps < switch_time:
            # Simple mode
            cv2.putText(frame1, "Corrupted", (10, 30), font, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame1, "Corrupted", (10, 30), font, 1, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            processed = np.vstack((frame1, frame2))
        else:
            # Apply color transfer from last frame (frame2)
            if prepared_stats is None:
                prepared_stats = color_transfer.prepare_for_transfer(frame2)

            colored = color_transfer.color_transfer_prepared(prepared_stats, frame1)
            cv2.putText(colored, "Reconstructed", (10, 30), font, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(colored, "Reconstructed", (10, 30), font, 1, (200, 200, 200), 1, lineType=cv2.LINE_AA)

            processed = np.vstack((colored, frame2))

        processed_i += 1
        if processed_i % 100 == 0:
            LOG.info("Processed %s frames." % processed_i)

        if i / fps >= limit_time:
            break

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

    process_video(args.input, args.palette, args.output, scale_fps=args.each_frame)
