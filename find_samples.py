import csv
import glob
import os
import random
from html import unescape

import numpy as np

#import soundfile as sf
import tqdm
from luigi_pipeline.audio_processing import MelSpectrogram

test_files = [
    "LP2bfIdrGi8",
    "390CWyYkH3M",
    "L7QZmnvrBd0",
    "TgLnt2PBczs",
    "jPcTv7EZWzo",
    "i2yQJ5VCHA0",
    "PHW_zx11ZP4",
    "XhPjeL3GQto",
    "1CW6NbZvR_w",
    "TLGaYNyOyd8",
    "l76zFLXW268",
    "WtOhZ--YeFY",
    "gdm-pZm8hoA",
    "li1aHjjqh3w",
    "41iHdxy7Kmg",
    "G0ESY__Ldhw",
    "4lBWtx4TuaY",
    "V5_vUE9FlXk",
    "-v8VWWi4gsI",
    "wY4u7F2ujNQ",
]

# for file_ in test_files:
#     file_name = os.path.basename(file_).replace(".npy", ".csv")
#     if os.path.isfile(
#         os.path.join(
#             "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/",
#             file_name,
#         )
#     ):
#         print(file_)


def fetch_audio():
    has_asr = [
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/LP2bfIdrGi8.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/390CWyYkH3M.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/L7QZmnvrBd0.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/TgLnt2PBczs.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/jPcTv7EZWzo.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/i2yQJ5VCHA0.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/PHW_zx11ZP4.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/XhPjeL3GQto.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/1CW6NbZvR_w.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/TLGaYNyOyd8.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/l76zFLXW268.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/li1aHjjqh3w.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/41iHdxy7Kmg.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/G0ESY__Ldhw.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/V5_vUE9FlXk.wav",
        "/projects/text2face/data2/DownloadYoutubeAudio/ffmpeg_bin_ffmpeg-fs_16000/-v8VWWi4gsI.wav",
    ]

    for file_ in has_asr:
        os.system(f"rsync pjjonell@130.237.67.85:{file_} has_asr")


## fetch
def fetch_asr():
    has_asr = [
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/LP2bfIdrGi8.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/390CWyYkH3M.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/L7QZmnvrBd0.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/TgLnt2PBczs.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/jPcTv7EZWzo.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/i2yQJ5VCHA0.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/PHW_zx11ZP4.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/XhPjeL3GQto.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/1CW6NbZvR_w.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/TLGaYNyOyd8.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/l76zFLXW268.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/li1aHjjqh3w.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/41iHdxy7Kmg.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/G0ESY__Ldhw.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/V5_vUE9FlXk.csv",
        "/projects/text2face/data2/DownloadYoutubeCaptions/caption_type_asr/-v8VWWi4gsI.csv",
    ]

    for file_ in has_asr:
        os.system(f"rsync pjjonell@130.237.67.85:{file_} has_asr")


def print_asr_times():
    contenders = []
    for file_ in glob.glob("has_asr/*.csv"):
        with open(file_) as f:
            csv_reader = csv.reader(f)
            past_end = 0.0
            next(csv_reader)
            for i, (word, start, duration) in enumerate(csv_reader):
                diff = float(start) - past_end
                if diff > 1200:
                    f = sf.SoundFile(file_.replace(".csv", ".wav"))

                    pose_path = os.path.basename(file_).replace(".csv", "")
                    start_frame = (float(start) / 1000.0) * 30
                    if (
                        (len(f) / f.samplerate) - float(start) / 1000.0 > 5
                        and find_best_pose_match(start_frame, pose_path) != 0
                    ):
                        contenders.append(
                            (file_, unescape(word), float(start) / 1000.0)
                        )
                past_end = float(start)
    print(random.sample(contenders, 10))


chosen_samples = [
    (
        "li1aHjjqh3w",
        "we're",
        105.49,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/li1aHjjqh3w000000--0.wav",
    ),
    (
        "li1aHjjqh3w",
        " everybody",
        10.37,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/li1aHjjqh3w000000--1.wav",
    ),
    (
        "TLGaYNyOyd8",
        " if",
        120.94,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/TLGaYNyOyd8000000--2.wav",
    ),
    (
        "jPcTv7EZWzo",
        " and",
        7.1,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/jPcTv7EZWzo000000--3.wav",
    ),
    (
        "L7QZmnvrBd0",
        " everybody",
        9.95,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/L7QZmnvrBd0000000--4.wav",
    ),
    # (
    #     "LP2bfIdrGi8",
    #     " 218",
    #     22.97,
    #     "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/LP2bfIdrGi8000000--5.wav",
    # ),
    (
        "1CW6NbZvR_w",
        " every",
        102.689,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/1CW6NbZvR_w000000--5.wav",
    ),
    (
        "LP2bfIdrGi8",
        " week",
        5.519,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/LP2bfIdrGi8000000--6.wav",
    ),
    (
        "390CWyYkH3M",
        " years",
        5.86,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/390CWyYkH3M000000--7.wav",
    ),
    (
        "XhPjeL3GQto",
        " the",
        5.93,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/XhPjeL3GQto000000--8.wav",
    ),
    (
        "41iHdxy7Kmg",
        " small",
        138.549,
        "https://s3-eu-west-1.amazonaws.com/furhat-users/c938c34e-4695-4fdb-9acf-33a6d85d7727/audio/41iHdxy7Kmg000000--9.wav",
    ),
]


def create_trimmed_file():
    for i, sample in enumerate(chosen_samples):
        in_name = f"has_asr/{sample[0]}.wav"
        out_name = f"the_deal/{sample[0]}000000--{i}.wav"
        if not os.path.isfile(out_name):
            os.system(f"sox {in_name} {out_name} trim {sample[2]} 5")


def fetch_pose():
    for file_ in tqdm.tqdm(test_files):
        file_name = f"/projects/text2face/data2/PostProcessOpenface/will_interpolate_True-min_confidence_0.7-min_value_0.001-min_mean_0.1-max_mean_1.0-min_std_0.001-with_blinks_True/{file_}.npy"
        os.system(f"rsync pjjonell@130.237.67.85:{file_name} has_asr")


def find_best_pose_match(frame, yt_id):
    a = np.load(f"has_asr/{yt_id}.npy")

    best_pick = 0
    prev_len = 0
    for start_frame, frames in a.item().items():
        if frame < start_frame:
            end_frame = best_pick + prev_len
            if frame + 160 > end_frame:
                best_pick = 0
            break

        if frame >= start_frame:
            best_pick = start_frame
            prev_len = frames.shape[0]

    return best_pick


def prepare_ground_truth():
    for i, sample in enumerate(chosen_samples):
        get_frame = round(sample[2] * 30)
        a = np.load(f"has_asr/{sample[0]}.npy")
        best_pick = find_best_pose_match(get_frame, sample[0])

        faces = a.item()[best_pick][
            (get_frame - best_pick) : (get_frame - best_pick) + 160
        ]

        np.save(f"ground_truth/{sample[0]}000000--{i}.npy", faces)


def get_random_places():
    print(random.sample(test_files, 10))


random_places_files = [
    "-v8VWWi4gsI",
    "V5_vUE9FlXk",
    "1CW6NbZvR_w",
    "WtOhZ--YeFY",
    "LP2bfIdrGi8",
    "PHW_zx11ZP4",
    "4lBWtx4TuaY",
    "390CWyYkH3M",
    "l76zFLXW268",
    "i2yQJ5VCHA0",
]


def get_random_places_keys():
    random_keys = []
    for place in random_places_files:
        b = np.load(f"has_asr/{place}.npy")
        filtered_list = [x for x, y in b.item().items() if len(y) > 160]
        random_keys.append((place, random.choice(filtered_list)))
    print(random_keys)


random_keys_with_places = [
    ("-v8VWWi4gsI", 158),
    ("V5_vUE9FlXk", 157),
    ("1CW6NbZvR_w", 303),
    ("WtOhZ--YeFY", 935),
    ("LP2bfIdrGi8", 3486),
    ("PHW_zx11ZP4", 281),
    ("4lBWtx4TuaY", 158),
    ("390CWyYkH3M", 156),
    ("l76zFLXW268", 157),
    ("i2yQJ5VCHA0", 142),
]


def get_random_starting_place_within_keys():
    random_pos = []
    for place, key in random_keys_with_places:
        c = np.load(f"has_asr/{place}.npy").item()
        max_starting_pos = len(c[key]) - 160
        random_pos.append((place, key, random.randint(0, max_starting_pos)))
    print(random_pos)


random_pos_within_key_within_place = [
    ("-v8VWWi4gsI", 158, 2407),
    ("V5_vUE9FlXk", 157, 605),
    ("1CW6NbZvR_w", 303, 2017),
    ("WtOhZ--YeFY", 935, 114),
    ("LP2bfIdrGi8", 3486, 801),
    ("PHW_zx11ZP4", 281, 7857),
    ("4lBWtx4TuaY", 158, 2793),
    ("390CWyYkH3M", 156, 3171),
    ("l76zFLXW268", 157, 1899),
    ("i2yQJ5VCHA0", 142, 7665),
]


def prepare_random_pose():
    for (i, sample), (place, key, starting_position) in zip(
        enumerate(chosen_samples), random_pos_within_key_within_place
    ):
        d = np.load(f"has_asr/{place}.npy").item()[key]
        np.save(
            f"random_place/{sample[0]}000000--{i}.npy",
            d[starting_position : starting_position + 160],
        )


def prepare_model():
    data_dir = "/projects/text2face/data2"
    for i, sample in enumerate(chosen_samples):
        mel_spec = np.load(
            MelSpectrogram(yt_video_id=sample[0], data_dir=data_dir, hop_duration=0.033)
            .output()
            .path
        )

        #print(sample[0], mel_spec.shape)

        audio_data = mel_spec[round(sample[2] * 30) : round(sample[2] * 30) + 160]
        print(audio_data.shape)
        np.save(f'audio_features/{sample[0]}000000--{i}.npy', audio_data)
import torch
from glow.config import JsonConfig
from glow.builder import build


def run_model():
    hparams_path = 'hparams/speech2face_gpu.json'
    hparams = JsonConfig(hparams_path)
    model = build(hparams, False)['graph']
    _ = model.eval()

    for i, sample in enumerate(chosen_samples):

        e = np.load(f"audio_features/{sample[0]}000000--{i}.npy")
        dbb = torch.from_numpy(e.T).unsqueeze(0).float().expand(100, -1, -1).to('cuda:3')
        new_old_img = model(audio_features=dbb, reverse=True)

        print(new_old_img.shape, new_old_img[0].shape, new_old_img[0].squeeze(-1).shape, new_old_img[0].squeeze(-1).cpu().numpy().T.shape)
        np.save(f'our_model_results/{sample[0]}000000--{i}.npy', new_old_img[0].squeeze(-1).cpu().numpy().T)
run_model()
#prepare_model()

# prepare_random_pose()
# get_random_starting_place_within_keys()
# get_random_places_keys()
# get_random_places()


# print_asr_times()
# prepare_ground_truth()
# create_trimmed_file()
