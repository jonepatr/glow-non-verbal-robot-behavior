import glob
from collections import defaultdict
from os.path import basename

import numpy as np
from torch.utils.data import Dataset

from luigi_pipeline.audio_processing import MelSpectrogram
from luigi_pipeline.youtube_downloader import (DownloadYoutubeAudio,
                                               DownloadYoutubeVideo)
from tqdm import tqdm


class Speech2FaceDataset(Dataset):
    def __init__(
        self, dataset_files, data_dir=None, total_frames=None, audio_feature_type=None
    ):

        self.data = []

        self.face_data = []
        self.audio_features_data = []

        for n, openface_file_path in enumerate(
            tqdm(dataset_files, desc="Loading dataset. Sit back and relax")
        ):
            filepath = basename(openface_file_path).replace(".npy", "")
            if audio_feature_type == "spectrogram":
                ms = MelSpectrogram(
                    data_dir=data_dir, yt_video_id=filepath, hop_duration=0.033
                )
                audio_feature_data = np.load(ms.output().path).astype(np.float32)
            audio_path = (
                DownloadYoutubeAudio(data_dir=data_dir, yt_video_id=filepath)
                .output()
                .path
            )
            video_path = (
                DownloadYoutubeVideo(data_dir=data_dir, yt_video_id=filepath)
                .output()
                .path
            )

            self.audio_features_data.append(audio_feature_data)
            self.face_data.append(defaultdict(list))

            openpose_data = np.load(openpose_file_path, allow_pickle=True)
            for frame, all_faces in openpose_data.item().items():
                all_faces_len = len(all_faces)
                face_d = []
                if all_faces_len > total_frames:
                    for i in range(all_faces_len):
                        first_frame = frame + i

                        # append is a hack so that instead of 7 we have 8 values
                        face_d.append(
                            np.append(all_faces[i], np.random.uniform())
                            .astype(np.float32)
                            .reshape(8, 1)
                        )

                        if first_frame + total_frames < all_faces_len:

                            face = (len(self.face_data) - 1, frame, i, i + total_frames)
                            audio_features = (
                                len(self.audio_features_data) - 1,
                                first_frame,
                                first_frame + total_frames,
                            )

                            if (
                                all_faces_len > i + total_frames
                                and len(self.audio_features_data[-1])
                                > first_frame + total_frames
                            ):
                                self.data.append(
                                    (
                                        face,
                                        audio_features,
                                        first_frame,
                                        audio_path,
                                        video_path,
                                    )
                                )
                    self.face_data[-1][frame] = np.array(face_d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        face, audio_features, first_frame, audio_path, video_path = self.data[index]

        face_index, frame, face_start, face_stop = face
        audio_feature_index, audio_feature_start, audio_feature_stop = audio_features

        return {
            "x": self.face_data[face_index][frame][face_start:face_stop].transpose(
                1, 0, 2
            ),
            "audio_features": self.audio_features_data[audio_feature_index][
                audio_feature_start:audio_feature_stop
            ].T,
            "first_frame": first_frame,
            "audio_path": audio_path,
            "video_path": video_path,
            "y": 1,
        }


if __name__ == "__main__":
    import cv2

    celeba = CelebADataset("/home/chaiyujin/Downloads/Dataset/CelebA")
    d = celeba[0]
    print(d["x"].size())
    img = d["x"].permute(1, 2, 0).contiguous().numpy()
    print(np.min(img), np.max(img))
    cv2.imshow("img", img)
    cv2.waitKey()
    print(d["x"].size())
    img = d["x"].permute(1, 2, 0).contiguous().numpy()
    print(np.min(img), np.max(img))
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.imshow("img", img)
    cv2.waitKey()
