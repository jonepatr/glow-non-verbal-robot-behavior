import glob
from collections import defaultdict
from os.path import basename

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from luigi_pipeline.audio_processing import MelSpectrogram
from luigi_pipeline.post_process_openface import PostProcessOpenface
from luigi_pipeline.youtube_downloader import DownloadYoutubeAudio
from tqdm import tqdm


class Speech2FaceDataset(Dataset):
    def __init__(
        self, data_dir=None, total_frames=None, small=None, audio_feature_type=None
    ):
        dataset_files = list(
            glob.glob(
                PostProcessOpenface(data_dir=data_dir, yt_video_id="*").output().path
            )
        )
        if small:
            dataset_files = dataset_files[:2]
        self.data = []

        self.face_data = []
        self.audio_features_data = []

        for n, openface_file_path in enumerate(
            tqdm(dataset_files, desc="Loading dataset. Sit back and relax")
        ):
            filepath = basename(openface_file_path).replace(".csv", "")
            if audio_feature_type == "spectrogram":
                ms = MelSpectrogram(
                    data_dir=data_dir, yt_video_id=filepath, hop_duration=0.01
                )
                audio_feature_data = np.load(ms.output().path).astype(np.float32)
            audio = DownloadYoutubeAudio(data_dir=data_dir, yt_video_id=filepath)

            self.audio_features_data.append(audio_feature_data)
            self.face_data.append(defaultdict(list))

            openface_data = pd.read_csv(openface_file_path)

            for frame, all_faces in openface_data.groupby("group"):

                all_faces_len = len(all_faces)
                face_d = []
                if all_faces_len > total_frames:
                    for i in range(all_faces_len):
                        first_frame = frame + i

                        face_d.append(
                            all_faces.iloc[i][
                                [
                                    "AU01_r",
                                    "AU02_r",
                                    "AU04_r",
                                    "pose_Rx",
                                    "pose_Ry",
                                    "pose_Rz",
                                ]
                            ]
                            .to_numpy()
                            .astype(np.float32)
                            .reshape(6, 1)
                        )

                        if first_frame + total_frames < all_faces_len:

                            face = (len(self.face_data) - 1, frame, i, i + total_frames)

                            first_audio_feature_frame = int(
                                round(first_frame * (1.0 / 30.0)) // ms.hop_duration
                            )

                            audio_feature_frames = int(
                                round((total_frames / 30.0) / ms.hop_duration)
                            )

                            last_audio_feature_frames = (
                                first_audio_feature_frame + audio_feature_frames
                            )

                            audio_features = (
                                len(self.audio_features_data) - 1,
                                first_audio_feature_frame,
                                last_audio_feature_frames,
                            )

                            if (
                                all_faces_len > i + total_frames
                                and last_audio_feature_frames
                                < len(self.audio_features_data[-1])
                            ):
                                self.data.append(
                                    (
                                        face,
                                        audio_features,
                                        first_frame,
                                        audio.output().path,
                                    )
                                )
                    self.face_data[-1][frame] = np.array(face_d)

    def prepare_pca(self, dataset_files, pca_dimensions):
        pca = PCA(pca_dimensions)

        pca_data = []
        for n, openpose_file_path in enumerate(
            tqdm(dataset_files, desc="Preparing PCA")
        ):
            openpose_data = np.load(openpose_file_path)

            # do PCA
            for frame, all_faces in openpose_data.item().items():
                all_faces = all_faces.astype(np.float32)
                for face in all_faces:
                    pca_data.append(face.reshape(140))
        data = np.array(pca_data)
        self.pca = pca.fit(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        face, audio_features, first_frame, audio_path, = self.data[index]

        face_index, frame, face_start, face_stop = face
        audio_feature_index, audio_feature_start, audio_feature_stop = audio_features

        return {
            "x": self.face_data[face_index][frame][face_start:face_stop].transpose(
                1, 0, 2
            ),
            "audio_features": self.audio_features_data[audio_feature_index][
                audio_feature_start:audio_feature_stop
            ],
            "first_frame": first_frame,
            "audio_path": audio_path,
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
