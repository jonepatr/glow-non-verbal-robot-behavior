import glob
import os
import re
from os.path import basename

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from luigi_pipeline.audio_processing import Autocorrelation, MelSpectrogram
from luigi_pipeline.post_process_openpose import PostProcessOpenpose
from luigi_pipeline.youtube_downloader import DownloadYoutubeAudio
from sklearn.decomposition import PCA
from tqdm import tqdm

IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "list_attr_celeba.txt"


def _is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def _find_images_and_annotation(root_dir):
    images = {}
    attr = None
    assert os.path.exists(root_dir), "{} not exists".format(root_dir)
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in sorted(fnames):
            if _is_image(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lower() == ATTR_ANNO:
                attr = os.path.join(root, fname)

    assert attr is not None, "Failed to find `list_attr_celeba.txt`"

    # begin to parse all image
    print("Begin to parse all image attrs")
    final = []
    with open(attr, "r") as fin:
        image_total = 0
        attrs = []
        for i_line, line in enumerate(fin):
            line = line.strip()
            if i_line == 0:
                image_total = int(line)
            elif i_line == 1:
                attrs = line.split(" ")
            else:
                line = re.sub("[ ]+", " ", line)
                line = line.split(" ")
                fname = os.path.splitext(line[0])[0]
                onehot = [int(int(d) > 0) for d in line[1:]]
                assert len(onehot) == len(attrs), "{} only has {} attrs < {}".format(
                    fname, len(onehot), len(attrs)
                )
                final.append({"path": images[fname], "attr": onehot})
    print("Find {} images, with {} attrs".format(len(final), len(attrs)))
    return final, attrs


class CelebADataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=transforms.Compose(
            [transforms.CenterCrop(160), transforms.Resize(32), transforms.ToTensor()]
        ),
    ):
        super().__init__()
        dicts, attrs = _find_images_and_annotation(root_dir)
        self.data = dicts
        self.attrs = attrs
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        path = data["path"]
        attr = data["attr"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"x": image, "y_onehot": np.asarray(attr, dtype=np.float32)}

    def __len__(self):
        return len(self.data)


class Speech2FaceDataset(Dataset):
    def __init__(
        self, data_dir=None, total_frames=None, small=None, audio_feature_type=None
    ):
        dataset_files = list(
            glob.glob(
                PostProcessOpenpose(data_dir=data_dir, yt_video_id="*").output().path
            )
        )
        if small:
            dataset_files = dataset_files[:40]
        self.data = []

        for n, openpose_file_path in enumerate(
            tqdm(dataset_files, desc="Loading dataset. Sit back and relax")
        ):
            filepath = basename(openpose_file_path).replace(".npy", "")
            if audio_feature_type == "spectrogram":
                ms = MelSpectrogram(data_dir=data_dir, yt_video_id=filepath)
                audio_feature_data = np.load(ms.output().path).astype(np.float32)

            audio = DownloadYoutubeAudio(data_dir=data_dir, yt_video_id=filepath)

            openpose_data = np.load(openpose_file_path)
            for frame, all_faces in openpose_data.item().items():
                all_faces = all_faces.astype(np.float32)
                if len(all_faces) > total_frames:
                    data_len = len(all_faces)
                    for i in range(data_len):
                        first_frame = frame + i
                        if first_frame + total_frames < data_len:

                            face = (
                                all_faces[i : i + total_frames]
                                .reshape(-1, 140, 1)
                                .transpose(1, 0, 2)
                            )
                            stack_of_audio_features = []

                            for n in range(total_frames):
                                first_time = round((first_frame + n) * (1.0 / 30.0))
                                stack_of_audio_features.append(
                                    audio_feature_data[
                                        int(
                                            (first_time - 0.26) // ms.hop_duration
                                        ) : int((first_time + 0.26) // ms.hop_duration)
                                    ]
                                )
                            # print(stack_of_audio_features)
                            audio_features = np.array(stack_of_audio_features).reshape(
                                total_frames, -1
                            )

                            if (
                                face.shape[1]
                                == total_frames
                                # and audio_features.shape[0] == total_frames
                            ):
                                assert face.dtype == np.float32
                                self.data.append(
                                    (
                                        face,
                                        audio_features,
                                        first_frame,
                                        audio.output().path,
                                    )
                                )

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
        return {
            "x": face,
            "audio_features": audio_features,
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
