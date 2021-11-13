"""
@author:  Tongjia (Tom) Chen
@contact: tomchen@hnu.edu.cn
"""
import os.path as osp
import cv2
import os
from sklearn.model_selection import train_test_split



class UCF101(object):
    """
    UCF101 dataset:
        13,320 videos, 101 classes
    The files are as following structure:
        UCF-101
    ├── ApplyEyeMakeup
    │   ├── v_ApplyEyeMakeup_g01_c01.avi
    │   └── ...
    ├── ApplyLipstick
    │   ├── v_ApplyLipstick_g01_c01.avi
    │   └── ...
    └── Archery
    │   ├── v_Archery_g01_c01.avi
    │   └── ...
    Note : v stands for video, g stands for groups, v stands for clips
    After preprocessing, the output files are as following structure:
        ucf101
    ├──train
    │   ├── ApplyEyeMakeup
    │   │   ├── v_ApplyEyeMakeup_g01_c01
    │   │   │   ├── 00001.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   ├── ApplyLipstick
    │   │   ├── v_ApplyLipstick_g01_c01
    │   │   │   ├── 00001.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── Archery
    │   │   ├── v_Archery_g01_c01
    │   │   │   ├── 00001.jpg
    │   │   │   └── ...
    │   │   └── ...
        """

    def __init__(self, path='./data', preprocess=False, resize_height=128, resize_width=171):
        self.output_dir = osp.join(path, 'ucf101')
        self.raw_dir = osp.join(path, 'UCF-101')
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.train_path = osp.join(self.output_dir, 'train')
        self.test_path = osp.join(self.output_dir, 'test')

        if preprocess:
            self._preprocess()

        classes = os.listdir(self.raw_dir)
        class2label = {class_: ind for ind, class_ in enumerate(classes)}

        train = []
        test = []
        for class_ in classes:
            groups_train = os.listdir(osp.join(self.train_path, class_))
            groups_test = os.listdir(osp.join(self.test_path, class_))
            for group in groups_train:
                train.append([osp.join(self.train_path, class_, group), class2label[class_]])
            for group in groups_test:
                test.append([osp.join(self.test_path, class_, group), class2label[class_]])

        train_videos = len(train)
        test_videos = len(test)

        self.train = train
        self.test = test
        self.train_videos = train_videos
        self.test_videos = test_videos

        print("==========================")
        print("==UCF101 dataset loaded ==")
        print("13,320 videos, 101 classes")
        print("Training set :{:5} videos".format(train_videos))
        print("Testing  set :{:5} videos".format(test_videos))
        print("==========================")

    def _preprocess(self):
        output_dir = self.output_dir
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir(osp.join(output_dir, 'train'))
            os.mkdir(osp.join(output_dir, 'test'))
        raw_dir = self.raw_dir
        classes = os.listdir(raw_dir)

        meter = 1
        for class_ in classes:
            print("now processing {}, {}/101".format(class_, meter))
            meter += 1
            os.mkdir(osp.join(output_dir, 'train', class_))
            os.mkdir(osp.join(output_dir, 'test', class_))  # create folders in output path
            videos = os.listdir(osp.join(raw_dir, class_))
            train_video, test_video = train_test_split(videos, test_size=0.2, random_state=42)
            for train in train_video:
                self._process_video(train, class_, output_dir)
            for test in test_video:
                self._process_video(test, class_, output_dir, train=False)

    def _process_video(self, video, class_, save_dir, train=True):
        video_name = video.split('.')[0]
        if train:
            os.mkdir(osp.join(save_dir, 'train', class_, video_name))
        else:
            os.mkdir(osp.join(save_dir, 'test', class_, video_name))

        capture = cv2.VideoCapture(osp.join(self.raw_dir, class_, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_height, self.resize_width))

                if train:
                    cv2.imwrite(filename=osp.join(save_dir, 'train', class_, video_name, '{}.jpg'.format(str(i))),
                                img=frame)
                else:
                    cv2.imwrite(filename=osp.join(save_dir, 'test', class_, video_name, '{}.jpg'.format(str(i))),
                                img=frame)
                i += 1
            count += 1

        capture.release()


if __name__ == '__main__':
    # from IPython import embed
    UCF101()
