import json
import os
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image


class VideoUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_video_and_save_frames(read_dir, feed_dir):
        vidcap = cv2.VideoCapture(read_dir)
        os.makedirs(feed_dir, exist_ok=True)
        success, frame = vidcap.read()
        count = 1
        while success:
            cv2.imwrite(os.path.join(feed_dir, "frame_{}.jpg".format(count)), frame)
            success, frame = vidcap.read()
            count += 1

    @staticmethod
    def split_frame_vertically(frame: Image, slices: int) -> List:
        """
        Slices a frame vertically into parts and returns the parts in a list.
        :param frame: The frame that needs to be split. A PIL Image object.
        :param slices: The number of slices the image needs to be split into
        :return:
        """
        split_frames = []
        width, height = frame.size
        left = 0
        slice_size = width / slices
        count = 1
        for s in range(slices):
            if count == slices:
                right = width
            else:
                right = int(count * slice_size)

            crop_box = (left, 0, right, height)
            split_frames.append(frame.crop(crop_box))
            left += slice_size
            count += 1
        return split_frames


class VideoAnalysis:
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self._sorted_file_names = sorted(os.listdir(frames_dir))

    def read_frames(self):
        for file_name in self._sorted_file_names:
            frame = Image.open(os.path.join(self.frames_dir, file_name))
            yield frame

    def read_vertically_split_frames(self):
        for file_name in self._sorted_file_names:
            frame = Image.open(os.path.join(self.frames_dir, file_name))
            sub_frames = VideoUtils.split_frame_vertically(frame, 2)
            yield sub_frames

    @staticmethod
    def get_no_of_high_intensity_pixels(arr):
        hip = 0
        for row in arr:
            for r, g, b in row:
                intensity = (r + g + b) / 3
                if intensity > 225:
                    hip += 1
        return hip

    def quantify_grayscale_frames(self, dir):
        quantified_frames = dict()
        qf_list = []
        for count, (f1, f2) in enumerate(self.read_vertically_split_frames()):
            f1_array = np.array(f1, dtype="int64")
            f2_array = np.array(f2, dtype="int64")
            f1_hip = self.get_no_of_high_intensity_pixels(f1_array)
            f2_hip = self.get_no_of_high_intensity_pixels(f2_array)
            quantified_frames["frame_{}".format(count + 1)] = [f1_hip, f2_hip]
            qf_list.append([f1_hip, f2_hip])
        df = pd.DataFrame(qf_list)
        df.to_csv("/Users/csatyajith/PycharmProjects/car_t_quantification/quantified_frames.csv", header=False)
        with open(os.path.join(dir, "quantified_frames.json"), "w") as f:
            json.dump(quantified_frames, f)


if __name__ == '__main__':
    # VideoUtils.get_video_and_save_frames("/Users/csatyajith/Datasets/dongfang/combined_stacks.avi",
    #                                      "/Users/csatyajith/PycharmProjects/car_t_quantification/combined_stacks")
    video_analysis = VideoAnalysis("/Users/csatyajith/PycharmProjects/car_t_quantification/combined_stacks")
    video_analysis.quantify_grayscale_frames("/Users/csatyajith/PycharmProjects/car_t_quantification/")

"""
"/Users/csatyajith/Datasets/dongfang"
"""
