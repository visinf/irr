from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import numpy as np


def fillingInNaN(flow):
    h, w, c = flow.shape
    indices = np.argwhere(np.isnan(flow))
    neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for ii, idx in enumerate(indices):
        sum_sample = 0
        count = 0
        for jj in range(0, len(neighbors) - 1):
            hh = idx[0] + neighbors[jj][0]
            ww = idx[1] + neighbors[jj][1]
            if hh < 0 or hh >= h:
                continue
            if ww < 0 or ww >= w:
                continue
            sample_flow = flow[hh, ww, idx[2]]
            if np.isnan(sample_flow):
                continue
            sum_sample += sample_flow
            count += 1
        if count is 0:
            print('FATAL ERROR: no sample')
        flow[idx[0], idx[1], idx[2]] = sum_sample / count

    return flow


class FlyingThings3d(data.Dataset):
    def __init__(self,
                 args,
                 images_root,
                 flow_root,
                 occ_root,
                 photometric_augmentations=False):

        self._args = args
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")
        if flow_root is not None and not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!")
        if occ_root is not None and not os.path.isdir(occ_root):
            raise ValueError("Occ directory '%s' not found!")

        if flow_root is not None:
            flow_f_filenames = sorted(glob(os.path.join(flow_root, "into_future/*.flo")))
            flow_b_filenames = sorted(glob(os.path.join(flow_root, "into_past/*.flo")))

        if occ_root is not None:
            occ1_filenames = sorted(glob(os.path.join(occ_root, "into_future/*.png")))
            occ2_filenames = sorted(glob(os.path.join(occ_root, "into_past/*.png")))

        all_img_filenames = sorted(glob(os.path.join(images_root, "*.png")))

        self._image_list = []
        self._flow_list = [] if flow_root is not None else None
        self._occ_list = [] if occ_root is not None else None

        assert len(all_img_filenames) != 0
        assert len(flow_f_filenames) != 0
        assert len(flow_b_filenames) != 0
        assert len(occ1_filenames) != 0
        assert len(occ2_filenames) != 0

        ## path definition
        path_flow_f = os.path.join(flow_root, "into_future")
        path_flow_b = os.path.join(flow_root, "into_past")
        path_occ_f = os.path.join(occ_root, "into_future")
        path_occ_b = os.path.join(occ_root, "into_past")

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------

        for ii in range(0, len(flow_f_filenames)):

            flo_f = flow_f_filenames[ii]

            idx_f = os.path.splitext(os.path.basename(flo_f))[0]
            idx_b = str(int(idx_f) + 1).zfill(7)

            flo_b = os.path.join(path_flow_b, idx_b + ".flo")

            im1 = os.path.join(images_root, idx_f + ".png")
            im2 = os.path.join(images_root, idx_b + ".png")
            occ1 = os.path.join(path_occ_f, idx_f + ".png")
            occ2 = os.path.join(path_occ_b, idx_b + ".png")

            if not os.path.isfile(flo_f) or not os.path.isfile(flo_b) or not os.path.isfile(im1) or not os.path.isfile(
                    im2) or not os.path.isfile(occ1) or not os.path.isfile(occ2):
                continue

            self._image_list += [[im1, im2]]
            self._flow_list += [[flo_f, flo_b]]
            self._occ_list += [[occ1, occ2]]

        self._size = len(self._image_list)

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._occ_list) == len(self._flow_list)
        assert len(self._image_list) != 0

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]
        flo_f_filename = self._flow_list[index][0]
        flo_b_filename = self._flow_list[index][1]
        occ1_filename = self._occ_list[index][0]
        occ2_filename = self._occ_list[index][1]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_f_np0 = common.read_flo_as_float32(flo_f_filename)
        flo_b_np0 = common.read_flo_as_float32(flo_b_filename)
        occ1_np0 = common.read_occ_image_as_float32(occ1_filename)
        occ2_np0 = common.read_occ_image_as_float32(occ2_filename)

        # temp - check isnan
        if np.any(np.isnan(flo_f_np0)):
            flo_f_np0 = fillingInNaN(flo_f_np0)

        if np.any(np.isnan(flo_b_np0)):
            flo_b_np0 = fillingInNaN(flo_b_np0)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # convert flow to FloatTensor
        flo_f = common.numpy2torch(flo_f_np0)
        flo_b = common.numpy2torch(flo_b_np0)

        # convert occ to FloatTensor
        occ1 = common.numpy2torch(occ1_np0)
        occ2 = common.numpy2torch(occ2_np0)

        # example filename
        basename = os.path.basename(im1_filename)[:5]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo_f,
            "target2": flo_b,
            "target_occ1": occ1,
            "target_occ2": occ2,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size


class FlyingThings3dFinalTrain(FlyingThings3d):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=True):
        images_root = os.path.join(root, "frames_finalpass")
        flow_root = os.path.join(root, "optical_flow")
        occ_root = os.path.join(root, "occlusion")
        super(FlyingThings3dFinalTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            photometric_augmentations=photometric_augmentations)


class FlyingThings3dFinalTest(FlyingThings3d):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False):
        images_root = os.path.join(root, "frames_finalpass")
        flow_root = os.path.join(root, "optical_flow")
        occ_root = os.path.join(root, "occlusion")
        super(FlyingThings3dFinalTest, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            photometric_augmentations=photometric_augmentations)


class FlyingThings3dCleanTrain(FlyingThings3d):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=True):
        images_root = os.path.join(root, "train", "image_clean", "left")
        flow_root = os.path.join(root, "train", "flow", "left")
        occ_root = os.path.join(root, "train", "flow_occlusions", "left")
        super(FlyingThings3dCleanTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            photometric_augmentations=photometric_augmentations)


class FlyingThings3dCleanTest(FlyingThings3d):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False):
        images_root = os.path.join(root, "frames_cleanpass")
        flow_root = os.path.join(root, "optical_flow")
        occ_root = os.path.join(root, "occlusion")
        super(FlyingThings3dCleanTest, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            photometric_augmentations=photometric_augmentations)
