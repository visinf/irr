from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import tools


VALIDATE_INDICES = [
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 536, 537, 538, 539,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661,
    662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
    674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
    686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978,
    979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990,
    991]


class _Sintel(data.Dataset):
    def __init__(self,
                 args,
                 dir_root=None,
                 photometric_augmentations=False,
                 imgtype=None,
                 dstype=None):

        self._args = args

        images_root = os.path.join(dir_root, imgtype)
        if imgtype is "comb":
            images_root = os.path.join(dir_root, "clean")
        flow_root = os.path.join(dir_root, "flow")
        occ_root = os.path.join(dir_root, "occlusions_rev")

        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")
        if flow_root is not None and not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!")
        if occ_root is not None and not os.path.isdir(occ_root):
            raise ValueError("Occ directory '%s' not found!")
        
        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))
        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = tools.cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        self._image_list = []
        self._flow_list = []
        self._occ_list = []

        for base_folder in base_folders:            
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            flo_filenames = [x for x in all_flo_filenames if base_folder in x]
            occ_filenames = [x for x in all_occ_filenames if base_folder in x]

            for i in range(len(img_filenames) - 1):

                im1 = img_filenames[i]
                im2 = img_filenames[i + 1]
                flo = flo_filenames[i]
                occ = occ_filenames[i]

                self._image_list += [[im1, im2]]
                self._flow_list += [flo]
                self._occ_list += [occ]

                # Sanity check
                im1_base_filename = os.path.splitext(os.path.basename(im1))[0]
                im2_base_filename = os.path.splitext(os.path.basename(im2))[0]
                flo_base_filename = os.path.splitext(os.path.basename(flo))[0]
                occ_base_filename = os.path.splitext(os.path.basename(occ))[0]
                im1_frame, im1_no = im1_base_filename.split("_")
                im2_frame, im2_no = im2_base_filename.split("_")
                assert(im1_frame == im2_frame)
                assert(int(im1_no) == int(im2_no) - 1)
                
                flo_frame, flo_no = flo_base_filename.split("_")
                assert(im1_frame == flo_frame)
                assert(int(im1_no) == int(flo_no))
                
                occ_frame, occ_no = occ_base_filename.split("_")
                assert(im1_frame == occ_frame)
                assert(int(im1_no) == int(occ_no))
        
        assert len(self._image_list) == len(self._flow_list)        
        assert len(self._image_list) == len(self._occ_list)

        # -------------------------------------------------------------
        # Remove invalid validation indices
        # -------------------------------------------------------------
        full_num_examples = len(self._image_list)
        validate_indices = [x for x in VALIDATE_INDICES if x in range(full_num_examples)]

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        list_of_indices = None
        if dstype == "train":
            list_of_indices = [x for x in range(full_num_examples) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(full_num_examples)
        else:
            raise ValueError("dstype '%s' unknown!", dstype)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = [self._image_list[i] for i in list_of_indices]
        self._flow_list = [self._flow_list[i] for i in list_of_indices]
        self._occ_list = [self._occ_list[i] for i in list_of_indices]

        if imgtype is "comb":
            image_list_final = [[val[0].replace("clean", "final"), val[1].replace("clean", "final")] for idx, val in enumerate(self._image_list)]
            self._image_list += image_list_final
            self._flow_list += self._flow_list
            self._occ_list += self._occ_list

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) == len(self._occ_list)

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

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]
        flo_filename = self._flow_list[index]
        occ_filename = self._occ_list[index]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_np0 = common.read_flo_as_float32(flo_filename)
        occ_np0 = common.read_occ_image_as_float32(occ_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)
        flo = common.numpy2torch(flo_np0)
        occ = common.numpy2torch(occ_np0)

        # e.g. "clean/alley_1/"
        basedir = os.path.splitext(os.path.dirname(im1_filename).replace(self._substract_base, "")[1:])[0]

        # example filename
        basename = os.path.splitext(os.path.basename(im1_filename))[0]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basedir": basedir,
            "basename": basename,
            "target1": flo,
            "target_occ1": occ
        }

        return example_dict

    def __len__(self):
        return self._size


class _Sintel_test(data.Dataset):
    def __init__(self,
                 args,
                 dir_root=None,
                 photometric_augmentations=False,
                 imgtype=None):

        self._args = args
        images_root = os.path.join(dir_root, imgtype)
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = tools.cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        self._image_list = []

        for base_folder in base_folders:            
            img_filenames = [x for x in all_img_filenames if base_folder in x]

            for i in range(len(img_filenames) - 1):

                im1 = img_filenames[i]
                im2 = img_filenames[i + 1]
                self._image_list += [[im1, im2]]

                # Sanity check
                im1_base_filename = os.path.splitext(os.path.basename(im1))[0]
                im2_base_filename = os.path.splitext(os.path.basename(im2))[0]
                im1_frame, im1_no = im1_base_filename.split("_")
                im2_frame, im2_no = im2_base_filename.split("_")
                assert(im1_frame == im2_frame)
                assert(int(im1_no) == int(im2_no) - 1)                

        full_num_examples = len(self._image_list)
        list_of_indices = range(full_num_examples)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = [self._image_list[i] for i in list_of_indices]
        
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

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # e.g. "clean/alley_1/"
        basedir = os.path.splitext(os.path.dirname(im1_filename).replace(self._substract_base, "")[1:])[0]

        # example filename
        basename = os.path.splitext(os.path.basename(im1_filename))[0]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basedir": basedir,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size


class SintelTrainingCleanTrain(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCleanTrain, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="train")


class SintelTrainingCleanValid(_Sintel):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCleanValid, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="valid")


class SintelTrainingCleanFull(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCleanFull, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="full")


class SintelTrainingFinalTrain(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingFinalTrain, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="train")


class SintelTrainingFinalValid(_Sintel):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingFinalValid, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="valid")


class SintelTrainingFinalFull(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingFinalFull, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="full")


class SintelTrainingCombTrain(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCombTrain, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="train")


class SintelTrainingCombValid(_Sintel):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCombValid, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="valid")


class SintelTrainingCombFull(_Sintel):
    def __init__(self, args, root, photometric_augmentations=True):
        dir_root = os.path.join(root, "training")
        super(SintelTrainingCombFull, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="full")


class SintelTestClean(_Sintel_test):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "test")
        super(SintelTestClean, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean")


class SintelTestFinal(_Sintel_test):
    def __init__(self, args, root, photometric_augmentations=False):
        dir_root = os.path.join(root, "test")
        super(SintelTestFinal, self).__init__(
            args,
            dir_root=dir_root,
            photometric_augmentations=photometric_augmentations,
            imgtype="final")
