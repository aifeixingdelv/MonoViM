# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from timm import create_model
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.save_opts()
        if not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
            setup_logging(os.path.join(self.log_path, 'logger.log'))
            logging.info("Experiment is named: %s", self.opt.model_name)
            logging.info("Saving to: %s", os.path.abspath(self.log_path))
        else:
            setup_logging(os.path.join(self.log_path, 'logger.log'), filemode='a')

        self.writers = {}
        for mode in ["train"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, "tensorboard", mode))

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        if self.opt.seed > 0:
            self.set_seed(self.opt.seed)
        else:
            cudnn.benchmark = True

        self.ep_start = 0
        self.batch_start = 0
        self.step = 0

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.backbone == "resnet":
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)

        if self.opt.backbone == "MonoViM":
            self.models["encoder"] = create_model(self.opt.backbone,
                                                  pretrained=self.opt.pretrained_weights_path).cuda()

            self.models["depth"] = networks.MambaDepthDecoder_v1(
                encoder_dims=self.models["encoder"].config["dims"]).cuda()

        if self.opt.backbone == "MonoViM_LST":
            self.models["encoder"] = create_model(self.opt.backbone,
                                                  pretrained=self.opt.pretrained_weights_path).cuda()

            self.models["depth"] = networks.MambaDepthDecoder_v2(
                encoder_dims=self.models["encoder"].config["dims"]).cuda()

        self.models["encoder"].to(self.device)
        self.models["depth"].to(self.device)

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)

        for k in self.models.keys():
            self.parameters_to_train += list(self.models[k].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
            
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes": datasets.CityscapesDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == "kitti":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.split, "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "{}_files.txt")
        elif self.opt.dataset == "kitti_odom":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", "odom", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/kitti", "odom", "{}_files_09.txt")
        elif self.opt.dataset == "cityscapes":
            fpath = os.path.join(os.path.dirname(__file__), "splits/cityscapes", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/cityscapes", "{}_files.txt")
        else:
            pass

        train_filenames = readlines(fpath.format("train"))
        test_filenames = readlines(fpath_test.format("test"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs
        
        if self.opt.dataset == "cityscapes":
            train_dataset = self.dataset(
                self.opt.data_path_pre, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            [0], self.num_scales, is_train=False, img_ext=img_ext)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        if self.opt.dataset == "kitti":
            gt_path = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        elif self.opt.dataset == "cityscapes":
            gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")
            self.gt_depths = []
            for i in range(len(test_dataset)):
                gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))
                self.gt_depths.append(gt_depth)
        else:
            pass

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        if self.opt.resume:
            checkpoint = self.load_ckpt()
            self.model_optimizer.load_state_dict(checkpoint["optimizer"])
            self.model_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            del checkpoint

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        logging.info("Using split: %s", self.opt.split)
        logging.info(
            "There are {:d} training items and {:d} test items\n".format(len(train_dataset), len(test_dataset)))

    def test_kitti(self):
        """Test the model on a single minibatch
        """
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        # Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()

        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            print("{}/{}".format(idx + 1, len(self.test_loader)), end='\r')
            input_color = data[("color", 0, 0)].to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i:i + 1].unsqueeze(0)
            pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=False)
            pred_depth = 1 / pred_disp[0, 0, :]
            if self.opt.eval_split == "eigen":
                mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
                crop_mask = torch.zeros_like(mask)
                crop_mask[
                int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                mask = mask * crop_mask
            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def test_cityscapes(self):
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()

        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            print("{}/{}".format(idx + 1, len(self.test_loader)), end='\r')
            input_color = data[("color", 0, 0)].to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]

            # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
            pred_disp = pred_disps[i:i + 1].unsqueeze(0)
            pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=True)
            pred_depth = 1 / pred_disp[0, 0, :]

            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

            mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def set_seed(self, seed=1234):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        for self.epoch in range(self.ep_start, self.opt.num_epochs):
            self.run_epoch()
            self.model_lr_scheduler.step()

            with torch.no_grad():
                if self.opt.dataset == "kitti":
                    self.test_kitti()
                elif self.opt.dataset == "cityscapes":
                    self.test_cityscapes()
                else:
                    pass

            self.save_model(ep_end=True)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        logging.info("Training epoch {}\n".format(self.epoch))
        self.set_train()

        start_data_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            self.step += 1
            start_fp_time = time.time()
            outputs, losses = self.process_batch(inputs)

            start_bp_time = time.time()
            self.model_optimizer.zero_grad()

            losses["loss"].backward()
            self.model_optimizer.step()

            # compute the process time
            data_time = start_fp_time - start_data_time
            fp_time = start_bp_time - start_fp_time
            bp_time = time.time() - start_bp_time

            # logging
            if ((batch_idx + self.batch_start) % self.opt.log_frequency == 0):
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log_time(batch_idx + self.batch_start, data_time, fp_time, bp_time, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)

            # save ckpt
            if ((batch_idx + self.batch_start) > 0 and (batch_idx + self.batch_start) % self.opt.save_frequency == 0):
                self.save_model(batch_idx=batch_idx + self.batch_start + 1)

            start_data_time = time.time()

        self.batch_start = 0

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            try:
                inputs[key] = ipt.to(self.device)
            except:
                pass

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features_out, features = self.models["mamba_unet"].forward_features(inputs[("color_aug", 0, 0)])
            # outputs = self.models["mamba_unet"].depth_forward_up_features(features_out, features)
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(iter(self.val_iter))
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, data_time, fp_time, bp_time, loss):
        """Print a logging statement to the terminal
        """
        batch_time = data_time + fp_time + bp_time
        # time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps - self.step) * batch_time if self.step > 1 else 0
        print_string = "epoch: {:>2}/{} | batch: {:>4}/{} | data time: {:.4f}" + " | batch time: {:.3f} | loss: {:.4f} | lr: {:.2e} | time left: {}"
        logging.info(
            print_string.format(self.epoch, self.opt.num_epochs - 1, batch_idx, self.num_steps_per_epoch, data_time,
                                batch_time, loss, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, ep_end=False, batch_idx=0):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        to_save = {}
        
        for model_name, model in self.models.items():
            to_save[model_name] = model.state_dict()
            if ep_end:
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save_every_model = model.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save_every_model['height'] = self.opt.height
                    to_save_every_model['width'] = self.opt.width
                    to_save_every_model['use_stereo'] = self.opt.use_stereo
                torch.save(to_save_every_model, save_path)

        to_save['height'] = self.opt.height
        to_save['width'] = self.opt.width
        to_save['use_stereo'] = self.opt.use_stereo
        if ep_end:
            to_save["epoch"] = self.epoch + 1
        else:
            to_save["epoch"] = self.epoch

        to_save['step_in_total'] = self.step
        to_save["batch_idx"] = batch_idx
        to_save['optimizer'] = self.model_optimizer.state_dict()
        to_save['lr_scheduler'] = self.model_lr_scheduler.state_dict()

        save_path = os.path.join(self.log_path, "ckpt.pth")
        torch.save(to_save, save_path)  ## also save the optimizer state for resuming

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        logging.info("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            logging.info("Loading {} weights successfully".format(n))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            logging.info("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            logging.info("Cannot find Adam weights so Adam is randomly initialized")

    def load_ckpt(self):
        """Load checkpoint to resume a training, used in training process.
        """
        logging.info(" ")
        load_path = os.path.join(self.log_path, "ckpt.pth")
        if not os.path.exists(load_path):
            logging.info("No checkpoint to resume, train from epoch 0.")
            return None

        logging.info("Resume checkpoint from {}".format(os.path.abspath(load_path)))
        checkpoint = torch.load(load_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.ep_start = checkpoint['epoch']
        self.batch_start = checkpoint['batch_idx']
        self.step = checkpoint['step_in_total']
        logging.info("Start at eopch {}, batch index {}".format(self.ep_start, self.batch_start))
        return checkpoint
