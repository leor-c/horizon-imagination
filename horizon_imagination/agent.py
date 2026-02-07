from pathlib import Path
import gymnasium as gym
from gymnasium.spaces import Dict, Space
from gymnasium import Env
import lightning as L
from torchrl.data import Storage, RandomSampler, LazyMemmapStorage
from einops import rearrange
import numpy as np
import torch
from loguru import logger
import wandb
import shutil
from functools import partial

from horizon_imagination.data import (
    get_replay_buffer, EpochDataIterator, ReplayBufferTrajectoryIterator,
    SegmentSampler, get_segment_replay_buffer
)
from horizon_imagination.models.tokenizer import CosmosImageTokenizer
from horizon_imagination.models.world_model import RectifiedFlowWorldModel
from horizon_imagination.models.controller import Controller
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities.types import ObsKey, Modality
from horizon_imagination.utilities.visualization import make_border, generate_video, to_img
from horizon_imagination.utilities import shift_fwd
from horizon_imagination.models.world_model.action_producer import (
    StablePolicyActionProducer, FixedActionProducer, NaivePolicyActionProducer,
    NaivePseudoPolicyActionProducer, StablePseudoPolicyActionProducer
)


def _log_files_to_dir(src_dir: Path, dst_dir: Path):
    # Save source code files
    source_dst = dst_dir  # / "source"
    source_dst.mkdir(parents=True, exist_ok=True    )

    # You can filter which files to copy (e.g., *.py only, excluding __pycache__)
    for src_file in Path(src_dir).rglob("*.py"):
        if "venv" in src_file.parts or "__pycache__" in src_file.parts:
            continue  # skip virtual env and cache
        dest_file = source_dst / src_file
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)


class Agent(Configurable, L.LightningModule):
    @dataclass(kw_only=True)
    class Config(BaseConfig):
        @dataclass(kw_only=True)
        class TrainingConfig:
            tokenizer_batch_size: int
            tokenizer_steps_per_epoch: int
            tokenizer_max_grad_norm: float

            world_model_batch_size: int
            world_model_steps_per_epoch: int
            world_model_horizon: int
            world_model_min_segment_length: int
            world_model_max_grad_norm: float

            controller_steps_per_epoch: int
            controller_max_grad_norm: float
            controller_test_frequency: int

        @dataclass(kw_only=True)
        class OnlineTrainingConfig(TrainingConfig):
            num_epochs: int
            tokenizer_train_from_epoch: int
            world_model_train_from_epoch: int
            controller_train_from_epoch: int
            collection_steps_per_epoch: int

        @dataclass(kw_only=True)
        class OfflineTrainingConfig(TrainingConfig):
            tokenizer_num_epochs: int
            world_model_num_epochs: int
            controller_num_epochs: int

            @property
            def num_epochs(self) -> int:
                return (
                    self.tokenizer_num_epochs + 
                    self.world_model_num_epochs + 
                    self.controller_num_epochs
                )
            
            @property
            def controller_first_epoch(self):
                return self.tokenizer_num_epochs + self.world_model_num_epochs 

        obs_space: Dict
        action_space: Space
        env: Env
        replay_buffer_storage: Storage
        image_tokenizer: CosmosImageTokenizer
        world_model: RectifiedFlowWorldModel
        controller: Controller

        training: TrainingConfig

        prefetch: int = 2

        test_env: Env = None

    def __init__(self, config: Config):
        super().__init__()
        self.automatic_optimization = False

        self.config = config

        self.replay_buffer_storage = config.replay_buffer_storage
        self.rb = get_replay_buffer(
            self.replay_buffer_storage, 
            RandomSampler(), 
            batch_size=config.training.tokenizer_batch_size
        )

        self.tokenizer: CosmosImageTokenizer = config.image_tokenizer

        self.world_model: RectifiedFlowWorldModel = config.world_model
        self.controller: Controller = config.controller

        self.components = [
            self.tokenizer,
            self.world_model,
            self.controller,
        ]
        self.max_grad_norms = [
            config.training.tokenizer_max_grad_norm,
            config.training.world_model_max_grad_norm,
            config.training.controller_max_grad_norm,
        ]

        self.epoch_data_iter: EpochDataIterator = None
        self.val_rb = None
        self.epoch = 0

    def on_fit_start(self):
        # log config:
        self.logger.experiment.config.update(self.config.__dict__)

        # copy source & configs to the log dir:
        # log_dir = Path(self.trainer.logger.experiment.dir)
        log_dir = Path(self.trainer.logger.save_dir) / self.trainer.logger.name / self.trainer.logger.experiment.id
        logger.info(f"Copying source & configs to log dir '{log_dir}'...")
        _log_files_to_dir(Path('config'), log_dir)
        _log_files_to_dir(Path('diffusion_wm'), log_dir)
        _log_files_to_dir(Path('experiments'), log_dir)
    
    def configure_optimizers(self):
        tokenizer_optim = self.tokenizer.configure_optimizers()
        wm_optim = self.world_model.configure_optimizers()
        controller_optim = self.controller.configure_optimizers()
        return [tokenizer_optim, wm_optim, controller_optim]

    def on_train_epoch_start(self):
        # collect data:
        self.controller.eval()

        if isinstance(self.config.training, Agent.Config.OnlineTrainingConfig):
            # rich_pbar = self.trainer.progress_bar_callback
            # task_id = rich_pbar.progress.add_task(
            #     "Data collection", total=self.config.training.collection_steps_per_epoch
            # )
            # def pbar_update():
            #     rich_pbar.progress.update(task_id, advance=1)
            #     rich_pbar.refresh()

            self.controller.forward(
                env=self.config.env,
                replay_buffer=self.rb,
                num_steps=self.config.training.collection_steps_per_epoch,
                log_dict_fn=self.log_dict,
                pbar_update_fn=None, #pbar_update
            )
            # rich_pbar.progress.remove_task(task_id)

            if isinstance(self.replay_buffer_storage, LazyMemmapStorage):
                self.replay_buffer_storage.save(Path(self.replay_buffer_storage.scratch_dir))

        self.controller.train()
        self.tokenizer.train()
        self.world_model.train()

    def train_dataloader(self):
        self.epoch += 1

        # initialize an epoch data iterator:
        tokenizer_steps = 0
        world_model_steps = 0
        controller_steps = 0

        if isinstance(self.config.training, Agent.Config.OnlineTrainingConfig):
            if self.epoch >= self.config.training.tokenizer_train_from_epoch:
                tokenizer_steps = self.config.training.tokenizer_steps_per_epoch

            if self.epoch >= self.config.training.world_model_train_from_epoch:
                world_model_steps = self.config.training.world_model_steps_per_epoch

            if self.epoch >= self.config.training.controller_train_from_epoch:
                controller_steps = self.config.training.controller_steps_per_epoch
        else:
            assert isinstance(self.config.training, Agent.Config.OfflineTrainingConfig)
            if self.epoch <= self.config.training.tokenizer_num_epochs:
                tokenizer_steps = self.config.training.tokenizer_steps_per_epoch

            elif self.epoch <= (self.config.training.tokenizer_num_epochs +
                              self.config.training.world_model_num_epochs):
                world_model_steps = self.config.training.world_model_steps_per_epoch

            elif self.epoch <= (self.config.training.tokenizer_num_epochs +
                              self.config.training.world_model_num_epochs +
                              self.config.training.controller_num_epochs):
                controller_steps = self.config.training.controller_steps_per_epoch
        
        if self.epoch_data_iter is None:
            self.epoch_data_iter = EpochDataIterator.Config(
                rb_storage=self.replay_buffer_storage,
                tokenizer_steps=tokenizer_steps,
                world_model_steps=world_model_steps,
                controller_steps=controller_steps,
                tokenizer_batch_size=self.config.training.tokenizer_batch_size,
                wm_segment_length=self.config.training.world_model_horizon,
                wm_min_segment_length=self.config.training.world_model_min_segment_length,
                wm_batch_size=self.config.training.world_model_batch_size,
                c_segment_length=self.controller.config.controller_context_length,
                c_min_segment_length=1,
                c_batch_size=self.controller.config.imagination_batch_size,
                prefetch=self.config.prefetch
            ).make_instance()
        else:
            self.epoch_data_iter.config.tokenizer_steps = tokenizer_steps
            self.epoch_data_iter.config.world_model_steps = world_model_steps
            self.epoch_data_iter.config.controller_steps = controller_steps
        
        return self.epoch_data_iter
    
    def training_step(self, batch, batch_idx):
        batch, component_idx = batch
        optimizer = self.optimizers()[component_idx]

        optimizer.zero_grad()
        log_fn = partial(self.log_dict, batch_size=batch.shape[0])
        loss = self.components[component_idx].training_step(batch, batch_idx, log_fn)

        self.manual_backward(loss)
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=self.max_grad_norms[component_idx], 
            gradient_clip_algorithm="norm"
        )
        optimizer.step()

        return loss
    
    def val_dataloader(self):
        eval_batch_size = 8

        if self.val_rb is None:
            self.val_rb = get_segment_replay_buffer(
                self.replay_buffer_storage,
                rb_sampler=SegmentSampler(
                    segment_len=self.config.training.world_model_horizon,
                    min_length=self.config.training.world_model_horizon,
                    traj_key='episode',
                    pad_direction='suffix',
                    uniform_prob=0.7
                ),
                batch_size=eval_batch_size * self.config.training.world_model_horizon
            )

        data_iterator = ReplayBufferTrajectoryIterator(
            rb=self.val_rb,
            segments_per_batch=eval_batch_size,
            steps_per_segment=self.config.training.world_model_horizon,
        )
        return data_iterator
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self._generate_validation_videos(batch)

        is_controller_learning = not (
            isinstance(self.config.training, Agent.Config.OfflineTrainingConfig) and 
            (self.epoch < self.config.training.controller_first_epoch)
        )
        if is_controller_learning and (self.epoch % self.config.training.controller_test_frequency == 0):
            self.test_controller()

    def _generate_validation_videos(self, batch):
        """
        Generate a video comparing ground truth vs 
        reconstruction vs sampling the policy with our
        method vs naive vs fixed (GT actions).
        """
        if batch is None:
            return
        
        segment_length = self.config.training.world_model_horizon
        context_length = max(2, segment_length // 8)
        horizon = segment_length - context_length

        num_iter = 1
        img_key = ObsKey.from_parts(Modality.image, 'features')
        batch_size = batch.shape[0]

        context_actions = shift_fwd(batch['action'][:, :context_length])
        context_obs = self.world_model.get_obs_from_batch(batch[:, :context_length])
        pad_mask = batch['mask']

        np_ctx = rearrange(batch[:, :context_length]['observation'][img_key].clone(), 'b t c h w -> b t h w c').cpu().numpy()
        np_ctx = make_border(np_ctx, width=3, color=(100, 100, 250))
        predictions = []
        
        if (
            isinstance(self.config.training, Agent.Config.OfflineTrainingConfig) 
            and
            (self.epoch <= self.config.training.tokenizer_num_epochs + 
             self.config.training.world_model_num_epochs)
        ):
            device = self.config.world_model.config.denoiser_config.device
            action_producers = [
                lambda a: FixedActionProducer(actions=a),
                lambda a: NaivePseudoPolicyActionProducer(
                    actions=a,
                    num_actions=self.config.action_space.n,
                    device=device,
                ),
                lambda a: StablePseudoPolicyActionProducer(
                    actions=a,
                    num_actions=self.config.action_space.n,
                    device=device
                )
            ]
        else:
            action_producers = [
                lambda a: FixedActionProducer(actions=a),
                lambda a: NaivePolicyActionProducer(self.controller.actor_critic),
                lambda a: StablePolicyActionProducer(self.controller.actor_critic),
            ]

        actions = batch['action'][:, context_length:]
        
        context = batch[:, :context_length]
        for action_producer in action_producers:
            action_dist, value, v_logits = self.controller.actor_critic.reset(context_actions, context_obs)
            first_action = action_dist.sample()
            context['action'][:, -1:] = first_action
            
            segments = []
            for i in range(num_iter):
                iter_actions = actions[:, i*horizon:(i+1)*horizon]
                ctx = context if i == 0 else None
            
                segment = self.world_model.imagine(
                    policy=action_producer(iter_actions),
                    batch_size=batch_size,
                    horizon=horizon,
                    obs_shape={k: v.shape for k, v in context_obs.items()},
                    denoising_steps=self.controller.config.num_denoising_steps,
                    context=ctx,
                    context_noise_level=self.controller.config.context_noise_level,
                )
                segments.append(segment)
            segment = {
                'observation': torch.cat([s['observation'] for s in segments], dim=1)
            }
            obs_hat = segment['observation']
            obs_hat = to_img(self.tokenizer, obs_hat[img_key])
            obs_hat = np.concatenate([np_ctx, obs_hat], axis=1)
            predictions.append(obs_hat)

        ground_truth = rearrange(batch[:, context_length:]['observation'][img_key], 'b t c h w -> b t h w c').cpu().numpy()
        ground_truth = np.concatenate([np_ctx, ground_truth], axis=1)
        rec = self.tokenizer.forward(batch[:, context_length:]['observation'][img_key].flatten(0, 1))
        rec = rearrange(rec, '(b t) c h w -> b t h w c', b=batch_size).cpu().numpy()
        rec = np.concatenate([np_ctx, rec], axis=1)

        labels = ['Ground Truth', 'Reconstructions', 'Fixed', 'Naive', 'Ours']
        sequences = [ground_truth, rec, *predictions]
        video_path = 'eval_last.mp4'
        frames = generate_video(sequences, labels, output_path=video_path, fps=5, scale=4)

        # assume wandb logger!
        wandb_logger = self.logger
        wandb_logger.log_video(f"Evaluation/imagination", videos=['eval_last.mp4'], format=['gif'])

    def test_controller(self):
        if self.config.test_env is not None:
            _, episodic_returns = self.controller.collect_test_episodes(
                num_episodes=10, 
                env=self.config.test_env
            )
            metrics = {
                'Evaluation/avg_return': np.mean(episodic_returns),
                'Evaluation/max_return': np.max(episodic_returns),
            }
            self.log_dict(metrics)

    def _close_envs(self):
        logger.info('closing envs...')
        self.config.env.close()
        logger.info('closed train env.')
        if self.config.test_env is not None:
            self.config.test_env.close()
            logger.info('closed test env.')

    def on_fit_end(self):
        self._close_envs()
