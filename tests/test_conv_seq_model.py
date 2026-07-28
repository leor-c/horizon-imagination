import torch
import gymnasium as gym
from tensordict.tensordict import TensorDict

from horizon_imagination.modules.conv_seq_model import ConvSeqModel
from horizon_imagination.models.world_model.reward_done_model import RewardDoneModel
from horizon_imagination.utilities.types import ObsKey, Modality


IMG_KEY = ObsKey.from_parts(Modality.image, 'features')


def _make_model(num_blocks=1, in_channels=4, spatial=(4, 4), latent_dim=32):
    config = ConvSeqModel.Config(
        in_channels=in_channels,
        latent_spatial_shape=spatial,
        base_channels=16,
        cnn_out_channels=8,
        num_blocks=num_blocks,
        latent_dim=latent_dim,
        ignore_actions=True,
    )
    return config.make_instance()


def _make_obs(b, t, c=4, h=4, w=4):
    x = torch.randn(b, t, c, h, w)
    return TensorDict({IMG_KEY: x}, batch_size=(b, t))


def test_forward_shape_and_finite():
    model = _make_model()
    obs = _make_obs(b=2, t=5)
    out, state = model(actions=None, obs=obs, state=None)
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()
    assert state.shape[1] == min(model.receptive_field - 1, 5)
    print("test_forward_shape_and_finite OK")


def test_causality():
    model = _make_model().eval()
    obs = _make_obs(b=2, t=6)
    out1, _ = model(actions=None, obs=obs, state=None)

    obs_perturbed = obs.clone()
    obs_perturbed[IMG_KEY][:, -1] = torch.randn_like(obs_perturbed[IMG_KEY][:, -1])
    out2, _ = model(actions=None, obs=obs_perturbed, state=None)

    assert torch.equal(out1[:, :-1], out2[:, :-1]), \
        "perturbing the last frame changed earlier outputs -> causal leakage"
    assert not torch.equal(out1[:, -1], out2[:, -1]), \
        "perturbing the last frame had no effect on its own output"
    print("test_causality OK")


def test_state_threading_equivalence():
    model = _make_model(num_blocks=1).eval()  # receptive_field = 5
    context_len, cont_len = 6, 4  # context_len >= receptive_field - 1
    obs_full = _make_obs(b=2, t=context_len + cont_len)

    out_full, _ = model(actions=None, obs=obs_full, state=None)

    obs_ctx = obs_full[:, :context_len]
    obs_cont = obs_full[:, context_len:]
    _, state = model(actions=None, obs=obs_ctx, state=None)
    out_cont, _ = model(actions=None, obs=obs_cont, state=state)

    assert torch.allclose(out_full[:, context_len:], out_cont, atol=1e-5), \
        "state-threaded continuation diverged from an equivalent full-sequence run"
    print("test_state_threading_equivalence OK")


def test_reward_done_model_training_step_smoke():
    action_space = gym.spaces.Discrete(4)
    backbone_config = ConvSeqModel.Config(
        action_space=action_space,
        in_channels=4,
        latent_spatial_shape=(4, 4),
        base_channels=16,
        cnn_out_channels=8,
        num_blocks=1,
        latent_dim=32,
        ignore_actions=True,
    )
    model = RewardDoneModel.Config(backbone_config=backbone_config).make_instance()

    b, t = 2, 5
    obs = _make_obs(b, t)
    actions = torch.randint(0, 4, (b, t))
    rewards = torch.randn(b, t)
    dones = torch.randint(0, 2, (b, t))

    reward_loss, done_loss, info = model.training_step(actions, obs, rewards, dones)
    assert torch.isfinite(reward_loss)
    assert torch.isfinite(done_loss)

    (reward_loss + done_loss).backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms)
    print("test_reward_done_model_training_step_smoke OK")


if __name__ == '__main__':
    test_forward_shape_and_finite()
    test_causality()
    test_state_threading_equivalence()
    test_reward_done_model_training_step_smoke()
