import os
import shutil

import numpy as np
import torch
import torchvision
import wandb
from torch.utils.tensorboard import SummaryWriter

from research_project.utils import Config, Context
from sen import huggingface_upload


class Logger:
    def __init__(self):
        self.learning_rate = float("inf")
        self.v_loss = float("inf")
        self.pg_loss = float("inf")
        self.entropy_loss = float("inf")
        self.old_approx_kl = float("inf")
        self.approx_kl = float("inf")
        self.clipfracs = float("inf")
        self.explained_var = float("inf")
        self.mean_episodic_return = float("inf")
        self.episode = float("inf")
        self.tax_frac = float("inf")
        self.mean_reward_across_envs = float("inf")

    def log_video(self, run_name, episode, track, obs):
        video = torch.cat(obs, dim=0).cpu()
        try:
            os.mkdir(f"./videos_{run_name}")
        except FileExistsError:
            pass
        torchvision.io.write_video(
            f"./videos_{run_name}/episode_{episode}.mp4",
            video,
            fps=20,
        )
        huggingface_upload.upload(f"./videos_{run_name}", run_name)
        if track:
            wandb.log(
                {"video": wandb.Video(f"./videos_{run_name}/episode_{episode}.mp4")}
            )
        os.remove(f"./videos_{run_name}/episode_{episode}.mp4")

    def log(
        self, run_name, args: Config, ctx: Context, writer: SummaryWriter, tax_frac
    ):
        if args.capture_video and ctx.current_episode % args.video_freq == 0:
            # currently only records first of any parallel games running but
            # this is easily changed at the point where we add to episode_world_obs
            self.log_video(
                run_name, ctx.current_episode, args.track, ctx.episode_world_obs
            )

        writer.add_scalar(
            "charts/learning_rate",
            ctx.optimizer.param_groups[0]["lr"],
            ctx.current_episode,
        )
        writer.add_scalar("losses/value_loss", self.v_loss.item(), ctx.current_episode)
        writer.add_scalar(
            "losses/policy_loss", self.pg_loss.item(), ctx.current_episode
        )
        writer.add_scalar(
            "losses/entropy", self.entropy_loss.item(), ctx.current_episode
        )
        writer.add_scalar(
            "losses/old_approx_kl", self.old_approx_kl.item(), ctx.current_episode
        )
        writer.add_scalar(
            "losses/approx_kl", self.approx_kl.item(), ctx.current_episode
        )
        writer.add_scalar(
            "losses/clipfrac", np.mean(self.clipfracs), ctx.current_episode
        )
        writer.add_scalar(
            "losses/explained_variance", self.explained_var, ctx.current_episode
        )
        writer.add_scalar(
            "charts/mean_episodic_return",
            torch.mean(self.mean_episodic_return),
            ctx.current_episode,
        )
        writer.add_scalar("charts/episode", ctx.current_episode, ctx.current_episode)
        writer.add_scalar("charts/tax_frac", self.tax_frac, ctx.current_episode)
        mean_rewards_across_envs = {
            player_idx: 0 for player_idx in range(0, ctx.num_agents)
        }
        for idx in range(len(ctx.episode_rewards)):
            mean_rewards_across_envs[idx % ctx.num_agents] += ctx.episode_rewards[
                idx
            ].item()
        mean_rewards_across_envs = list(
            map(
                lambda x: x / args.num_parallel_games,
                mean_rewards_across_envs.values(),
            )
        )

        for player_idx in range(ctx.num_agents):
            writer.add_scalar(
                f"charts/episodic_return-player{player_idx}",
                mean_rewards_across_envs[player_idx],
                ctx.current_episode,
            )
        print(
            f"Finished episode {ctx.current_episode}, with {ctx.num_policy_updates_per_ep} policy updates"
        )
        print(f"Mean episodic return: {torch.mean(ctx.episode_rewards)}")
        print(f"Episode returns: {mean_rewards_across_envs}")
        print(f"Principal returns: {ctx.principal_episode_rewards.tolist()}")
        for game_id in range(args.num_parallel_games):
            writer.add_scalar(
                f"charts/principal_return_game{game_id}",
                ctx.principal_episode_rewards[game_id].item(),
                ctx.current_episode,
            )
            for tax_period in range(len(ctx.tax_values)):
                tax_step = (
                    ctx.current_episode - 1
                ) * args.episode_length // args.tax_period + tax_period
                for bracket in range(0, 3):
                    writer.add_scalar(
                        f"charts/tax_value_game{game_id}_bracket_{bracket+1}",
                        np.array(
                            ctx.tax_values[tax_period][f"game_{game_id}"][bracket]
                        ),
                        tax_step,
                    )

        print(
            f"Tax a_values this episode (for each period): {ctx.tax_values}, capped by multiplier {tax_frac}"
        )
        print("*******************************")

        if args.save_model and ctx.current_episode % args.save_model_freq == 0:
            try:
                os.mkdir(f"./models_{run_name}")
            except FileExistsError:
                pass
            torch.save(
                ctx.agent.state_dict(),
                f"./models_{run_name}/agent_{ctx.current_episode}.pth",
            )
            torch.save(
                ctx.principal_agent.state_dict(),
                f"./models_{run_name}/principal_{ctx.current_episode}.pth",
            )
            huggingface_upload.upload(f"./models_{run_name}", run_name)
            os.remove(f"./models_{run_name}/agent_{ctx.current_episode}.pth")
            os.remove(f"./models_{run_name}/principal_{ctx.current_episode}.pth")

            huggingface_upload.upload(f"./saved_params_{run_name}", run_name)
            shutil.rmtree(f"./saved_params_{run_name}/ep{ctx.current_episode}")
            print("model saved")
