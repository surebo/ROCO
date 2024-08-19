REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .role_episode_runner import RoleEpisodeRunner
REGISTRY["roleEpisode"] = RoleEpisodeRunner