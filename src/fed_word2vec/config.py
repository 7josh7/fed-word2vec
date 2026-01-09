"""
Configuration management for Fed Word2Vec pipeline.

Provides centralized configuration for paths, download settings, and text
cleaning parameters. Supports loading from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    """Directory paths configuration."""
    data_root: Path = Path("data")
    downloads_dir: Path = Path("data/fed_downloads")
    text_dir: Path = Path("data/fed_text")
    model_dir: Path = Path("model")
    
    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.downloads_dir = Path(self.downloads_dir)
        self.text_dir = Path(self.text_dir)
        self.model_dir = Path(self.model_dir)


@dataclass
class DownloadConfig:
    """Download settings."""
    user_agent: str = "FedResearchDownloader/1.0 (contact: research@example.com)"
    delay: float = 0.4
    timeout: int = 30
    start_year: int = 1994
    end_year: int | None = None  # None = current year
    target_speakers: list[str] = field(default_factory=lambda: ["Powell", "Yellen", "Bernanke"])


@dataclass
class CleaningConfig:
    """Text cleaning parameters."""
    removal_guard: float = 0.4  # Max allowed removal ratio before fallback
    min_guard: float = 0.30
    max_guard: float = 0.60


@dataclass
class Word2VecConfig:
    """Word2Vec model hyperparameters."""
    vector_size: int = 200
    window: int = 10
    min_count: int = 5
    workers: int = 4
    epochs: int = 10


@dataclass
class Config:
    """Main configuration container."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    word2vec: Word2VecConfig = field(default_factory=Word2VecConfig)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        paths = PathsConfig(**data.get("paths", {}))
        download = DownloadConfig(**data.get("download", {}))
        cleaning = CleaningConfig(**data.get("cleaning", {}))
        word2vec = Word2VecConfig(**data.get("word2vec", {}))
        return cls(paths=paths, download=download, cleaning=cleaning, word2vec=word2vec)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "paths": {
                "data_root": str(self.paths.data_root),
                "downloads_dir": str(self.paths.downloads_dir),
                "text_dir": str(self.paths.text_dir),
                "model_dir": str(self.paths.model_dir),
            },
            "download": {
                "user_agent": self.download.user_agent,
                "delay": self.download.delay,
                "timeout": self.download.timeout,
                "start_year": self.download.start_year,
                "end_year": self.download.end_year,
                "target_speakers": self.download.target_speakers,
            },
            "cleaning": {
                "removal_guard": self.cleaning.removal_guard,
                "min_guard": self.cleaning.min_guard,
                "max_guard": self.cleaning.max_guard,
            },
            "word2vec": {
                "vector_size": self.word2vec.vector_size,
                "window": self.word2vec.window,
                "min_count": self.word2vec.min_count,
                "workers": self.word2vec.workers,
                "epochs": self.word2vec.epochs,
            },
        }


def load_config(path: Path | str | None = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses default config.
        
    Returns:
        Config object with loaded or default settings.
    """
    if path is None:
        return Config()
    
    path = Path(path)
    if not path.exists():
        print(f"Config file not found: {path}, using defaults")
        return Config()
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return Config.from_dict(data)


def save_config(config: Config, path: Path | str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        path: Path to save config file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


# Default config file location
DEFAULT_CONFIG_PATH = Path("config.yaml")
