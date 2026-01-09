"""
Command-line interface for Fed Word2Vec pipeline.

Provides commands for downloading Fed documents and cleaning text.
"""

import argparse
import sys
from pathlib import Path

from .config import Config, load_config, save_config, DEFAULT_CONFIG_PATH
from .downloader import create_session, collect_fomc_documents, collect_speeches
from .text_cleaner import process_tree, recommend_guard_from_profile


def cmd_download(args: argparse.Namespace, config: Config) -> int:
    """Download Fed documents."""
    session = create_session(config.download.user_agent)
    
    downloads_dir = config.paths.downloads_dir
    
    if args.what in ("all", "fomc"):
        print("=== Downloading FOMC documents ===")
        fomc_dir = downloads_dir / "fomc"
        records = collect_fomc_documents(session, fomc_dir)
        print(f"Downloaded {len(records)} FOMC documents to {fomc_dir}")
    
    if args.what in ("all", "speeches"):
        print("\n=== Downloading Fed Chair speeches ===")
        speeches_dir = downloads_dir / "speeches"
        records = collect_speeches(
            session,
            speeches_dir,
            start_year=config.download.start_year,
            end_year=config.download.end_year,
        )
        print(f"Downloaded {len(records)} speeches to {speeches_dir}")
    
    print("\nDownload complete!")
    return 0


def cmd_clean(args: argparse.Namespace, config: Config) -> int:
    """Clean downloaded HTML files to text."""
    downloads_dir = config.paths.downloads_dir
    text_dir = config.paths.text_dir
    
    removed_lines: list[str] = []
    
    # Process FOMC documents
    if args.what in ("all", "fomc"):
        print("=== Cleaning FOMC documents ===")
        fomc_in = downloads_dir / "fomc"
        fomc_out = text_dir / "fomc"
        
        if fomc_in.exists():
            # Profile to get recommended guard
            print("Profiling corpus for removal guard recommendation...")
            _, _, _, _, ratios = process_tree(
                fomc_in, fomc_out, is_fomc=True, removal_guard=None
            )
            guard = recommend_guard_from_profile(ratios, config.cleaning.removal_guard)
            
            # Process with recommended guard
            files, chars, tokens, counter, _ = process_tree(
                fomc_in, fomc_out,
                is_fomc=True,
                removal_guard=guard,
                removed_collector=removed_lines if args.save_removed else None
            )
            print(f"Processed {files} FOMC files -> {fomc_out}")
            print(f"  Characters: {chars:,} | Tokens: {tokens:,}")
        else:
            print(f"FOMC directory not found: {fomc_in}")
    
    # Process speeches
    if args.what in ("all", "speeches"):
        print("\n=== Cleaning speeches ===")
        speeches_in = downloads_dir / "speeches"
        speeches_out = text_dir / "speeches"
        
        if speeches_in.exists():
            files, chars, tokens, counter, _ = process_tree(
                speeches_in, speeches_out,
                is_fomc=False,
                removal_guard=None
            )
            print(f"Processed {files} speech files -> {speeches_out}")
            print(f"  Characters: {chars:,} | Tokens: {tokens:,}")
        else:
            print(f"Speeches directory not found: {speeches_in}")
    
    # Save removed lines for analysis
    if args.save_removed and removed_lines:
        removed_path = text_dir / "removed_lines.txt"
        removed_path.write_text("\n".join(removed_lines), encoding="utf-8")
        print(f"\nRemoved lines saved to: {removed_path}")
    
    print("\nCleaning complete!")
    return 0


def cmd_init_config(args: argparse.Namespace, config: Config) -> int:
    """Initialize a config file with defaults."""
    output_path = Path(args.output) if args.output else DEFAULT_CONFIG_PATH
    
    if output_path.exists() and not args.force:
        print(f"Config file already exists: {output_path}")
        print("Use --force to overwrite")
        return 1
    
    save_config(Config(), output_path)
    print(f"Created config file: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fed-word2vec",
        description="Fed Word2Vec data pipeline - download and clean Federal Reserve documents",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config.yaml if exists)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download Fed documents")
    dl_parser.add_argument(
        "what",
        choices=["all", "fomc", "speeches"],
        default="all",
        nargs="?",
        help="What to download (default: all)",
    )
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean HTML to text")
    clean_parser.add_argument(
        "what",
        choices=["all", "fomc", "speeches"],
        default="all",
        nargs="?",
        help="What to clean (default: all)",
    )
    clean_parser.add_argument(
        "--save-removed",
        action="store_true",
        help="Save removed lines to file for analysis",
    )
    
    # Init config command
    init_parser = subparsers.add_parser("init", help="Create default config file")
    init_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for config file",
    )
    init_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing config file",
    )
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Load config
    config_path = args.config
    if config_path is None and DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    
    # Dispatch to command handler
    if args.command == "download":
        return cmd_download(args, config)
    elif args.command == "clean":
        return cmd_clean(args, config)
    elif args.command == "init":
        return cmd_init_config(args, config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
