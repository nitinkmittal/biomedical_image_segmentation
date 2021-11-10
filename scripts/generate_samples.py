import numpy as np
from pathlib import Path
from typing import Union, List, Any, Tuple, Optional
import concurrent.futures
import argparse
from time import time
import sys
import os


DEFAULT_NUM_WORKERS = 1
SEED = 40


def _check_and_convert_str2bool(v: str) -> bool:
    """Convert standard string boolean flags into pythonic boolean values.
    Args:
        v: string flag
    Returns:
        A boolean flag corresponding to it's string version
    """
    # https://stackoverflow.com/a/43357954/14916147
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def run_tiling(
    data_dir: Union[Path, str],
    tiles_dir: Union[Path, str],
    pad_tiles: bool,
    max_workers: Union[int, None],
    verbose: bool,
):
    """Run tiling operation over NumPy data files in parallel.
    Args:
        data_dir: directory containing NumPy data files
        tiles_dir: directory to store tiles
        pad_tiles: if True, tiles smaller than required shape are center padded,
            otherwise ignored
        max_workers: number of workers to execute tasks in parallel
        verbose: controls verbosity
    """
    fps = [
        os.path.join(data_dir, fp)
        for fp in os.listdir(data_dir)
        if is_specimen_data_fp(fp)
    ]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(_run_tiling, fp, tiles_dir, pad_tiles, verbose)
            for fp in fps
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                status = future.result()
            except Exception as e:
                print(f"Tiling failed for: {fps[i]}, Exception: {e}")
            else:
                if status:
                    print(f"Tiling success for: {fps[i]}")
                else:
                    print(f"Tiling failed for: {fps[i]}")


def _load_tiles_as_array(
    fps: List[Union[str, Path]],
    tile_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Load list/batch of tiles as an NumPy array.
    Args:
        fps: list of file pointers to tiles/NumPy data files
        tile_shape: shape of an individual tile
    Returns:
        4-D NumPy array containing a batch of tiles
    """
    arr_tiles = np.zeros((len(fps), *tile_shape))
    for i, fp in enumerate(fps):
        arr_tiles[i] = np.load(fp)
    return arr_tiles


def _run_embedding(fps) -> np.ndarray:
    """Generate embeddings from tiles.
    Args:
        fps: list of file pointers to tiles/NumPy data files
    Returns:
        A 2-D NumPy array containing a batch of embbeded vectors
    """
    tiles = _load_tiles_as_array(fps, tile_shape=TILE_SHAPE)
    embed = Embedder(
        image_shape=TILE_SHAPE, vector_size=VECTOR_SIZE, delay=0.0
    )
    return embed.encode(tiles)


def run_embedding(
    tiles_dir: Union[str, Path],
    embed_dir: Union[Path, str],
    max_workers: int,
    verbose: bool,
):
    """Generate and save embeddings from tiles in parallel.
    Args:
        tiles_dir: directory containing tiles as NumPy data files
        embed_dir: directory to store embeddings
        max_workers: number of workers to execute task in parallel
        verbose: controls verbosity
    """
    specimens = os.listdir(tiles_dir)

    batch_size = Embedder.MAX_TILES

    for specimen in specimens:
        # getting file pointers to available tiles
        fp_tiles = [
            os.path.join(tiles_dir, specimen, fp)
            for fp in os.listdir(os.path.join(tiles_dir, specimen))
            if fp.endswith(".npy")
        ]
        num_tiles = len(fp_tiles)

        # creating dir to store embeddings
        make_dirs(embed_dir, verbose=verbose)

        embeddings = np.zeros((num_tiles, VECTOR_SIZE))

        # generating embeddings for batches of tiles
        futures = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for i in range(0, num_tiles, batch_size):
                j = i + batch_size
                if j > num_tiles:
                    j = num_tiles
                futures[executor.submit(_run_embedding, fp_tiles[i:j])] = (
                    i,
                    j,
                )

            for future in concurrent.futures.as_completed(futures):
                i, j = futures[future]
                try:
                    embeddings[i:j] = future.result()
                except Exception as e:
                    print(
                        f"Embedding failed for: {fp_tiles[i:j]}, Exception: {e}"
                    )
                else:
                    print(f"Embedded {j-i} tiles from specimen: {specimen}")

        np.save(
            os.path.join(embed_dir, f"specimen={specimen}_embeddings.npy"),
            embeddings,
        )


def run_classification(
    embed_dir: Union[str, Path], label_dir: Union[str, Path], verbose: bool
):
    """Classify labels from embeddings.
    Args:
        embed_dir: directory containing embeddings from different specimens
            as NumPy data files
        label_dir: directory to store file with labels for each specimen
        verbose: controls verbosity
    """
    make_dirs(label_dir, verbose=verbose)
    classifier_obj = Classifier(delay=0, vector_size=VECTOR_SIZE)

    label_fp = os.path.join(label_dir, "labels.csv")

    def _extract_specimen(fp: Union[str, Path]):
        return os.path.basename(fp).split("=")[1].split("_")[0]

    for i, fp in enumerate(os.listdir(embed_dir)):
        embeddings = np.load(os.path.join(embed_dir, fp))
        label = classifier_obj.predict(embeddings)

        header = "specimen, label\n"
        line = f"{_extract_specimen(fp)}, {label[0]}\n"
        if i == 0:
            line = f"{header}{line}"
            mode = "w"
        else:
            mode = "a"
        write_file(fp=label_fp, msg=line, mode=mode, verbose=verbose)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument(
        "task",
        type=str,
        help=f"Name of the task, available tasks: {REGISTERED_TASK}",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory for storing sample data",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Output directory for storing tiles",
    )

    parser.add_argument(
        "--pad_tiles",
        "-p",
        help=(
            f"Boolean flag, If True, tiles with size smaller than: {TILE_SHAPE} are zero padded, "
            "else ignored, by default False"
        ),
        default=False,
        type=_check_and_convert_str2bool,
        required=False,
    )

    parser.add_argument(
        "--num_workers",
        "-n",
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of workers to run processes in parallel, "
            f"default: {DEFAULT_NUM_WORKERS}, "
            "can be set to -1 to determine optimal value automatically"
        ),
        type=int,
        required=False,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        help=f"Boolean flag, If True increases verbosity, by default False",
        default=False,
        type=_check_and_convert_str2bool,
        required=False,
    )

    args = parser.parse_args()

    if args.task not in REGISTERED_TASK:
        parser.error(f"Task: {args.task} not present")

    num_workers = args.num_workers
    if args.num_workers < 0:
        num_workers = None

    print(f"Running task: {args.task}")

    if args.task == "tiling":
        start_time = time()
        run_tiling(
            data_dir=args.data_dir,
            tiles_dir=args.out_dir,
            pad_tiles=args.pad_tiles,
            max_workers=num_workers,
            verbose=args.verbose,
        )
        print(f"Took {time()-start_time:.03f} seconds to complete")

    if args.task == "embed":
        start_time = time()

        run_tiling(
            data_dir=args.data_dir,
            tiles_dir="tiles",
            pad_tiles=args.pad_tiles,
            max_workers=num_workers,
            verbose=args.verbose,
        )

        run_embedding(
            tiles_dir="tiles",
            embed_dir=args.out_dir,
            max_workers=num_workers,
            verbose=args.verbose,
        )
        print(f"Took {time()-start_time:.03f} seconds to complete")

    if args.task == "classify":
        start_time = time()

        run_tiling(
            data_dir=args.data_dir,
            tiles_dir="tiles",
            pad_tiles=args.pad_tiles,
            max_workers=num_workers,
            verbose=args.verbose,
        )

        run_embedding(
            tiles_dir="tiles",
            embed_dir="embeds",
            max_workers=num_workers,
            verbose=args.verbose,
        )

        run_classification(
            embed_dir="embeds",
            label_dir=args.out_dir,
            verbose=args.verbose,
        )
        print(f"Took {time()-start_time:.03f} seconds to complete")
