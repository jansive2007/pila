# Multi-Ghost Data Collection and Training

This system records multiple leaderboard ghost demonstrations independently and trains a single imitation policy from all runs.

## Folder Structure

```text
dataset/
  ghost_1/
    runs/
      run_0001/
        frames/
          000000.jpg
          000001.jpg
          ...
        actions.csv
  ghost_2/
    runs/
      run_0001/
        frames/
        actions.csv
  index.csv
```

## Controls (record_ghosts.py)

- **F1**: start recording current `ghost_id`
- **F2**: stop recording and save run
- **F3**: save/update `dataset/index.csv` (global manifest)
- **F4**: switch `ghost_id` (prompts for a new integer)
- **Esc**: exit safely

## Data Format

`actions.csv` columns:

- `timestamp`: seconds from run start
- `ghost_id`: integer id of ghost expert
- `run_id`: run folder id
- `frame_idx`: frame index in this run
- `frame_path`: relative path from `dataset/`
- `w`, `a`, `s`, `d`, `q`, `e`, `space`, `shift_ctrl`: binary labels

`index.csv` merges rows from all runs, useful as a global manifest.

## Usage

### 1) Record data

```bash
python record_ghosts.py --dataset-root dataset --fps 30 --ghost-id 1
```

Optional region capture:

```bash
python record_ghosts.py --region 0,0,1920,1080
```

### 2) Train with mixed ghost data

```bash
python train_multighost_example.py --dataset-root dataset --batch-size 32 --stack-size 4 --epochs 5
```

## PyTorch Dataset

Use `multi_ghost.dataset.MultiGhostDataset`:

- Loads all `ghost_*/runs/run_*/actions.csv`
- Optional `ghost_ids=[1,2]` filter
- Supports frame stacking (`stack_size`)
- Returns:
  - `frames`: float tensor `(stack_size*3, H, W)`
  - `actions`: float tensor `(8,)`
  - `meta`: dictionary with `ghost_id`, `run_id`, `last_frame`

When used with a `DataLoader(shuffle=True)`, samples are randomly mixed across ghosts and runs.
