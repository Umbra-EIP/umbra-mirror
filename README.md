# Data Version Control (DVC) Usage Guide

This project uses **DVC (Data Version Control)** to manage datasets and model files efficiently. DVC allows versioning large files, sharing data between collaborators, and keeping your Git repository lightweight.

---

## ⚙️ Prerequisites

DVC is already configured for this repository.
Make sure you have DVC installed locally:

```bash
dvc --version
```

If DVC is not installed -> pip install requirements.txt

---

## 📁 Adding Data to DVC

To track a new dataset or model file with DVC:

```bash
dvc add path/to/your/data
```

What this does:
- Creates a `.dvc` file (for example, `data.csv.dvc`) that points to the large file.
- Adds the actual large file to the DVC cache and ignores it in Git.

Then commit the generated `.dvc` file and updated `.gitignore`:

```bash
git add path/to/your/data.dvc .gitignore
git commit -m "Track dataset with DVC"
```

---

## ☁️ Pushing Data to Remote Storage

To upload tracked files from your local cache to the remote storage (already configured for this repo):

```bash
dvc push
```

Notes:
- `dvc push` uploads only DVC-tracked files.
- The remote configuration is already set up, so you don’t need extra flags.

To push a specific file:

```bash
dvc push path/to/your/data.dvc
```

---

## ⬇️ Pulling Data from Remote

When you clone the repository or switch to a new branch that references data you don’t have locally:

```bash
dvc pull
```

This downloads all DVC-tracked files needed for the current Git commit.

To pull a specific file:

```bash
dvc pull path/to/your/data.dvc
```

---

## 🔁 Updating Data Versions

If you modify a tracked dataset or model:

1. Re-add it with DVC
   ```bash
   dvc add path/to/your/data
   ```
2. Commit the updated `.dvc` file
   ```bash
   git add path/to/your/data.dvc
   git commit -m "Update dataset"
   ```
3. Push the new version
   ```bash
   dvc push
   ```
