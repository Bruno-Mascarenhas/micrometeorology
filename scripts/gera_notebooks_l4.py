"""Escreve os notebooks do Colab Enterprise para a L4: a fila sequencial e um por braco.

Os notebooks compartilham cada celula menos a que declara a fila e o prefixo do
arquivo: `05_fila_l4.ipynb` roda os tres bracos de 512 px em sequencia numa L4
so, e cada `06_l4_<braco>.ipynb` roda um braco sozinho, com prefixo proprio no
bucket, um por execucao. Toda a logica de fila mora
em `notebooks/colab/_colab_runner.py`, que tem teste; a celula do notebook so a
chama, porque codigo que so existe dentro de um `.ipynb` nao roda em CI e foi
assim que uma sessao de quatorze horas morreu com o resultado ainda na VM.

Usage
-----
::

    uv run python scripts/gera_notebooks_l4.py
"""

import json
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from allsky.config import ExperimentConfig, load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs/allsky/experiments/l4"
#: A fila do notebook 05: os tres bracos de 512 px, em sequencia numa L4.
ARMS = ("l4bloco512_s42", "l4v3res512_s44", "l4v3res512_s45")
#: Um notebook 06 por braco, para a L4 do Colab Enterprise: os tres de 512 px e a
#: familia de 1024 px (tres sementes e o controle a 512 px com a mesma receita).
ARMS_L4 = (
    *ARMS,
    "l4res1024_s42",
    "l4res1024_s43",
    "l4res1024_s44",
    "l4ctl512_s42",
    "l4res1024cos23_s42",
    "l4res1024cos23ord_s42",
)
#: Notebooks do Colab Pro+ (rota do Drive), mantidos para os bracos de 512 px.
ARMS_DRIVE = ARMS
#: Template de runtime do Colab Enterprise e a maquina que ele pede.
TEMPLATE = "labmim-l4"
MAQUINA = "g2-standard-8: 8 vCPU, 1x NVIDIA L4 24 GB"


def config_de(arm: str) -> ExperimentConfig:
    """O ExperimentConfig validado de *arm*, lido de ``configs/allsky/experiments/l4``.

    E dele que saem a raiz do dataset e o ``num_workers`` que o notebook cita: um
    valor copiado a mao para ca foi o que fez um notebook procurar um bundle que
    nao existia (2026-09-10).
    """
    return load_experiment_config(CONFIG_DIR / f"{arm}.yaml")


def dataset_de(arm: str) -> str:
    """Nome da raiz do dataset que o config de *arm* declara em ``data.data_root``."""
    return Path(config_de(arm).data.data_root).name


def bundle_de(dataset: str) -> str:
    """Nome do tar.gz que ``allsky export-colab-bundle`` escreve para *dataset*.

    A convencao do acervo e trocar o prefixo: ``dataset-iso-20260906`` viaja como
    ``bundle-iso-20260906.tar.gz``. Derivar em vez de repetir a mao evita o que
    aconteceu em 2026-09-10, quando o notebook procurou um
    ``bundle-dataset-iso-...`` que nunca existiu.
    """
    return f"bundle-{dataset.removeprefix('dataset-')}.tar.gz"


class Destino(StrEnum):
    """Onde a execucao arquiva: o bucket do Colab Enterprise ou o Drive do Colab Pro+."""

    BUCKET = "bucket"
    DRIVE = "drive"


NOTEBOOK_DIR = Path("notebooks/colab")
BUCKET = "gs://labmim-allsky-506901"
BRANCH = "condicao-do-ceu-multitarefa"


def _markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def _code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def _abertura(
    arms: tuple[str, ...],
    artifacts_suffix: str,
    destino: Destino = Destino.BUCKET,
    numero: str = "06",
) -> str:
    fila = "\n".join(
        f"{i + 1}. `configs/allsky/experiments/l4/{a}.yaml`" for i, a in enumerate(arms)
    )
    if destino is Destino.DRIVE:
        paralelo = (
            "Este notebook roda **um braco no Colab Pro+**, com o Drive como arquivo. E o caminho"
            " para treinar em paralelo com o Colab Enterprise: a cota de GPU do projeto GCP vale"
            " para o projeto inteiro (1 GPU), e a L4 da assinatura nao passa por ela."
        )
    elif len(arms) > 1:
        paralelo = "Este notebook roda os tres bracos **em sequencia** numa L4 so."
    else:
        paralelo = (
            "Este notebook roda **um braco** numa L4 so, com o seu prefixo no bucket, sem"
            " estado compartilhado com as execucoes dos outros notebooks."
        )
    return f"""
# {"05 — Fila na L4" if len(arms) > 1 else f"{numero} — {arms[0]} numa L4"}

{paralelo}

A fila:

{fila}

Os configs vivem no repositorio e sao rodados como estao: este notebook nao escreve YAML
derivado. O `amp` deles e `bf16`; a celula de hardware confere que a GPU atribuida tem
bfloat16 antes de baixar o bundle de dados.

## O que garante que nada se perde

Uma VM do Colab Enterprise e devolvida sem aviso e morre junto com o notebook: o que estiver
so no disco dela quando algo levantar excecao esta perdido, e numa fila de treino isso e
hora de GPU. Cinco coisas seguram isso, e todas tem teste em `tests/allsky/test_colab_runner.py`:

1. **Voo de teste antes da fila.** A secao 5 percorre, com dados sinteticos e em menos de um
   minuto, cada passo que roda fora do treino — pontuar por bloco no interpretador do venv,
   arquivar, espelhar ao vivo, retomar e escrever no destino final. Um `import` que so existe
   no venv, um caminho sem permissao ou um espelho apontando para lugar nenhum falham aqui,
   nao na decima quarta hora.
2. **Cada avaliacao e arquivada e espelhada assim que existe**, antes da proxima comecar e
   antes de qualquer pontuacao.
3. **Checkpoints viajam no espelho ao vivo** (`_live/<braco>/last.ckpt`), a cada 5 min. Uma
   execucao relancada restaura o checkpoint e continua com `allsky train --resume auto`.
4. **Nada depois do treino levanta.** Falha em pontuar vira coluna vazia e nota na linha;
   falha num braco nao derruba a fila.
5. **O relato sobrevive a sessao.** Cada passo vai para `fila_l4.log` no bucket.

## O que fica arquivado, e onde

{"Em `MyDrive/labmim/runs/allsky" + artifacts_suffix + "/` (o Drive e o arquivo: nao ha espelho a fazer, o que se escreve ali ja esta fora da VM):" if destino is Destino.DRIVE else "Em `" + BUCKET + "/runs/allsky" + artifacts_suffix + "/`:"}

- `<braco>/` — `metrics.csv`/`metrics.json` do treino, os relatorios `eval-test` (best),
  `eval-val` (best) e `eval-test-last` (last) com `predictions.parquet`, `stratified.csv`,
  `confusion.csv` e `report.md`, o YAML do config **e os checkpoints**.
- `_live/` — `metrics.csv`, `last.ckpt`, `best.ckpt` de cada braco e `heartbeat.json` com a
  GPU: e o que diz, de fora, se a VM esta treinando ou parada.
- `fila_l4.log`, `campanha_parcial.csv` a cada braco, `campanha.csv` e `campanha_resumo.json`
  no fechamento, `preflight.json` no comeco.

## Antes de rodar

{{PRE_REQUISITOS}}
"""


_PRE_BUCKET = """1. O bucket tem `colab/micrometeorology.bundle` (git bundle da branch),
   `allsky-mm/__BUNDLE__` e `dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth`.
2. Template `__TEMPLATE__` (__MAQUINA__), que casa com o
   `num_workers: __WORKERS__` do config.
3. Para trocar ou pular um braco com a execucao ja rodando: ponha `<braco>.yaml` ou
   `<braco>.skip` em `fila-l4/` no bucket."""

_PRE_DRIVE = """1. **Runtime -> Change runtime type -> L4 GPU** (ou A100). Os configs declaram
   `amp: bf16`, entao uma T4 para na celula de hardware, antes de desempacotar 1,4 GB.
2. O Drive tem `MyDrive/labmim/allsky-mm/bundle-iso-20260906.tar.gz` e
   `MyDrive/labmim/dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth`.
3. Rode as celulas em ordem. A de ambiente leva uns 8 min (venv 3.14 mais torch CUDA) e a
   fila de 8 a 10 h: deixe a aba aberta, com execucao em segundo plano ligada.
4. Para pular o braco com a sessao ja rodando: crie `MyDrive/labmim/fila-l4/<braco>.skip`."""


_RUNTIME = r"""
import subprocess
import time

SESSION_START = time.time()
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
"""

_CLONE_BUCKET = r"""
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Bruno-Mascarenhas/micrometeorology.git"
BRANCH = "__BRANCH__"
ON_VERTEX = os.environ.get("VERTEX_PRODUCT") == "COLAB_ENTERPRISE" or not os.path.isdir("/content")
BUCKET = "__BUCKET__"
BASE = str(Path.home()) if ON_VERTEX else "/content"
WORKDIR = f"{BASE}/micrometeorology"

if not os.path.exists(WORKDIR):
    if ON_VERTEX:
        bundle = f"{BASE}/micrometeorology.bundle"
        subprocess.run(["gcloud", "storage", "cp", f"{BUCKET}/colab/micrometeorology.bundle", bundle], check=True)
        subprocess.run(["git", "clone", "-b", BRANCH, bundle, WORKDIR], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, WORKDIR], check=True)
if not os.path.isdir(f"{WORKDIR}/configs/allsky/experiments/l4"):
    raise RuntimeError(f"a branch {BRANCH} do bundle nao carrega configs/allsky/experiments/l4/ — refaca o bundle")
"""

_CLONE_DRIVE = r"""
import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
BRANCH = "__BRANCH__"
STORE = "/content/drive/MyDrive/labmim"
BASE = "/content"
WORKDIR = f"{BASE}/micrometeorology"

# O repositorio vem do git bundle no Drive, nao do GitHub: a branch desta campanha
# e local e nunca foi publicada.
BUNDLE_GIT = f"{STORE}/colab/micrometeorology.bundle"
if not os.path.exists(BUNDLE_GIT):
    raise RuntimeError(f"{BUNDLE_GIT} nao existe no Drive — suba o git bundle antes de rodar")
if not os.path.exists(WORKDIR):
    subprocess.run(["git", "clone", "-b", BRANCH, BUNDLE_GIT, WORKDIR], check=True)
if not os.path.isdir(f"{WORKDIR}/configs/allsky/experiments/l4"):
    raise RuntimeError(f"o bundle de {BRANCH} nao carrega configs/allsky/experiments/l4/ — refaca o bundle")
"""

#: O que os dois ambientes compartilham depois do clone: uv, torch CUDA, DINOv3 e
#: a verificacao do runner. As tres marcas sao o que muda entre bucket e Drive.
_INSTALL = r"""subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "python", "install", "3.14"], cwd=WORKDIR, check=True)
subprocess.run(["uv", "venv", "--python", "3.14", ".venv"], cwd=WORKDIR, check=True)
subprocess.run(["uv", "sync", "--locked", "--extra", "allsky"], cwd=WORKDIR, check=True)
subprocess.run(
    ["uv", "pip", "install", "--python", ".venv/bin/python", "--reinstall", "--torch-backend", "auto", "torch==2.13.0"],
    cwd=WORKDIR,
    check=True,
)

PY = f"{WORKDIR}/.venv/bin/python"
os.environ["PATH"] = f"{WORKDIR}/.venv/bin:" + os.environ["PATH"]

__DINOV3_NOTA__
DINOV3_REPO_DIR = f"{BASE}/dinov3"
if not os.path.exists(DINOV3_REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov3", DINOV3_REPO_DIR], check=True)
os.environ["ALLSKY_DINOV3_REPO"] = DINOV3_REPO_DIR
sys.path.insert(0, f"{WORKDIR}/notebooks/colab")

verify = subprocess.run([PY, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"], capture_output=True, text=True, check=False)
print(verify.stdout)
if "True" not in verify.stdout:
    raise RuntimeError("__SEM_CUDA__")

import _colab_runner as runner  # noqa: E402

lacking = [name for name in ("preflight", "run_arm", "pull_live_run", "score_by_sensor_block_in") if not hasattr(runner, name)]
if lacking:
    raise RuntimeError(f"o _colab_runner de {BRANCH} nao tem {lacking} — __SEM_RUNNER__")
"""


def _ambiente(destino: Destino) -> str:
    """A celula de ambiente de *destino*: o clone proprio mais a instalacao compartilhada."""
    if destino is Destino.DRIVE:
        return _CLONE_DRIVE + _INSTALL.replace("__DINOV3_NOTA__\n", "").replace(
            "__SEM_CUDA__",
            "torch sem CUDA — Runtime > Change runtime type > GPU, e rode esta celula de novo",
        ).replace("__SEM_RUNNER__", "refaca o bundle de uma branch que os carregue")
    nota = (
        "# O DINOv3 nao vem pelo torch.hub (o hubconf arrasta torchmetrics/omegaconf/submitit):\n"
        "# o pacote importa hub/backbones.py direto do clone, apontado por ALLSKY_DINOV3_REPO.\n"
    )
    return _CLONE_BUCKET + _INSTALL.replace("__DINOV3_NOTA__\n", nota).replace(
        "__SEM_CUDA__", "torch sem CUDA — pare e reinstale antes de treinar"
    ).replace("__SEM_RUNNER__", "aponte BRANCH para uma branch que os carregue")


_HARDWARE = r"""
import json

probe = subprocess.run(
    [PY, "-c", 'import json, sys; sys.path.insert(0, "' + WORKDIR + '/notebooks/colab"); import _colab_runner as r; print(json.dumps(r.probe_accelerator()))'],
    capture_output=True,
    text=True,
    check=True,
)
HW = json.loads(probe.stdout.strip().splitlines()[-1])
print(HW)
if HW["amp_dtype"] != "bf16":
    raise RuntimeError(f"{HW['name']} sem bfloat16: os configs de l4/ declaram amp bf16 — peca o template __TEMPLATE__")
if HW["cpus"] < __WORKERS__:
    print(f"ATENCAO: {HW['cpus']} vCPU para num_workers: __WORKERS__ do config — o loader vai disputar CPU")
"""

_DADOS_BUCKET = r"""
import os
from pathlib import Path

if not ON_VERTEX:
    raise RuntimeError("este notebook so roda no Colab Enterprise: dado e arquivo vivem no bucket, nao ha Drive")

STORE = f"{BASE}/labmim"
BUNDLE_REL = "allsky-mm/__BUNDLE__"
WEIGHTS_REL = "dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth"
for rel in (BUNDLE_REL, WEIGHTS_REL):
    os.makedirs(f"{STORE}/{os.path.dirname(rel)}", exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", "-n", f"{BUCKET}/{rel}", f"{STORE}/{rel}"], check=True)

BUNDLE = f"{STORE}/{BUNDLE_REL}"
DATA = f"{BASE}/allsky-mm"
ARTIFACTS = f"{STORE}/runs/allsky__SUFFIX__"
REMOTE_ARTIFACTS = f"{BUCKET}/runs/allsky__SUFFIX__"
OVERRIDES = f"{STORE}/fila-l4"
REMOTE_OVERRIDES = f"{BUCKET}/fila-l4"
os.environ["ALLSKY_DINOV3_WEIGHTS"] = f"{STORE}/{WEIGHTS_REL}"

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(OVERRIDES, exist_ok=True)
listing = subprocess.run(["gcloud", "storage", "ls", REMOTE_ARTIFACTS], capture_output=True, text=True, check=False)
pulled = runner.mirror_once([(REMOTE_ARTIFACTS, ARTIFACTS)])
if pulled and listing.returncode == 0:
    raise RuntimeError(f"{REMOTE_ARTIFACTS} existe mas o rsync de volta falhou: sem o arquivo a fila retreinaria tudo")
print("arquivo trazido do bucket:", "ok" if listing.returncode == 0 else "prefixo ainda nao existe, primeira execucao")
print("ja arquivado:", sorted(p.name for p in Path(ARTIFACTS).iterdir()) or "nada")
MIRROR = [(ARTIFACTS, REMOTE_ARTIFACTS)]
print("espelho inicial:", runner.mirror_once(MIRROR) or "ok")

"""

_DADOS_DRIVE_HEAD = r"""
BUNDLE = f"{STORE}/allsky-mm/__BUNDLE__"
WEIGHTS = f"{STORE}/dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth"
for caminho in (BUNDLE, WEIGHTS):
    if not os.path.exists(caminho):
        raise RuntimeError(f"{caminho} nao existe no Drive — suba antes de rodar")
os.environ["ALLSKY_DINOV3_WEIGHTS"] = WEIGHTS

DATA = "/content/allsky-mm"
ARTIFACTS = f"{STORE}/runs/allsky__SUFFIX__"
OVERRIDES = f"{STORE}/fila-l4"
REMOTE_OVERRIDES = None
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(OVERRIDES, exist_ok=True)
# O Drive montado e o proprio arquivo: o que se escreve ali ja esta fora da VM,
# entao nao ha destino remoto a espelhar.
MIRROR = []
print("ja arquivado:", sorted(p.name for p in Path(ARTIFACTS).iterdir()) or "nada")

"""

#: Do bundle em diante os dois destinos fazem o mesmo: staging, link e diretorio de runs.
_DADOS_TAIL = r"""ROOT = runner.stage_bundle(BUNDLE, DATA, python=PY)
for required in ("manifest.parquet", "splits.json", "frames"):
    if not (Path(ROOT) / required).exists():
        raise RuntimeError(f"{ROOT} sem {required}: o bundle nao e o __DATASET__ com frames")

DATASET_LINK = Path(WORKDIR) / "output/allsky-mm/__DATASET__"
DATASET_LINK.parent.mkdir(parents=True, exist_ok=True)
if DATASET_LINK.is_symlink():
    DATASET_LINK.unlink()
elif DATASET_LINK.exists():
    raise RuntimeError(f"{DATASET_LINK} existe e nao e um link: nao vou sobrescrever")
DATASET_LINK.symlink_to(ROOT, target_is_directory=True)
os.chdir(WORKDIR)
OUT = Path(WORKDIR) / "output/allsky-mm/experiments/l4"
OUT.mkdir(parents=True, exist_ok=True)
print(f"{DATASET_LINK} -> {os.readlink(DATASET_LINK)}; cwd {os.getcwd()}; runs em {OUT}")
"""


def _dados(destino: Destino) -> str:
    """A celula de dados de *destino*: a origem propria mais o staging compartilhado."""
    head = _DADOS_DRIVE_HEAD if destino is Destino.DRIVE else _DADOS_BUCKET
    return head + _DADOS_TAIL


_PREFLIGHT = r"""
for check in runner.preflight(PY, artifacts=ARTIFACTS, mirror=MIRROR, work_dir=BASE):
    print("  ok:", check)
WATCHERS = [runner.start_live_sync(OUT, Path(ARTIFACTS) / runner.LIVE_DIR), runner.start_mirror(MIRROR)]
"""

_FILA = r'''
from pathlib import Path

import pandas as pd

FILA = __FILA__
missing = [rel for rel in FILA if not (Path(WORKDIR) / rel).is_file()]
if missing:
    raise RuntimeError(f"{missing}: a branch {BRANCH} do bundle nao carrega os configs de l4/")

LOG = Path(ARTIFACTS) / "fila_l4.log"
rows = []


def log(line):
    """Print *line* and append it, timestamped, to the log the mirror carries."""
    stamped = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {line}"
    print(stamped)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


for entry in FILA:
    started = time.time()
    row = runner.run_arm(
        Path(WORKDIR) / entry,
        python=PY,
        out_dir=OUT,
        artifacts=Path(ARTIFACTS),
        mirror=MIRROR,
        overrides=Path(OVERRIDES),
        override_mirror=[(REMOTE_OVERRIDES, OVERRIDES)] if REMOTE_OVERRIDES else [],
        watchers=WATCHERS,
        log=log,
    )
    rows.append(row)
    pd.DataFrame(rows).to_csv(f"{ARTIFACTS}/campanha_parcial.csv", index=False)
    log(runner.summarise_arm(row))
    log(f"  {(time.time() - started) / 3600:.1f} h nesta entrada; {(time.time() - SESSION_START) / 3600:.1f} h de sessao")
    runner.mirror_once(MIRROR)
'''

_FECHAMENTO = r"""
frame = pd.DataFrame(rows)
frame.to_csv(f"{ARTIFACTS}/campanha.csv", index=False)
with open(f"{ARTIFACTS}/campanha_resumo.json", "w") as handle:
    json.dump(
        {"hardware": HW, "fila": FILA, "n_runs": len(rows), "session_hours": round((time.time() - SESSION_START) / 3600, 2), "rows": rows},
        handle,
        indent=2,
        default=str,
    )
for row in rows:
    print(runner.summarise_arm(row))
print(frame.to_string())
if MIRROR:
    print("espelho final:", runner.mirror_once(MIRROR) or "ok")
print("artefatos em", ARTIFACTS)
"""


def build(
    arms: tuple[str, ...],
    *,
    suffix: str,
    slug: str,
    destino: Destino = Destino.BUCKET,
    numero: str = "06",
) -> dict[str, object]:
    """One notebook: the shared cells plus the queue this one owns.

    *destino* picks where the run is archived: ``bucket`` for Colab Enterprise,
    which mirrors to Cloud Storage, or ``drive`` for Colab Pro+, where the
    mounted Drive is itself the archive and there is nothing to mirror. *suffix*
    names the archive prefix under ``runs/allsky`` and *slug* the cell ids.
    """
    fila = [f"configs/allsky/experiments/l4/{arm}.yaml" for arm in arms]
    datasets = {dataset_de(arm) for arm in arms}
    if len(datasets) != 1:
        raise ValueError(f"uma fila so pode ter um dataset, e {arms} declaram {sorted(datasets)}")
    dataset = datasets.pop()
    workers = str(config_de(arms[0]).train.num_workers)
    pre_requisitos = (
        _PRE_DRIVE
        if destino is Destino.DRIVE
        else _PRE_BUCKET.replace("__BUNDLE__", bundle_de(dataset))
        .replace("__TEMPLATE__", TEMPLATE)
        .replace("__MAQUINA__", MAQUINA)
        .replace("__WORKERS__", workers)
    )
    cells = [
        _markdown(
            _abertura(arms, suffix, destino, numero).replace("{PRE_REQUISITOS}", pre_requisitos)
        ),
        _markdown("## 1. Runtime e GPU"),
        _code(_RUNTIME),
        _markdown(
            "## 2. Ambiente\n\nClona o repositorio onde o `_colab_runner` mora, instala o torch CUDA pelo backend que o\n"
            "driver da VM pede e verifica. O repositorio vem de um `git bundle`"
            + (" no Drive" if destino is Destino.DRIVE else " no bucket")
            + ": a branch desta campanha e local e nunca foi publicada."
        ),
        _code(_ambiente(destino).replace("__BRANCH__", BRANCH).replace("__BUCKET__", BUCKET)),
        _markdown(
            "## 3. Hardware\n\nO probe roda no interpretador do venv. Os configs declaram `amp: bf16`, entao uma GPU sem\n"
            "bfloat16 (T4, Turing) para aqui — antes de baixar o bundle de dados."
        ),
        _code(_HARDWARE.replace("__TEMPLATE__", TEMPLATE).replace("__WORKERS__", workers)),
        _markdown(
            "## 4. Dados e artefatos\n\nSo bucket. O que sai da VM sai por `ARTIFACTS`, espelhado para o prefixo deste notebook;\n"
            "`fila-l4/` e lido do bucket antes de cada braco e **nunca** espelhado de volta, porque o `rsync`\n"
            "nao tem direcao e a copia antiga da VM sobrescreveria o arquivo posto la de fora."
        ),
        _code(
            _dados(destino)
            .replace("__SUFFIX__", suffix)
            .replace("__DATASET__", dataset)
            .replace("__BUNDLE__", bundle_de(dataset))
        ),
        _markdown(
            "## 5. Voo de teste\n\nPercorre, com dados sinteticos, cada passo que roda fora do treino, e escreve\n"
            "`preflight.json` no destino final. E o que transforma uma quebra de quatorze horas numa de um\n"
            "minuto. Em seguida arma o espelho ao vivo (checkpoints inclusos) e o espelho para o bucket."
        ),
        _code(_PREFLIGHT),
        _markdown(
            "## 6. A fila\n\nCada entrada passa por `runner.run_arm`, que tem teste: override do bucket, retomada do\n"
            "checkpoint espelhado, treino, as tres avaliacoes **arquivando e espelhando cada uma assim que\n"
            "existe**, e so entao a pontuacao por bloco, no interpretador do venv e dentro de um `try`.\n"
            "Nada aqui levanta: o estado de cada braco vira uma linha da tabela."
        ),
        _code(_FILA.replace("__FILA__", json.dumps(fila, indent=4))),
        _markdown("## 7. Fechamento\n\nGrava o indice da campanha e espelha uma ultima vez."),
        _code(_FECHAMENTO),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"{slug}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "gpuType": "L4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the queue notebook, one notebook per arm for the bucket, and one per arm for Drive.

    The notebooks are formatted here with the repository's own formatter, so
    regenerating them produces no diff against what pre-commit would write.
    """
    written = []
    for path, arms, suffix, slug, destino, numero in [
        (NOTEBOOK_DIR / "05_fila_l4.ipynb", ARMS, "-l4", "l4-fila", Destino.BUCKET, "05"),
        *[
            (
                NOTEBOOK_DIR / f"06_l4_{arm}.ipynb",
                (arm,),
                f"-l4-{arm}",
                f"l4-{arm}",
                Destino.BUCKET,
                "06",
            )
            for arm in ARMS_L4
        ],
        *[
            (
                NOTEBOOK_DIR / f"07_prop_{arm}.ipynb",
                (arm,),
                f"-l4-{arm}",
                f"l4-{arm}-drive",
                Destino.DRIVE,
                "07",
            )
            for arm in ARMS_DRIVE
        ],
    ]:
        path.write_text(
            json.dumps(
                build(arms, suffix=suffix, slug=slug, destino=destino, numero=numero),
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(path)
    formatter = shutil.which("ruff")
    if formatter is not None:
        subprocess.run(  # noqa: S603 — lista literal, e o ruff e o formatador do repositorio
            [formatter, "format", "--quiet", *[str(path) for path in written]], check=False
        )
    for path in written:
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
