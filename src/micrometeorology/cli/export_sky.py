"""CLI: publish the artifacts the sky page reads.

Two documents, both written into the site's ``Ceu/`` directory:

- ``kt_cumulative.json`` (``labmim-kt-cumulative-v1``) — the cumulative frequency
  of the clearness index with the four sky conditions of Escobedo et al. (2009)
  and the share of the record in each.
- ``ktkd.json`` (``labmim-ktkd-v1``) — the Kt-Kd plane: the two-dimensional
  density, the three published diffuse-fraction models scored against the
  measured Kd, and the hourly points.

Both derive from the same hourly database and the same solar geometry as the
climatology export, so a reader can put the three pages side by side.

The hourly database is indexed by naive station-local stamps, from the
datalogger's own clock; the solar geometry behind both documents takes its
offset from the pinned :data:`~micrometeorology.common.site.STATION_UTC_OFFSET_HOURS`
rather than the host's zone.

Examples
--------
::

    labmim-sky -i output/archive/station_hourly.parquet -o ../site-labmim/site/Ceu
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.common.site_json import write_json
from micrometeorology.stats import ktkd as ktkd_stats
from micrometeorology.stats.sky_condition import (
    KT_CUMULATIVE_EDGES,
    build_kt_cumulative_payload,
    sky_condition_summary,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

CUMULATIVE_FILENAME = "kt_cumulative.json"
KTKD_FILENAME = "ktkd.json"

#: Recorte ids, matching the climatology manifest's spelling so the two pages do
#: not name the same slice differently, and their captions, which travel in the
#: payload because ``Ceu/`` has no manifest beside it.
SUBSET_LABELS = {
    "observed_all": "Todo o registro",
    "observed_djf": "Verão (DJF)",
    "observed_jja": "Inverno (JJA)",
}

CUMULATIVE_CAVEATS = [
    "A curva é a soma corrida das barras do histograma de Kt: mesma seleção, mesmas arestas congeladas. Uma é a integral da outra por construção.",
    "As quatro condições de céu são as de Escobedo et al. (2009), aplicadas aos limites publicados de Kt (0,35 / 0,55 / 0,65) com a aresta superior FECHADA: um Kt exatamente em 0,35 é condição I, não II.",
    "A fração por classe vem calculada daqui e não é reobtida da curva: interpolar F(Kt) no cliente seria um segundo caminho numérico para o mesmo número publicado.",
    "n conta o recorte INTEIRO, as amostras fora das arestas incluídas, então F só chega a 1 quando nada caiu fora do eixo.",
    "Este Kt é a média HORÁRIA com a correção de ponto médio da BSRN no denominador extraterrestre, restrita às horas com o sol acima de 10°. Ele não é o Kt por quadro do manifesto da câmera, então estas frações não têm de bater com a distribuição de sky_class do conjunto multimodal.",
]

KTKD_CAVEATS = [
    "Kd é a fração difusa Hd/H — difusa sobre GLOBAL, não sobre a extraterrestre.",
    "Só Marques Filho et al. (2016) é função de Kt sozinho, por isso é o único desenhado como curva. Lemos et al. (2017) e o BRL de Ridley et al. (2010) leem também a hora solar aparente, a elevação, o Kt diário e a persistência: a um dado Kt eles preveem uma dispersão, resumida aqui como mediana e envelope p10-p90.",
    "Uma faixa com menos amostras que min_samples_per_bin publica median null. Decida por median null, nunca por n_per_bin zero: uma faixa suprimida quase sempre tem contagem diferente de zero.",
    "As métricas comparam cada modelo ao Kd MEDIDO no mesmo período e sob os mesmos filtros.",
]


def _seasonal_samples(kt: pd.Series) -> dict[str, np.ndarray]:
    """Split a clearness-index series into the recortes the page offers."""
    month = pd.DatetimeIndex(kt.index).month
    return {
        "observed_all": kt.to_numpy(),
        "observed_djf": kt[np.isin(month, (12, 1, 2))].to_numpy(),
        "observed_jja": kt[np.isin(month, (6, 7, 8))].to_numpy(),
    }


def build_payloads(hourly: pd.DataFrame, *, version: str) -> dict[str, Any]:
    """Build both sky artifacts from the hourly database.

    Parameters
    ----------
    hourly:
        The hourly database indexed by naive station-local stamps.
    version:
        Run stamp repeated in both documents, so a half-updated directory is
        detectable from the browser.

    Returns
    -------
    dict
        ``{filename: payload}`` for every artifact this command publishes.
    """
    prepared = ktkd_stats.prepare_ktkd(
        hourly, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    kt, kd = prepared.kt, prepared.kd
    if kt.empty:
        raise ValueError("no hour survived the gates; refusing to publish empty sky artifacts")

    # The clearness record is gated on the global channel alone: conditioning it
    # on the diffuse sensor's availability would publish a different population.
    clearness = ktkd_stats.prepare_clearness(
        hourly, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    cumulative = build_kt_cumulative_payload(
        _seasonal_samples(clearness), SUBSET_LABELS, version=version, caveats=CUMULATIVE_CAVEATS
    )

    predictors = (prepared.ast, prepared.elevation, prepared.daily_kt, prepared.psi)
    edges = np.asarray(KT_CUMULATIVE_EDGES, dtype=float)
    ktkd_payload = ktkd_stats.build_ktkd_payload(
        kt,
        kd,
        models={
            "marques_filho_2016": ktkd_stats.marques_filho_2016(kt.to_numpy()),
            "lemos_2017": ktkd_stats.lemos_2017(kt.to_numpy(), *predictors),
            "ridley_brl_2010": ktkd_stats.ridley_brl_2010(kt.to_numpy(), *predictors),
        },
        sky_conditions=sky_condition_summary(kt.to_numpy()),
        kt_edges=edges,
        kd_edges=edges,
        station={
            "name": "Estação Micrometeorológica LabMiM",
            "institution": "Instituto de Física — UFBA",
            "latitude": STATION_SITE.latitude,
            "longitude": STATION_SITE.longitude,
            "timezone": "America/Bahia",
        },
        sources=["station_hourly.parquet"],
        filters=prepared.filters,
        caveats=KTKD_CAVEATS,
        version=version,
    )
    return {CUMULATIVE_FILENAME: cumulative, KTKD_FILENAME: ktkd_payload}


@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Option("-i", "--input", help="Hourly database from labmim-archive.", exists=True),
    ],
    output_dir: Annotated[Path, typer.Option("-o", "--output", help="The site's Ceu/ directory.")],
) -> None:
    """Publish ``kt_cumulative.json`` and ``ktkd.json`` into *output_dir*."""
    setup_logging()
    version = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    hourly = pd.read_parquet(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, payload in build_payloads(hourly, version=version).items():
        path = write_json(output_dir / filename, payload)
        typer.echo(f"  [ok] {path.name}")

    typer.echo(f"\n>> 2 arquivos em {output_dir} (versão {version})")


def main() -> None:
    """Entry point for the ``labmim-sky`` command."""
    app()


if __name__ == "__main__":
    main()
