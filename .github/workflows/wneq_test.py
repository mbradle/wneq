import requests, io
import wnnet.nuc as wn
import wneq as wq
import numpy as np


def test_wneq():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content)
    )

    ng = wq.Ng(nuc)

    t9 = 1.0
    rho = 1.0e5
    yz = {}
    yz[50] = 0.005
    result = ng.compute_with_root(t9, rho, yz)

    assert 0 < result["mass fractions"][("n", 0, 1)] < 1
    assert float(result["properties"]["mun_kt"]) < 0


def test_zone():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content)
    )

    ng = wq.Ng(nuc)

    properties = {}
    properties["t9"] = 2
    properties["rho"] = 1.0e6

    mass_fractions = {}
    x1 = 0.02
    x2 = 0.03
    mass_fractions[("zr90", 40, 90)] = x1
    mass_fractions[("sn120", 50, 120)] = x2

    zone = {}
    zone["properties"] = properties
    zone["mass fractions"] = mass_fractions

    result = ng.compute_with_root_from_zone(zone)

    yz = np.zeros([100])

    for key, value in result["mass fractions"].items():
        yz[key[1]] += value / key[2]

    assert abs(yz[40] - x1 / 90.0) < 1.0e-15
    assert abs(yz[50] - x2 / 120.0) < 1.0e-15

def test_qse():
    nuc = wn.Nuc(io.BytesIO(requests.get("https://osf.io/kyhbs/download").content))

    eq = wq.equil.Equil(nuc)
    zone = eq.compute(5.0, 1.0e8, ye=0.5, clusters={"[z > 2]": 0.01})

    xsum = 0
    for value in zone['mass fractions'].values():
        xsum += value

    assert np.isclose(1., xsum, atol = 1.e-8)
