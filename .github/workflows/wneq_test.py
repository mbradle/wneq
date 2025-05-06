import requests, io
import wnnet.nuc as wn
import wnnet.zones as wz
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


def test_nse():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content)
    )
    my_ye = 0.4

    eq = wq.equil.Equil(nuc)
    zone = eq.compute(5.0, 1.0e7, ye=my_ye)

    xsum = 0
    ye = 0
    for key, value in zone["mass fractions"].items():
        xsum += value
        ye += key[1] * value / key[2]

    assert np.isclose(1.0, xsum, atol=1.0e-8)
    assert np.isclose(my_ye, ye, atol=1.0e-8)


def test_qse():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content)
    )
    my_ye = 0.5
    my_yc = 0.01

    eq = wq.equil.Equil(nuc)
    zone = eq.compute(5.0, 1.0e8, ye=my_ye, clusters={"[z > 2]": my_yc})

    xsum = 0
    ye = 0
    y = {}
    for key, value in zone["mass fractions"].items():
        xsum += value
        y[key[0]] = value / key[2]
        ye += key[1] * y[key[0]]

    assert np.isclose(1.0, xsum, atol=1.0e-8)
    assert np.isclose(my_ye, ye, atol=1.0e-8)

    yc = 0
    for species in nuc.get_nuclides(nuc_xpath="[z > 2]"):
        if species in y:
            yc += y[species]

    assert np.isclose(my_yc, yc, atol=1.0e-8)


def test_eq_zone():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/m8erz/download").content)
    )

    eq = wq.equil.Equil(nuc)

    zone_data = wz.Zones(
        io.BytesIO(requests.get("https://osf.io/m8erz/download").content)
    )

    net_zones = zone_data.get_zones(zone_xpath="[position() <= 20]")
    net_list = list(net_zones.values())

    for i in range(len(net_list)):
        my_ye = 0
        for key, value in net_list[i]["mass fractions"].items():
            my_ye += key[1] * value / key[2]

        nse_zone = eq.compute_from_zone(net_list[i])
        eq.update_initial_guesses(
            {
                "n": nse_zone["properties"]["mun"],
                "p": nse_zone["properties"]["mup"],
            }
        )

        xsum = 0
        ye = 0
        for key, value in nse_zone["mass fractions"].items():
            xsum += value
            ye += key[1] * value / key[2]

        assert np.isclose(1.0, xsum, atol=1.0e-8)
        assert np.isclose(my_ye, ye, atol=1.0e-8)

    cluster_nucs = nuc.get_nuclides(nuc_xpath="[z > 2]")
    for i in range(len(net_list)):
        my_ye = 0
        my_yc = 0
        for key, value in net_list[i]["mass fractions"].items():
            my_ye += key[1] * value / key[2]
            if key[0] in cluster_nucs:
                my_yc += value / key[2]

        qse_zone = eq.compute_from_zone(
            net_list[i], clusters=["[z > 2]"]
        )
        eq.update_initial_guesses(
            {
                "n": qse_zone["properties"]["mun"],
                "p": qse_zone["properties"]["mup"],
                "[z > 2]": qse_zone["properties"][
                    ("cluster", "[z > 2]", "mu_kT")
                ],
            }
        )

        xsum = 0
        ye = 0
        yc = 0
        for key, value in qse_zone["mass fractions"].items():
            xsum += value
            ye += key[1] * value / key[2]
            if key[0] in cluster_nucs:
                yc += value / key[2]

        assert np.isclose(1.0, xsum, atol=1.0e-8)
        assert np.isclose(my_ye, ye, atol=1.0e-8)
        assert np.isclose(my_yc, yc, atol=1.0e-8)

def test_low_teq():
    nuc = wn.Nuc(
        io.BytesIO(requests.get("https://osf.io/grd4u/download").content)
    )

    eq = wq.equil.Equil(nuc)

    for ye in np.linspace(0, 1, 11):
        low_t_zone = eq.compute_low_temperature_nse(ye = ye)

        xsum = 0
        my_ye = 0
        for key, value in low_t_zone['mass fractions'].items():
            xsum += value
            my_ye += key[1] * value / key[2]

        assert np.isclose(1.0, xsum, atol=1.0e-8)
        assert np.isclose(my_ye, ye, atol=1.0e-8)

