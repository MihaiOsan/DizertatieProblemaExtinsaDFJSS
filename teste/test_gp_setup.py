import gpSetup as gp

def test_protected_div():
    assert gp.protected_div(6, 2) == 3         # normal
    assert gp.protected_div(5, 0) == 1e9       # +∞ safe
    assert gp.protected_div(-2, 0) == -1e9     # -∞ safe
    assert gp.protected_div(0, 0) == 0.0       # 0/0 → 0
    assert gp.protected_div(2, 1e-10) == 1e9   # near‑zero denom

def test_erc_range():
    for _ in range(50):
        v = gp.generate_random_value_for_erc()
        assert -5 <= v <= 5                    # în interval
        assert round(v, 2) == v                # 2 zecimale

def test_toolbox_structure_and_compile():
    tb = gp.create_toolbox(np=0)               # fără threads
    pset = tb.pset

    # 12 arg. şi ultimele două sunt cele legate de ETPC
    assert len(pset.arguments) == 12
    assert pset.arguments[-2:] == ["ETPC_D", "N_ETPC_S"]

    # "protected_div" trebuie să fie printre primitive indiferent de tipul‑cheie
    assert any(p.name == "protected_div" for prims in pset.primitives.values() for p in prims)

    # Compilăm un individ şi verificăm că funcţia rezultată rulează pe 12 intrări
    ind = tb.individual()
    fn = tb.compile(expr=ind)
    assert isinstance(fn(*range(12)), (int, float))
