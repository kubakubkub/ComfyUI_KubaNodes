r"""Settings in kubakub.ini [settings] with plain names; the old environment variables still work as a fallback.

Run: python_embeded\python.exe custom_nodes\ComfyUI_KubaNodes\tests\test_settings.py
"""

import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kub_settings_test", os.path.join(os.path.dirname(HERE), "settings.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def main():
    real = st.INI
    with tempfile.TemporaryDirectory() as d:
        st.INI = os.path.join(d, "kubakub.ini")
        for k in ("KUBA_CACHE_GB", "KUBA_CACHE", "KUBA_BLENDER_WORKER"):
            os.environ.pop(k, None)
        assert st.number("cache_gb", 20) == 20.0                         # no file, no env: the default
        os.environ["KUBA_CACHE_GB"] = "7"
        assert st.number("cache_gb", 20) == 7.0                          # the old variable still works
        with open(st.INI, "w", encoding="utf-8") as f:
            f.write("[menu]\nlab = off   # keep\n\n[settings]\n# cache_gb = 99\ncache_gb = 3  # small\n")
        assert st.number("cache_gb", 20) == 3.0                          # the ini wins over the old variable
        os.environ["KUBA_CACHE"] = "somewhere"
        assert st.get("cache_folder") == "somewhere"                     # old name mapped to the new key
        assert st.switch("blender_worker", True) is True
        os.environ["KUBA_BLENDER_WORKER"] = "0"
        assert st.switch("blender_worker", True) is False
        st.save("blender", r"C:\Program Files\Blender 100%\blender.exe")
        st.save("cache_gb", "5")
        text = open(st.INI, encoding="utf-8").read()
        assert "lab = off   # keep" in text and "# cache_gb = 99" in text   # comments and other lines kept
        assert st.get("blender") == r"C:\Program Files\Blender 100%\blender.exe"
        assert st.number("cache_gb", 20) == 5.0 and text.count("cache_gb = 5") == 1
        os.remove(st.INI)
        st.save("blender_idle", "30")
        assert st.number("blender_idle", 600) == 30.0
        for k in ("KUBA_CACHE_GB", "KUBA_CACHE", "KUBA_BLENDER_WORKER"):
            os.environ.pop(k, None)
    st.INI = real
    assert not os.path.isfile(real) or "Blender 100%" not in open(real, encoding="utf-8").read()   # never the real file
    print("ALL OK")


if __name__ == "__main__":
    main()
