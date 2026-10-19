"""
Recompila ``studio/device/injector/igp-injector.jar`` a partir de ``TouchInjector.java``.

Requer JDK (javac/jar) e o Android SDK com uma plataforma (android.jar) e
build-tools (d8). O .jar gerado já vem versionado no projeto; só é preciso
rodar este script ao alterar o código Java.

Uso: python scripts/build_injector.py [--sdk CAMINHO_DO_SDK]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJECTOR_DIR = os.path.join(ROOT, "studio", "device", "injector")


def latest(pattern):
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(f"Não encontrado: {pattern}")
    return found[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sdk", default=os.environ.get("ANDROID_HOME")
                        or os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk"))
    args = parser.parse_args()

    android_jar = latest(os.path.join(args.sdk, "platforms", "android-*", "android.jar"))
    d8 = latest(os.path.join(args.sdk, "build-tools", "*", "d8.bat" if os.name == "nt" else "d8"))
    with tempfile.TemporaryDirectory() as tmp:
        classes = os.path.join(tmp, "classes")
        out = os.path.join(tmp, "out")
        os.makedirs(classes)
        os.makedirs(out)
        subprocess.run(["javac", "-source", "8", "-target", "8", "-bootclasspath", android_jar,
                        "-d", classes, os.path.join(INJECTOR_DIR, "TouchInjector.java")], check=True)
        class_files = glob.glob(os.path.join(classes, "igp", "*.class"))
        subprocess.run([d8, "--min-api", "26", "--output", out, *class_files], check=True)
        jar_path = os.path.join(tmp, "igp-injector.jar")
        subprocess.run(["jar", "cf", jar_path, "-C", out, "classes.dex"], check=True)
        shutil.copyfile(jar_path, os.path.join(INJECTOR_DIR, "igp-injector.jar"))
    print("igp-injector.jar atualizado em", INJECTOR_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
