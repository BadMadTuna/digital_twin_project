import sys
import importlib
import pkgutil
import TTS

print("🔍 Searching for 'DiscreteVAE' or 'dvae' modules in TTS...")

def find_dvae_class():
    # List of potential paths to check
    candidates = [
        "TTS.tts.layers.tortoise.dvae",
        "TTS.tts.models.xtts.dvae",
        "TTS.tts.models.tortoise.dvae",
        "TTS.vc.models.dvae",
        "TTS.tts.layers.dvae",
    ]
    
    for path in candidates:
        try:
            module = importlib.import_module(path)
            if hasattr(module, "DiscreteVAE"):
                print(f"✅ FOUND CLASS: DiscreteVAE in {path}")
                return path
            else:
                print(f"⚠️  Found module {path}, but it has no DiscreteVAE class.")
        except ImportError:
            print(f"❌ Could not find module: {path}")

    print("\n🔍 Deep searching submodules...")
    # Walk through TTS packages
    for importer, modname, ispkg in pkgutil.walk_packages(TTS.__path__, "TTS."):
        if "dvae" in modname:
            try:
                module = importlib.import_module(modname)
                if hasattr(module, "DiscreteVAE"):
                    print(f"🎉 SUCCESS! Found DiscreteVAE in: {modname}")
                    return modname
            except:
                pass

find_dvae_class()