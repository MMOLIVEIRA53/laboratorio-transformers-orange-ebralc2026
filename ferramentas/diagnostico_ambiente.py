"""Executar no mesmo interpretador usado pelo widget Python Script do Orange."""
import sys, platform, importlib.metadata as md
print('Python:', sys.version)
print('Executavel:', sys.executable)
print('Plataforma:', platform.platform())
for lib in ['Orange3', 'Orange3-Text', 'numpy', 'torch', 'transformers', 'matplotlib']:
    try: print(lib, md.version(lib))
    except md.PackageNotFoundError: print(lib, '[nao instalado]')
# Se houver Hugging Face Hub no ambiente, identifique a revisao resolvida
# com model_info(...).sha apenas se a rede e o acesso forem permitidos.
