"""Download pinned Eedi CSVs. Junyi Info_Content is local; optional exercise table is not fetched here."""
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def main():
    manifest = json.loads((DATA_DIR / 'sources.json').read_text())
    destination = DATA_DIR / 'raw' / 'eedi'
    destination.mkdir(parents=True, exist_ok=True)
    for name, metadata in manifest['files'].items():
        target = destination / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == metadata['sha256']:
            print(f'Verified cached {name}')
            continue
        with urlopen(metadata['url'], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != metadata['sha256']:
            raise ValueError(f'Checksum mismatch: {name}; existing file unchanged')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'Downloaded {name}: {len(content)} bytes')
    junyi_info = DATA_DIR / 'junyi' / 'archive' / 'Info_Content.csv'
    junyi_table = DATA_DIR / 'raw' / 'junyi' / 'junyi_Exercise_table.csv'
    if junyi_table.is_file():
        print(f'Junyi expert table present: {junyi_table}')
    elif junyi_info.is_file():
        print(f'Junyi hierarchy present: {junyi_info}')
    else:
        print('Junyi graph files missing. See data/README.md.')


if __name__ == '__main__':
    main()
